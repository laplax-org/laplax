"""Posterior covariance functions for various curvature estimates."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger

from laplax.curv.lanczos import lanczos_lowrank
from laplax.curv.lobpcg import lobpcg_lowrank
from laplax.curv.utils import LowRankTerms
from laplax.enums import CurvApprox, LowRankMethod
from laplax.types import (
    Array,
    Callable,
    CurvatureKeyType,
    CurvatureMV,
    FlatParams,
    Float,
    Layout,
    Num,
    PosteriorState,
    PriorArguments,
    PyTree,
)
from laplax.util.flatten import (
    create_pytree_flattener,
    wrap_factory,
    wrap_function,
)
from laplax.util.mv import diagonal, to_dense
from laplax.util.tree import get_size

# -----------------------------------------------------------------------
# FULL
# -----------------------------------------------------------------------


def create_full_curvature(
    mv: CurvatureMV, layout: Layout, **kwargs
) -> Num[Array, "P P"]:
    """Generate a full curvature approximation.

    The curvature is densed and flattened into a 2D array, that corresponds to the
    flattened parameter layout.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        **kwargs: Additional arguments (unused).

    Returns:
        A dense matrix representing the full curvature approximation.
    """
    del kwargs
    if isinstance(layout, int):
        msg = (
            "Full curvature assumes parameter dictionary as input, "
            f"got type {type(layout)} instead. Proceeding without wrapper."
        )
        logger.warning(msg)
        mv_wrapped = mv
    else:
        flatten, unflatten = create_pytree_flattener(layout)
        mv_wrapped = wrap_function(mv, input_fn=unflatten, output_fn=flatten)
    curv_estimate = to_dense(mv_wrapped, layout=get_size(layout))
    return curv_estimate


def full_curvature_to_precision(
    curv_estimate: Num[Array, "P P"],
    prior_arguments: PriorArguments,
    loss_scaling_factor: Float = 1.0,
) -> Num[Array, "P P"]:
    """Add prior precision to the curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the curvature estimate.

    Args:
        curv_estimate: Full curvature estimate matrix.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.
        loss_scaling_factor: Factor by which the user-provided loss function is
            scaled. Defaults to 1.0.

    Returns:
        Updated curvature matrix with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    # sigma = prior_arguments["sigma"] if "sigma" in prior_arguments else sigma
    sigma = prior_arguments.get("sigma", 1.0)

    # prior_arguments["sigma"] if "sigma" in prior_arguments else sigma
    return (
        sigma * curv_estimate + prior_prec * jnp.eye(curv_estimate.shape[-1])
    ) / loss_scaling_factor


def full_prec_to_scale(prec: Num[Array, "P P"]) -> Num[Array, "P P"]:
    """Convert precision matrix to scale matrix using Cholesky decomposition.

    Implementation of the corresponding torch function for converting a precision
    matrix to a scale lower triangular matrix.
    See: torch.distributions.multivariate_normal._precision_to_scale_tril.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.

    Raises:
        ValueError: If the precision matrix is not positive definite.
    """
    Lf = jnp.linalg.cholesky(jnp.flip(prec, axis=(-2, -1)))

    # JIT COMPILE ERROR DUE TO DEPENDENCY ON VALUE.
    # if jnp.any(jnp.isnan(Lf)):
    #     msg = "matrix is not positive definite"
    #     raise ValueError(msg)

    L_inv = jnp.transpose(jnp.flip(Lf, axis=(-2, -1)), axes=(-2, -1))
    Id = jnp.eye(prec.shape[-1], dtype=prec.dtype)
    L = jax.scipy.linalg.solve_triangular(L_inv, Id, trans="T")
    return L


def full_prec_to_posterior_state(
    prec: Num[Array, "P P"],
) -> dict[str, Num[Array, "P P"]]:
    """Convert precision matrix to scale matrix.

    The provided precision matrix is converted to a scale matrix, which is the lower
    triangular matrix L such that L @ L.T is the covariance matrix using
    `prec_to_scale`.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.
    """
    scale = full_prec_to_scale(prec)

    return {"scale": scale}


def full_posterior_state_to_scale(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    """Create a scale matrix-vector product function.

    The scale matrix is read from the state dictionary and is used to create a
    corresponding matrix-vector product function representing the action of the scale
    matrix on a vector.

    Args:
        state: Dictionary containing the scale matrix.

    Returns:
        A function that computes the scale matrix-vector product.
    """

    def scale_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] @ vec

    return scale_mv


def full_posterior_state_to_cov(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    """Create a covariance matrix-vector product function.

    The scale matrix is read from the state dictionary and is used to create a
    corresponding matrix-vector product function representing the action of the cov
    matrix on a vector. The covariance matrix is computed as the product of the scale
    matrix and its transpose.

    Args:
        state: Dictionary containing the scale matrix.

    Returns:
        A function that computes the covariance matrix-vector product.
    """
    cov = state["scale"] @ state["scale"].T

    def cov_mv(vec: FlatParams) -> FlatParams:
        return cov @ vec

    return cov_mv


# ---------------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------------


def create_diagonal_curvature(mv: CurvatureMV, layout: Layout, **kwargs) -> FlatParams:
    """Generate a diagonal curvature.

    The diagonal of the curvature matrix-vector product is computed as an approximation
    to the full matrix.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        **kwargs: Additional arguments (unused).

    Returns:
        A 1D array representing the diagonal curvature.
    """
    del kwargs
    curv_diagonal = diagonal(mv, layout=layout)
    return curv_diagonal


def diagonal_curvature_to_precision(
    curv_estimate: FlatParams,
    prior_arguments: PriorArguments,
    loss_scaling_factor: Float = 1.0,
) -> FlatParams:
    """Add prior precision to the diagonal curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the diagonal curvature estimate.

    Args:
        curv_estimate: Diagonal curvature estimate.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.
        loss_scaling_factor: Factor by which the user-provided loss function is
            scaled. Defaults to 1.0.

    Returns:
        Updated diagonal curvature with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    sigma = prior_arguments.get("sigma", 1.0)
    return (
        sigma * curv_estimate + prior_prec * jnp.ones_like(curv_estimate.shape[-1])
    ) / loss_scaling_factor


def diagonal_prec_to_posterior_state(prec: FlatParams) -> dict[str, FlatParams]:
    """Convert precision matrix to scale matrix.

    The provided diagonal precision matrix is converted to the corresponding scale
    diagonal, which is returned as a PosteriorState dictionary.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.
    """
    return {"scale": jnp.sqrt(jnp.reciprocal(prec))}


def diagonal_posterior_state_to_scale(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    """Create a scale matrix-vector product function.

    The diagonal scale matrix is read from the state dictionary and is used to create
    a corresponding matrix-vector product function representing the action of the
    diagonal scale matrix on a vector.

    Args:
        state: Dictionary containing the diagonal scale matrix.

    Returns:
        A function that computes the diagonal scale matrix-vector product.
    """

    def diag_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] * vec

    return diag_mv


def diagonal_posterior_state_to_cov(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    """Create a covariance matrix-vector product function.

    The diagonal covariance matrix is computed as the product of the diagonal scale
    matrix with itself.

    Args:
        state: Dictionary containing the diagonal scale matrix.

    Returns:
        A function that computes the diagonal covariance matrix-vector product.
    """
    arr = state["scale"] ** 2

    def diag_mv(vec: FlatParams) -> FlatParams:
        return arr * vec

    return diag_mv


# ---------------------------------------------------------------------------------
# Low-rank
# ---------------------------------------------------------------------------------


def create_low_rank_curvature(
    mv: CurvatureMV,
    layout: Layout,
    low_rank_method: LowRankMethod = LowRankMethod.LANCZOS,
    **kwargs,
) -> LowRankTerms:
    """Generate a low-rank curvature approximation.

    The low-rank curvature is computed as an approximation to the full curvature
    matrix using the provided matrix-vector product function and either the Lanczos
    or LOBPCG algorithm.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        low_rank_method: Method to use for computing the low-rank approximation.
            Can be either "lanczos" or "lobpcg". Defaults to "lanczos".
        **kwargs: Additional arguments passed to the low-rank method.

    Returns:
        A LowRankTerms object representing the low-rank curvature approximation.
    """
    # Select and apply the low-rank method.
    low_rank_terms = {
        LowRankMethod.LANCZOS: lanczos_lowrank,
        LowRankMethod.LOBPCG: lobpcg_lowrank,
    }[low_rank_method](mv, layout=layout, **kwargs)

    return low_rank_terms


def create_low_rank_mv(
    low_rank_terms: LowRankTerms,
) -> Callable[[FlatParams], FlatParams]:
    r"""Create a low-rank matrix-vector product function.

    The low-rank matrix-vector product is computed as the sum of the scalar multiple
    of the vector by the scalar and the product of the matrix-vector product of the
    eigenvectors and the eigenvalues times the eigenvector-vector product:

    $$
    scalar * \text{vec} + U @ (S * (U.T @ \text{vec}))
    $$

    Args:
        low_rank_terms: Low-rank curvature approximation.

    Returns:
        A function that computes the low-rank matrix-vector product.
    """
    U, S, scalar = jax.tree_util.tree_leaves(low_rank_terms)

    def low_rank_mv(vec: FlatParams) -> FlatParams:
        return scalar * vec + U @ (S * (U.T @ vec))

    return low_rank_mv


def low_rank_square(state: LowRankTerms) -> LowRankTerms:
    r"""Square the low-rank curvature approximation.

    This returns the LowRankTerms which correspond to the squared low rank
    approximation.

    $$ (U S U^{\top} + scalar I) ** 2
    = scalar**2 + U ((S + scalar) ** 2 - scalar**2) U^{\top} $$

    Args:
        state: Low-rank curvature approximation.

    Returns:
        A LowRankTerms object representing the squared low-rank curvature approximation.
    """
    U, S, scalar = jax.tree_util.tree_leaves(state)
    scalar_sq = scalar**2
    return LowRankTerms(
        U=U,
        S=(S + scalar) ** 2 - scalar_sq,
        scalar=scalar_sq,
    )


def low_rank_curvature_to_precision(
    curv_estimate: LowRankTerms,
    prior_arguments: PriorArguments,
    loss_scaling_factor: Float = 1.0,
) -> LowRankTerms:
    """Add prior precision to the low-rank curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read from the
    `prior_arguments` dictionary and added to the scalar component of the
    LowRankTerms.

    Args:
        curv_estimate: Low-rank curvature approximation.
        prior_arguments: Dictionary containing prior precision
            as 'prior_prec'.
        loss_scaling_factor: Factor by which the user-provided loss function is
            scaled. Defaults to 1.0.

    Returns:
        LowRankTerms: Updated low-rank curvature approximation with added prior
            precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    sigma = prior_arguments.get("sigma", 1.0)
    U, S, _ = jax.tree.leaves(curv_estimate)
    return LowRankTerms(
        U=U,
        S=(sigma * S),
        scalar=prior_prec / loss_scaling_factor,
    )


def low_rank_prec_to_posterior_state(
    curv_estimate: LowRankTerms,
) -> dict[str, LowRankTerms]:
    """Convert the low-rank precision representation to a posterior state.

    The scalar component and eigenvalues of the low-rank curvature estimate
    are transformed to represent the posterior scale, creating again a `LowRankTerms`
    representation.

    Args:
        curv_estimate: Low-rank curvature estimate.

    Returns:
        A dictionary with the posterior state represented as `LowRankTerms`.
    """
    U, S, scalar = jax.tree_util.tree_leaves(curv_estimate)
    scalar_sqrt_inv = jnp.reciprocal(jnp.sqrt(scalar))
    return {
        "scale": LowRankTerms(
            U=U,
            S=jnp.reciprocal(jnp.sqrt(S + scalar)) - scalar_sqrt_inv,
            scalar=scalar_sqrt_inv,
        )
    }


def low_rank_posterior_state_to_scale(
    state: dict[str, LowRankTerms],
) -> Callable[[FlatParams], FlatParams]:
    """Create a matrix-vector product function for the scale matrix.

    The state dictionary containing the low-rank representation of the covariance state
    is used to create a function that computes the matrix-vector product for the scale
    matrix.

    Args:
        state: Dictionary containing the low-rank scale.

    Returns:
        A function that computes the scale matrix-vector product.
    """
    return create_low_rank_mv(state["scale"])


def low_rank_posterior_state_to_cov(
    state: dict[str, LowRankTerms],
) -> Callable[[FlatParams], FlatParams]:
    """Create a matrix-vector product function for the covariance matrix.

    The state dictionary containing the low-rank representation of the covariance state
    is used to create a function that computes the matrix-vector product for the
    covariance matrix.

    Args:
        state: Dictionary containing the low-rank scale.

    Returns:
        A function that computes the covariance matrix-vector product.
    """
    return create_low_rank_mv(low_rank_square(state["scale"]))


# ---------------------------------------------------------------------------------
# General api for curvature types
# ---------------------------------------------------------------------------------

CURVATURE_METHODS: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: create_full_curvature,
    CurvApprox.DIAGONAL: create_diagonal_curvature,
    CurvApprox.LANCZOS: create_low_rank_curvature,
    CurvApprox.LOBPCG: partial(
        create_low_rank_curvature, low_rank_method=LowRankMethod.LOBPCG
    ),
}

CURVATURE_PRECISION_METHODS: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_curvature_to_precision,
    CurvApprox.DIAGONAL: diagonal_curvature_to_precision,
    CurvApprox.LANCZOS: low_rank_curvature_to_precision,
    CurvApprox.LOBPCG: low_rank_curvature_to_precision,
}

CURVATURE_TO_POSTERIOR_STATE: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_prec_to_posterior_state,
    CurvApprox.DIAGONAL: diagonal_prec_to_posterior_state,
    CurvApprox.LANCZOS: low_rank_prec_to_posterior_state,
    CurvApprox.LOBPCG: low_rank_prec_to_posterior_state,
}

CURVATURE_STATE_TO_SCALE: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_posterior_state_to_scale,
    CurvApprox.DIAGONAL: diagonal_posterior_state_to_scale,
    CurvApprox.LANCZOS: low_rank_posterior_state_to_scale,
    CurvApprox.LOBPCG: low_rank_posterior_state_to_scale,
}

CURVATURE_STATE_TO_COV: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_posterior_state_to_cov,
    CurvApprox.DIAGONAL: diagonal_posterior_state_to_cov,
    CurvApprox.LANCZOS: low_rank_posterior_state_to_cov,
    CurvApprox.LOBPCG: low_rank_posterior_state_to_cov,
}


# ---------------------------------------------------------------------------------
# General api for creating posterior functions
# ---------------------------------------------------------------------------------


@dataclass
class Posterior:
    state: PosteriorState
    cov_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    scale_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]


def estimate_curvature(
    curv_type: CurvApprox | str,
    mv: CurvatureMV,
    layout: Layout | None = None,
    **kwargs,
):
    """Estimate the curvature based on the provided type.

    Args:
        curv_type: Type of curvature approximation ('full', 'diagonal', 'lanczos',
            'lobpcg').
        mv: Function representing the curvature.
        layout: Defines the format of the layout for matrix-vector products. If None or
            an integer, no flattening/unflattening is used.
        **kwargs: Additional key-word arguments passed to the curvature estimation
            function.

    Returns:
        The estimated curvature.
    """
    curv_estimate = CURVATURE_METHODS[curv_type](mv, layout=layout, **kwargs)

    # Ignore lazy evaluation
    curv_estimate = jax.tree.map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        curv_estimate,
    )

    return curv_estimate


def set_posterior_fn(
    curv_type: CurvatureKeyType, curv_estimate: PyTree, *, layout: Layout, **kwargs
) -> Callable:
    """Set the posterior function based on the curvature estimate.

    Args:
        curv_type: Type of curvature approximation. Options include ('full', 'diagonal',
            'lanczos', 'lobpcg').
        curv_estimate: Estimated curvature.
        layout: Defines the format of the layout for matrix-vector products.
        **kwargs: Additional key-word arguments (unused).

    Returns:
        A function that computes the posterior state.
    """
    del kwargs
    if layout is not None and not isinstance(layout, int | PyTree):
        msg = "Layout must be an integer, PyTree or None."
        raise ValueError(msg)

    # Create functions for flattening and unflattening if required
    if layout is None or isinstance(layout, int):
        flatten = unflatten = None
    else:
        # Use custom flatten/unflatten functions for complex pytrees
        flatten, unflatten = create_pytree_flattener(layout)

    def posterior_fn(
        prior_arguments: PriorArguments,
        loss_scaling_factor: Float = 1.0,
    ) -> PosteriorState:
        """Compute the posterior state.

        Args:
            prior_arguments: Prior arguments for the posterior.
            loss_scaling_factor: Factor by which the user-provided loss function is
                scaled. Defaults to 1.0.

        Returns:
            PosteriorState: Dictionary containing:
                - 'state': Updated state of the posterior.
                - 'cov_mv': Function to compute covariance matrix-vector product.
                - 'scale_mv': Function to compute scale matrix-vector product.
        """
        # Calculate posterior precision.
        precision = CURVATURE_PRECISION_METHODS[curv_type](
            curv_estimate=curv_estimate,
            prior_arguments=prior_arguments,
            loss_scaling_factor=loss_scaling_factor,
        )

        # Calculate posterior state
        state = CURVATURE_TO_POSTERIOR_STATE[curv_type](precision)

        # Extract matrix-vector product
        scale_mv_from_state = CURVATURE_STATE_TO_SCALE[curv_type]
        cov_mv_from_state = CURVATURE_STATE_TO_COV[curv_type]

        return Posterior(
            state=state,
            cov_mv=wrap_factory(cov_mv_from_state, flatten, unflatten),
            scale_mv=wrap_factory(scale_mv_from_state, flatten, unflatten),
        )

    return posterior_fn


def create_posterior_fn(
    curv_type: CurvApprox | str,
    mv: CurvatureMV,
    layout: Layout | None = None,
    **kwargs,
) -> Callable:
    """Factory function to create the posterior function given a curvature type.

    This sets up the posterior function which can then be initiated using
    prior_arguments by computing a specified curvature approximation and encoding the
    sequential computational order of:
        1) CURVATURE_PRIOR_METHODS,
        2) CURVATURE_TO_POSTERIOR_STATE,
        3) CURVATURE_STATE_TO_SCALE, and
        4) CURVATURE_STATE_TO_COV. All methods are selected from the corresponding
    dictionary by the curv_type argument. New methods can be registered using the
    register_curvature_method.

    Args:
        curv_type: Type of curvature approximation ('full', 'diagonal',
            'lanczos', 'lobpcg').
        mv: Function representing the curvature.
        layout: Defines the format of the layout for matrix-vector products. If None or
            an integer, no flattening/unflattening is used.
        **kwargs: Additional key-word arguments passed to the curvature estimation
            function.

    Returns:
        Callable: A posterior function that takes the prior_arguments and returns the
            posterior_state.
    """
    # Retrieve the curvature estimator based on the provided type
    curv_estimate = estimate_curvature(curv_type, mv=mv, layout=layout, **kwargs)

    # Set posterior fn based on curv_estimate
    posterior_fn = set_posterior_fn(curv_type, curv_estimate, layout=layout)

    return posterior_fn
