"""Low-rank curvature approximation."""

import jax
import jax.numpy as jnp

from laplax.curv.lanczos import lanczos_lowrank
from laplax.curv.lobpcg import lobpcg_lowrank
from laplax.curv.utils import LowRankTerms
from laplax.enums import LowRankMethod
from laplax.types import (
    Callable,
    CurvatureMV,
    FlatParams,
    Float,
    Kwargs,
    Layout,
    PriorArguments,
)


def create_low_rank_curvature(
    mv: CurvatureMV,
    layout: Layout,
    low_rank_method: LowRankMethod = LowRankMethod.LANCZOS,
    **kwargs: Kwargs,
) -> LowRankTerms:
    r"""Generate a low-rank curvature approximation.

    The low-rank curvature is computed as an approximation to the full curvature matrix
    using the provided matrix-vector product function and either the Lanczos or LOBPCG
    algorithm. The low-rank approximation is returned as a `LowRankTerms` object.
    The low-rank approximation is computed as:

    $$
    \text{\textbf{Curv}} \approx U S U^{\top}
    $$

    where $U$ are the eigenvectors and $S$ are the eigenvalues. The `LowRankTerms` holds
    the eigenvectors, eigenvalues, and a scalar factor. The latter can be used to
    express an isotropic Gaussian prior.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        low_rank_method: Method to use for computing the low-rank approximation.
            Can be either `LowRankMethod.LANCZOS` or `LowRankMethod.LOBPCG`.
            Defaults to `LowRankMethod.LANCZOS`.
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

    The low-rank matrix-vector product is computed as the sum of the scalar multiple of
    the vector by the scalar and the product of the matrix-vector product of the
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


def low_rank_square(
    state: LowRankTerms,
) -> LowRankTerms:
    r"""Square the low-rank curvature approximation.

    This returns the `LowRankTerms` which correspond to the squared low-rank
    approximation. The squared low-rank approximation is computed as:

    $$
    (U S U^{\top} + scalar I) ** 2
    = scalar**2 + U ((S + scalar) ** 2 - scalar**2) U^{\top}
    $$

    Args:
        state: Low-rank curvature approximation.

    Returns:
        A `LowRankTerms` object representing the squared low-rank curvature
            approximation.
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
    r"""Add prior precision to the low-rank curvature estimate.

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
    sigma_squared = prior_arguments.get("sigma_squared", 1.0)
    U, S, _ = jax.tree.leaves(curv_estimate)
    return LowRankTerms(
        U=U,
        S=(sigma_squared * S),
        scalar=prior_prec / loss_scaling_factor,
    )


def low_rank_prec_to_posterior_state(
    curv_estimate: LowRankTerms,
) -> dict[str, LowRankTerms]:
    """Convert the low-rank precision representation to a posterior state.

    The scalar component and eigenvalues of the low-rank curvature estimate are
    transformed to represent the posterior scale, creating again a `LowRankTerms`
    representation. The scale matrix is the diagonal matrix with the inverse of the
    square root of the low-rank approximation using the Woodbury identity for analytic
    inversion.

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
    matrix. The scale matrix is the diagonal matrix with the inverse of the square root
    of the eigenvalues.

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
    covariance matrix. The covariance matrix is the low-rank approximation squared.

    Args:
        state: Dictionary containing the low-rank scale.

    Returns:
        A function that computes the covariance matrix-vector product.
    """
    return create_low_rank_mv(low_rank_square(state["scale"]))
