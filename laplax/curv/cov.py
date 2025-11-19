"""Posterior covariance functions for various curvature estimates."""

from dataclasses import dataclass
from functools import partial

import jax

from laplax.curv.diagonal import (
    create_diagonal_curvature,
    diagonal_curvature_to_precision,
    diagonal_posterior_state_to_cov,
    diagonal_posterior_state_to_scale,
    diagonal_prec_to_posterior_state,
)
from laplax.curv.full import (
    create_full_curvature,
    full_curvature_to_precision,
    full_posterior_state_to_cov,
    full_posterior_state_to_scale,
    full_prec_to_posterior_state,
)
from laplax.curv.low_rank import (
    create_low_rank_curvature,
    low_rank_curvature_to_precision,
    low_rank_posterior_state_to_cov,
    low_rank_posterior_state_to_scale,
    low_rank_prec_to_posterior_state,
)
from laplax.curv.utils import LowRankTerms
from laplax.enums import CurvApprox, LowRankMethod
from laplax.types import (
    Callable,
    CurvatureKeyType,
    CurvatureMV,
    FlatParams,
    Float,
    Kwargs,
    Layout,
    PosteriorState,
    PriorArguments,
    PyTree,
)
from laplax.util.flatten import (
    create_pytree_flattener,
    wrap_factory,
)

# -----------------------------------------------------------------------------
# General api for curvature types
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# General api for creating posterior functions
# -----------------------------------------------------------------------------


@dataclass
class Posterior:
    """Represents a posterior distribution with associated matrix-vector operations."""

    state: PosteriorState
    cov_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    scale_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    rank: int | None = None  # If scale_mv is low-rank, rank is the rank of the low-rank
    low_rank_terms: LowRankTerms | None = None  # Optional low-rank decomposition terms


def estimate_curvature(
    curv_type: CurvApprox | str,
    mv: CurvatureMV,
    layout: Layout | None = None,
    **kwargs: Kwargs,
) -> PyTree:
    """Estimate the curvature based on the provided type.

    Args:
        curv_type: Type of curvature approximation (`CurvApprox.FULL`,
            `CurvApprox.DIAGONAL`, `CurvApprox.LANCZOS`, `CurvApprox.LOBPCG`) or
            corresponding string (`'full'`, `'diagonal'`, `'lanczos'`, `'lobpcg'`).
        mv: Function representing the curvature-vector product.
        layout: Defines the input layer format of the matrix-vector products. If None or
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
    curv_type: CurvatureKeyType,
    curv_estimate: PyTree,
    *,
    layout: Layout,
    **kwargs: Kwargs,
) -> Callable:
    """Set the posterior function based on the curvature estimate.

    Args:
        curv_type: Type of curvature approximation (`CurvApprox.FULL`,
            `CurvApprox.DIAGONAL`, `CurvApprox.LANCZOS`, `CurvApprox.LOBPCG`) or
            corresponding string (`'full'`, `'diagonal'`, `'lanczos'`, `'lobpcg'`).
        curv_estimate: Estimated curvature.
        layout: Defines the input/output layout of the corresponding curvature-vector
            products. If `None` or an integer, no flattening/unflattening is used.
        **kwargs: Additional key-word arguments (unused).

    Returns:
        A function that computes the posterior state.

    Raises:
        ValueError: When layout is neither an integer, a PyTree, nor None.
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
    **kwargs: Kwargs,
) -> Callable:
    """Factory function to create the posterior function given a curvature type.

    This sets up the posterior function, which can then be initiated using
    `prior_arguments` by computing a specified curvature approximation and encoding the
    sequential computational order of:

        1. `CURVATURE_PRIOR_METHODS`
        2. `CURVATURE_TO_POSTERIOR_STATE`
        3. `CURVATURE_STATE_TO_SCALE`
        4. `CURVATURE_STATE_TO_COV`

    All methods are selected from the corresponding dictionary by the `curv_type`
    argument. New methods can be registered using the
    :func:`laplax.register.register_curvature_method` method.
    See the :mod:`laplax.register` module for more details.

    Args:
        curv_type: Type of curvature approximation (`CurvApprox.FULL`,
            `CurvApprox.DIAGONAL`, `CurvApprox.LANCZOS`, `CurvApprox.LOBPCG`) or
            corresponding string (`'full'`, `'diagonal'`, `'lanczos'`, `'lobpcg'`).
        mv: Function representing the curvature.
        layout: Defines the format of the layout for matrix-vector products. If `None`
            or an integer, no flattening/unflattening is used.
        **kwargs: Additional keyword arguments passed to the curvature estimation
            function.

    Returns:
        A posterior function that takes the `prior_arguments` and returns the
            `posterior_state`.
    """
    # Retrieve the curvature estimator based on the provided type
    curv_estimate = estimate_curvature(curv_type, mv=mv, layout=layout, **kwargs)

    # Set posterior fn based on curv_estimate
    posterior_fn = set_posterior_fn(curv_type, curv_estimate, layout=layout)

    return posterior_fn
