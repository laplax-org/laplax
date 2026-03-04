"""Diagonal curvature approximation."""

import jax.numpy as jnp

from laplax.types import (
    Callable,
    CurvatureMV,
    FlatParams,
    Float,
    Kwargs,
    Layout,
    PriorArguments,
)
from laplax.util.mv import diagonal


def create_diagonal_curvature(
    mv: CurvatureMV,
    layout: Layout,
    **kwargs: Kwargs,
) -> FlatParams:
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
    r"""Add prior precision to the diagonal curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the diagonal curvature estimate. The curvature (here:
    diagonal) is scaled by the $\sigma^2$ parameter.

    Args:
        curv_estimate: Diagonal curvature estimate.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.
        loss_scaling_factor: Factor by which the user-provided loss function is
            scaled. Defaults to 1.0.

    Returns:
        Updated diagonal curvature with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    sigma_squared = prior_arguments.get("sigma_squared", 1.0)
    return (
        sigma_squared * curv_estimate
        + prior_prec * jnp.ones_like(curv_estimate.shape[-1])
    ) / loss_scaling_factor


def diagonal_prec_to_posterior_state(
    prec: FlatParams,
) -> dict[str, FlatParams]:
    """Convert precision matrix to scale matrix.

    The provided diagonal precision matrix is converted to the corresponding scale
    diagonal, which is returned as a `PosteriorState` dictionary. The scale matrix is
    the diagonal matrix with the inverse of the diagonal elements.

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
