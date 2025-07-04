"""Full curvature approximation."""

import jax
import jax.numpy as jnp
from loguru import logger

from laplax.types import (
    Array,
    Callable,
    CurvatureMV,
    FlatParams,
    Float,
    Kwargs,
    Layout,
    Num,
    PriorArguments,
)
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import to_dense
from laplax.util.tree import get_size


def create_full_curvature(
    mv: CurvatureMV,
    layout: Layout,
    **kwargs: Kwargs,
) -> Num[Array, "P P"]:
    """Generate a full curvature approximation.

    The curvature is densed and flattened into a 2D array, that corresponds to the
    flattened parameter layout.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function. If `None` or an integer, no
            flattening/unflattening is used.
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
    r"""Add prior precision to the curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the curvature estimate. The curvature is scaled by the
    $\sigma^2$ parameter.

    Args:
        curv_estimate: Full curvature estimate matrix.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.
        loss_scaling_factor: Factor by which the user-provided loss function is
            scaled. Defaults to 1.0.

    Returns:
        Updated curvature matrix with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    sigma_squared = prior_arguments.get("sigma_squared", 1.0)

    return (
        sigma_squared * curv_estimate + prior_prec * jnp.eye(curv_estimate.shape[-1])
    ) / loss_scaling_factor


def full_prec_to_scale(
    prec: Num[Array, "P P"],
) -> Num[Array, "P P"]:
    """Convert precision matrix to scale matrix using Cholesky decomposition.

    This converts a precision matrix to a scale matrix using a Cholesky decomposition.
    The scale matrix is the lower triangular matrix L such that L @ L.T is the
    covariance matrix.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.
    """
    Lf = jnp.linalg.cholesky(jnp.flip(prec, axis=(-2, -1)))
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
    :func: `full_prec_to_scale`.

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
