"""Objective builders for FSP training.

This module provides reusable loss/objective constructors that were previously
duplicated in downstream projects. They are now part of Laplax util so that
experiments can import from a stable location.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from laplax.enums import LossFn  # noqa: TC001
from laplax.types import (  # noqa: TC001
    Callable,
    Data,
    Float,
    Int,
    ModelFn,
    Params,
    PredArray,
)


def create_loss_nll(
    model_fn: ModelFn,
    dataset_size: int | None = None,
) -> LossFn:
    r"""Create a Negative Log-Likelihood (Gaussian) loss.
    Computes the negative log-likelihood:
        -log p(y | f(X)) = -sum_i log N(y_i | f(x_i), sigma^2)
    The returned loss has signature `(data, params, scale) -> Float`.
    """  # noqa: D205, DOC201

    def loss_nll(
        data: Data, params: Params, scale: Float | Params | None = None
    ) -> Float:
        preds = jax.vmap(model_fn, in_axes=(0, None))(data["input"], params)
        nll = -jax.scipy.stats.norm.logpdf(
            data["target"], loc=preds, scale=scale
        ).mean()
        return nll if dataset_size is None else nll * dataset_size

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
    *,
    has_batch_dim: bool = True,
) -> LossFn:
    r"""Create the FSP RKHS regularization term.
    Computes the regularizer:
        1/2 (f(c) - m)^T K^{-1}(c, c) (f(c) - m)
    The returned loss has signature `(context_points, params) -> Float`.
    """  # noqa: D205, DOC201
    if not has_batch_dim:

        def loss_reg(context_points: PredArray, params: Params) -> Float:
            f_c = (
                jax.vmap(model_fn, in_axes=(0, None))(context_points, params)
                - prior_mean
            )
            K_c_c = prior_cov_kernel(*context_points)
            left = jax.scipy.linalg.solve(K_c_c, f_c, assume_a="sym")
            return 0.5 * jnp.einsum("ij,ij->", f_c, left)

    else:

        def loss_reg(context_points: Data, params: Params) -> Float:
            f_c = jax.vmap(model_fn, in_axes=(0, None))(
                context_points["context"], params
            )
            K_c_c = prior_cov_kernel(context_points["context"], context_points["grid"])
            left = jax.scipy.linalg.solve(K_c_c, f_c, assume_a="sym")
            return 0.5 * jnp.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    dataset_size: Int,
    prior_mean: PredArray,
    prior_cov_kernel: Callable,
) -> LossFn:
    """Create the FSP objective combining NLL and regularization terms."""  # noqa: DOC201
    loss_nll = create_loss_nll(model_fn, dataset_size)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    def fsp_objective(
        data: Data,
        context_points: Data | PredArray,
        params: Params,
        scale: Float | Params | None = None,
    ) -> Float:
        nll_term = loss_nll(data, params, scale)
        reg_term = loss_reg(context_points, params)
        return nll_term + reg_term

    return fsp_objective
