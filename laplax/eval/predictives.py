import math

import jax
import jax.numpy as jnp

from laplax.types import Kwargs

LAMBDA_0 = math.pi / 8


def laplace_bridge(
    mean: jax.Array,
    var: jax.Array,
    *,
    use_correction: bool,
) -> jax.Array:
    """Laplace bridge approximation.

    Returns:
        The predictive.
    """
    num_classes = mean.shape[1]

    if use_correction:
        c = jnp.sum(var, axis=0) / math.sqrt(num_classes / 2)  # [...]
        c_expanded = jnp.expand_dims(c, axis=0)  # [1, ...]
        mean = mean / jnp.sqrt(c_expanded)  # [C, ...]
        var = var / c_expanded  # [C, ...]

    # Laplace bridge
    sum_exp_neg_mean_p = jnp.sum(jnp.exp(-mean), axis=0)  # [...]
    sum_exp_neg_mean_p_expanded = jnp.expand_dims(
        sum_exp_neg_mean_p, axis=0
    )  # [1, ...]
    dirichlet_params = (
        1
        - 2 / num_classes
        + jnp.exp(mean) * sum_exp_neg_mean_p_expanded / (num_classes**2)
    ) / var  # [C, ...]

    return dirichlet_predictive(dirichlet_params)


def dirichlet_predictive(dirichlet_params: jax.Array) -> jax.Array:
    """Predictive mean of Dirichlet distributions.

    Returns:
        The predictive.
    """
    predictive = dirichlet_params / jnp.sum(dirichlet_params)  # [C, ...]

    return predictive


def mean_field_0_predictive(
    mean: jax.Array, var: jax.Array, **kwargs: Kwargs
) -> jax.Array:
    del kwargs

    predictive = jax.nn.softmax(mean / jnp.sqrt(1 + LAMBDA_0 * var))

    return predictive


def mean_field_1_predictive(
    mean: jax.Array, var: jax.Array, **kwargs: Kwargs
) -> jax.Array:
    del kwargs

    mu_diff = mean[None, :] - mean[:, None]  # [C, C]
    s_sum = var[:, None] + var[None, :]  # [C, C]
    exp_terms = jnp.exp(mu_diff / jnp.sqrt(1 + LAMBDA_0 * s_sum))  # [C, C]
    sum_terms = jnp.sum(exp_terms, axis=1)  # [C]
    predictive = 1 / sum_terms

    return predictive


def mean_field_2_predictive(mean: jax.Array, cov: jax.Array) -> jax.Array:
    mu_diff = mean[None, :] - mean[:, None]  # [C, C]
    s_diag = jnp.diag(cov)
    s_sum = s_diag[:, None] + s_diag[None, :] - 2 * cov  # [C, C]
    # s_sum[i, j] = S[i, i] + S[j, j] - 2 * S[i, j]
    exp_terms = jnp.exp(mu_diff / jnp.sqrt(1 + LAMBDA_0 * s_sum))  # [C, C]
    sum_terms = jnp.sum(exp_terms, axis=1)  # [C]
    predictive = 1 / sum_terms

    return predictive
