# likelihood.py

"""Compute the marginal log-likelihood for different curvature estimations.

Implemented according to:
Smith, J., et al. (2023):
Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning.
Proceedings of the International Conference on Machine Learning, 25(3), 234-245.

It includes functions to calculate the marginal log-likelihood based on various
curvature approximations, including:

- full
- diagonal
- low-rank
"""

from collections.abc import Callable

import jax.numpy as jnp

from laplax.curv.cov import CURVATURE_PRECISION_METHODS
from laplax.curv.utils import LowRankTerms, concatenate_model_and_loss_fn
from laplax.enums import CurvApprox, LossFn
from laplax.types import (
    Array,
    CurvatureKeyType,
    Data,
    FlatParams,
    Float,
    ModelFn,
    Num,
    Params,
    PriorArguments,
    PyTree,
)
from laplax.util.flatten import full_flatten
from laplax.util.tree import get_size


def joint_log_likelihood(
    full_fn: Callable,
    prior_arguments: PriorArguments,
    params: Params,
    data: Data,
) -> Float:
    r"""Computes the joint log-likelihood for a model.

    This function computes the joint log-likelihood for a model, which is given by:

    $$
    \log p(D, \theta | M) = \log p(D | \theta, M) + \log p(\theta | M)
    $$

    If we assume a Gaussian prior on the parameters with precision $\tau^{-2}$,
    then the log-prior is given by:

    $$
        \log p(\theta \vert \tau^{-2}) = -\frac{1}{2} \log |\frac{1}{2\pi} \tau^{-2}
        \vert - \frac{1}{2} \tau^{-2} \vert \theta \vert^2
    $$

    Args:
        full_fn: model loss function that has the parameters and the data as input and
            output the loss
        prior_arguments: prior arguments
        params: model parameters
        data: training data

    Returns:
        The joint log-likelihood.
    """
    # Compute the log-prior
    params_square_norm = jnp.sum(full_flatten(params) ** 2)
    prior_prec = prior_arguments["prior_prec"]
    log_prior_term1 = -0.5 * prior_prec * params_square_norm
    log_prior_term2 = (
        -0.5 * get_size(params) * (jnp.log(2 * jnp.pi) - jnp.log(prior_prec))
    )
    log_prior = log_prior_term1 + log_prior_term2

    # Compute the log-likelihood
    sigma = prior_arguments["sigma"]
    log_likelihood = -(1 / (2 * sigma**2)) * full_fn(
        data["input"],
        data["target"],
        params,
    )  # Assumes summed loss

    # Compute the joint log-likelihood
    return log_likelihood + log_prior


def full_marginal_log_likelihood(
    posterior_precision: Num[Array, "P P"],
    prior_arguments: PriorArguments,
    full_fn: Callable,
    params: Params,
    data: Data,
) -> Float:
    r"""Computes the marginal log likelihood for the full posterior function.

    The marginal log-likelihood is given by:

    $$
        \log p(D | M) = \log p(D, \theta_* | M)
        - \frac{1}{2} \log |\frac{1}{2\pi} H_{\theta_*}\vert
    $$

    Args:
        posterior_precision: posterior precision
        prior_arguments: prior arguments
        full_fn: model loss function that has the parameters and the data as input and
            output the loss
        params: model parameters
        data: training data

    Returns:
        The marginal likelihood estimation
    """
    # Compute the log-likelihood
    log_likelihood = joint_log_likelihood(
        full_fn=full_fn,
        prior_arguments=prior_arguments,
        params=params,
        data=data,
    )

    # Log det of posterior precision
    log_det_H = jnp.linalg.slogdet(posterior_precision)[1]
    param_count = get_size(params)
    evidence = -0.5 * param_count * jnp.log(2 * jnp.pi) + 0.5 * log_det_H

    # Compute the marginal log-likelihood
    lml = log_likelihood - evidence

    return lml


def diagonal_marginal_log_likelihood(
    posterior_precision: FlatParams,
    prior_arguments: PriorArguments,
    full_fn: Callable,
    params: Params,
    data: Data,
) -> Float:
    r"""Computes the marginal log likelihood for a diagonal approximation.

    The marginal log-likelihood is given by:

    $$
        \log p(D | M) = \log p(D, \theta_* | M)
            - \frac{1}{2} \log |\frac{1}{2\pi} H_{\theta_*}\vert
    $$

    Here the log-determinant of the posterior precision simplifies to:

    $$
        \sum_{i=1}^{P} \log d_i
    $$

    where $d_i$ is the $i$-th diagonal element of the posterior precision.

    Args:
        posterior_precision: posterior precision
        prior_arguments: prior arguments
        full_fn: model loss function that has the parameters and the data as input and
            output the loss
        params: model parameters
        data: training data

    Returns:
        The marginal likelihood estimation.
    """
    # Compute the log-likelihood
    log_likelihood = joint_log_likelihood(
        full_fn=full_fn,
        prior_arguments=prior_arguments,
        params=params,
        data=data,
    )

    # Log det of posterior precision
    log_det_H = jnp.sum(jnp.log(posterior_precision))
    param_count = get_size(params)
    evidence = -0.5 * param_count * jnp.log(2 * jnp.pi) + 0.5 * log_det_H

    # Compute the marginal log-likelihood
    lml = log_likelihood - evidence

    return lml


def low_rank_marginal_log_likelihood(
    posterior_precision: LowRankTerms,
    prior_arguments: PriorArguments,
    full_fn: Callable,
    params: Params,
    data: Data,
) -> Float:
    r"""Computes the marginal log likelihood for a low-rank approximation.

    The marginal log-likelihood is given by:

    $$
    \log p(D | M) = \log p(D, \theta_* | M)
        - \frac{1}{2} \log |\frac{1}{2\pi} H_{\theta_*}\vert
    $$

    Here the log-determinant of the posterior precision (with $U\Lambda U^T + D$)
    simplifies to:

    $$
    \sum_{i=1}^{R} \log ( 1 + d_i^{-1} \cdot \lambda_i) + \sum_{i=1}^{P} \log d_i
    $$

    where $d_i$ is the $i$-th diagonal element of the prior precision and
    $\lambda_i$ is the $i$-th eigenvalue of the low-rank approximation.

    Args:
        posterior_precision: posterior precision
        prior_arguments: prior arguments
        full_fn: model loss function that has the parameters and the data as input and
            output the loss
        params: model parameters
        data: training data

    Returns:
        The marginal likelihood estimation.
    """
    # Compute the log-likelihood
    log_likelihood = joint_log_likelihood(
        full_fn=full_fn,
        prior_arguments=prior_arguments,
        params=params,
        data=data,
    )

    # Log det of posterior precision
    rank = posterior_precision.S.shape[0]
    log_det_H = rank * jnp.log(posterior_precision.scalar) + jnp.sum(
        jnp.log(1 + posterior_precision.scalar * posterior_precision.S)
    )
    param_count = get_size(params)
    evidence = -0.5 * param_count * jnp.log(2 * jnp.pi) + 0.5 * log_det_H

    # Compute the marginal log-likelihood
    lml = log_likelihood - evidence

    return lml


CURVATURE_MARGINAL_LOG_LIKELIHOOD: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_marginal_log_likelihood,
    CurvApprox.DIAGONAL: diagonal_marginal_log_likelihood,
    CurvApprox.LANCZOS: low_rank_marginal_log_likelihood,
    CurvApprox.LOBPCG: low_rank_marginal_log_likelihood,
}


def marginal_log_likelihood(
    curv_estimate: PyTree,
    prior_arguments: PriorArguments,
    data: Data,
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable,
    curv_type: CurvatureKeyType,
    *,
    vmap_over_data: bool = False,
    loss_scaling_factor: Float = 1.0,
) -> Float:
    r"""Compute the marginal log-likelihood for a given curvature approximation.

    The marginal log-likelihood is given by:

    $$
    \log p(D | M) = \log p(D, \theta_* | M)
        - \frac{1}{2} \log |\frac{1}{2\pi} H_{\theta_*}\vert
    $$

    Here $H_{\theta_*}$ is the Hessian/GGN of the loss function evaluated at the
    model parameters. The likelihood function is given by the negative loss function.

    Args:
        curv_estimate: curvature estimate
        prior_arguments: prior arguments
        data: training data
        model_fn: model function
        params: model parameters
        loss_fn: loss function
        curv_type: curvature type
        vmap_over_data: whether the model has a batch dimension
        loss_scaling_factor: loss scaling factor

    Returns:
        The marginal log-likelihood.
    """
    full_fn = concatenate_model_and_loss_fn(
        model_fn,
        loss_fn,
        vmap_over_data=vmap_over_data,
    )

    posterior_precision = CURVATURE_PRECISION_METHODS[curv_type](
        curv_estimate,
        prior_arguments,
        loss_scaling_factor=loss_scaling_factor,
    )

    marginal_log_likelihood = CURVATURE_MARGINAL_LOG_LIKELIHOOD[curv_type](
        posterior_precision,
        prior_arguments,
        full_fn,
        params,
        data,
    )

    return marginal_log_likelihood
