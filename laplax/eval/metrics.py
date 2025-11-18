# metrics.py

r"""Regression and Classification Metrics for Uncertainty Quantification.

This module provides a comprehensive suite of classification and regression metrics for
evaluating probabilistic models.

## Key Features

### Classification Metrics

- **Accuracy**
- **Top-k Accuracy**
- **Cross-Entropy**
- **Multiclass Brier Score**
- **Expected Calibration Error (ECE)**
- **Maximum Calibration Error (MCE)**

### Regression Metrics

- **Root Mean Squared Error (RMSE)**
- **Chi-squared**
- **Negative Log-Likelihood (NLL)** for Gaussian distributions

### Bin Metrics

- **Confidence and Correctness Metrics** binned by confidence intervals

---

The module leverages **JAX** for efficient numerical computation and supports flexible
evaluation for diverse model outputs.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

from laplax.curv.lanczos import lanczos_lowrank
from laplax.curv.utils import LowRankTerms
from laplax.enums import CalibrationErrorNorm
from laplax.eval.utils import apply_fns
from laplax.types import Array, Float, Kwargs

# --------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------


def correctness(pred: Array, target: Array, **kwargs: Kwargs) -> Array:
    """Determine if each target label matches the top-1 prediction.

    Computes a binary indicator for whether the predicted class matches the
    target class. If the target is a 2D array, it is first reduced to its
    class index using `argmax`.

    Args:
        pred: Array of predictions with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).
        **kwargs: Additional arguments (ignored).

    Returns:
        Boolean array of shape `(batch_size,)` indicating correctness
            for each prediction.
    """
    del kwargs

    pred = jnp.argmax(pred, axis=-1)

    if target.ndim == 2:
        target = jnp.argmax(target, axis=-1)

    return pred == target


def accuracy(
    pred: Array, target: Array, top_k: tuple[int] = (1,), **kwargs: Kwargs
) -> list[Array]:
    """Compute top-k accuracy for specified values of k.

    For each k in `top_k`, this function calculates the fraction of samples
    where the ground truth label is among the top-k predictions. If the target
    is a 2D array, it is reduced to its class index using `argmax`.

    Args:
        pred: Array of predictions with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).
        top_k: Tuple of integers specifying the values of k for top-k accuracy.
        **kwargs: Additional arguments (ignored).

    Returns:
        A list of accuracies corresponding to each k in `top_k`,
            expressed as percentages.
    """
    del kwargs
    max_k = min(max(top_k), pred.shape[1])
    batch_size = target.shape[0]

    _, pred = lax.top_k(pred, max_k)
    pred = pred.T

    if target.ndim == 2:
        target = jnp.argmax(target, axis=-1)

    correctness = pred == target.reshape(1, -1)

    return [
        jnp.sum(correctness[: min(k, max_k)].reshape(-1).astype(jnp.float32))
        * 100.0
        / batch_size
        for k in top_k
    ]


def cross_entropy(
    prob_p: Array, prob_q: Array, axis: int = -1, **kwargs: Kwargs
) -> Array:
    """Compute cross-entropy between two probability distributions.

    This function calculates the cross-entropy of `prob_p` relative to `prob_q`,
    summing over the specified axis.

    Args:
        prob_p: Array of true probability distributions.
        prob_q: Array of predicted probability distributions.
        axis: Axis along which to compute the cross-entropy (default: -1).
        **kwargs: Additional arguments (ignored).

    Returns:
        Cross-entropy values for each sample.
    """
    del kwargs
    p_log_q = jax.scipy.special.xlogy(prob_p, prob_q)

    return -p_log_q.sum(axis=axis)


def multiclass_brier(prob: Array, target: Array, **kwargs: Kwargs) -> Array:
    """Compute the multiclass Brier score.

    The Brier score is a measure of the accuracy of probabilistic predictions.
    For multiclass classification, it calculates the mean squared difference
    between the predicted probabilities and the true target.

    Args:
        prob: Array of predicted probabilities with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).
        **kwargs: Additional arguments (ignored).

    Returns:
        Mean Brier score across all samples.
    """
    del kwargs
    if target.ndim == 1:
        target = jax.nn.one_hot(target, num_classes=prob.shape[-1])

    preds_squared_sum = jnp.sum(prob**2, axis=-1, keepdims=True)
    score_components = 1 - 2 * prob + preds_squared_sum

    return -jnp.mean(target * score_components)


def calculate_bin_metrics(
    confidence: Array, correctness: Array, num_bins: int = 15, **kwargs: Kwargs
) -> tuple[Array, Array, Array]:
    """Calculate bin-wise metrics for confidence and correctness.

    Computes the proportion of samples, average confidence, and average accuracy
    within each bin, where the bins are defined by evenly spaced confidence
    intervals.

    Args:
        confidence: Array of predicted confidence values with shape `(n,)`.
        correctness: Array of correctness labels (0 or 1) with shape `(n,)`.
        num_bins: Number of bins for dividing the confidence range (default: 15).
        **kwargs: Additional arguments (ignored).

    Returns:
        Tuple of arrays containing:

            - Bin proportions: Proportion of samples in each bin.
            - Bin confidences: Average confidence for each bin.
            - Bin accuracies: Average accuracy for each bin.
    """
    del kwargs

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    indices = jnp.digitize(confidence, bin_boundaries) - 1
    indices = jnp.clip(indices, min=0, max=num_bins - 1)

    bin_counts = jnp.zeros(num_bins, dtype=confidence.dtype)
    bin_confidences = jnp.zeros(num_bins, dtype=confidence.dtype)
    bin_accuracies = jnp.zeros(num_bins, dtype=correctness.dtype)

    bin_counts = bin_counts.at[indices].add(1)
    bin_confidences = bin_confidences.at[indices].add(confidence)
    bin_accuracies = bin_accuracies.at[indices].add(correctness)

    bin_proportions = bin_counts / bin_counts.sum()
    pos_counts = bin_counts > 0
    bin_confidences = jnp.where(pos_counts, bin_confidences / bin_counts, 0)
    bin_accuracies = jnp.where(pos_counts, bin_accuracies / bin_counts, 0)

    return bin_proportions, bin_confidences, bin_accuracies


def calibration_error(
    confidence: jax.Array,
    correctness: jax.Array,
    num_bins: int,
    norm: CalibrationErrorNorm,
    **kwargs: Kwargs,
) -> jax.Array:
    """Compute the expected/maximum calibration error.

    Args:
        confidence: Float tensor of shape (n,) containing predicted confidences.
        correctness: Float tensor of shape (n,) containing the true correctness
            labels.
        num_bins: Number of equally sized bins.
        norm: Whether to return ECE (L1 norm) or MCE (inf norm).
        **kwargs: Additional arguments (ignored).

    Returns:
        The ECE/MCE.
    """
    del kwargs
    bin_proportions, bin_confidences, bin_accuracies = calculate_bin_metrics(
        confidence, correctness, num_bins
    )

    abs_diffs = jnp.abs(bin_accuracies - bin_confidences)

    if norm == CalibrationErrorNorm.L1:
        score = (bin_proportions * abs_diffs).sum()
    else:
        score = abs_diffs.max()

    return score


def expected_calibration_error(
    confidence: jax.Array, correctness: jax.Array, num_bins: int, **kwargs: Kwargs
) -> jax.Array:
    """Compute the expected calibration error.

    Args:
        confidence: Float tensor of shape (n,) containing predicted confidences.
        correctness: Float tensor of shape (n,) containing the true correctness
            labels.
        num_bins: Number of equally sized bins.
        **kwargs: Additional arguments (ignored).

    Returns:
        The ECE/MCE.

    """
    del kwargs
    return calibration_error(
        confidence=confidence,
        correctness=correctness,
        num_bins=num_bins,
        norm=CalibrationErrorNorm.L1,
    )


def maximum_calibration_error(
    confidence: jax.Array, correctness: jax.Array, num_bins: int, **kwargs: Kwargs
) -> jax.Array:
    """Compute the maximum calibration error.

    Args:
        confidence: Float tensor of shape (n,) containing predicted confidences.
        correctness: Float tensor of shape (n,) containing the true correctness
            labels.
        num_bins: Number of equally sized bins.
        **kwargs: Additional arguments (ignored).

    Returns:
        The ECE/MCE.

    """
    del kwargs

    return calibration_error(
        confidence=confidence,
        correctness=correctness,
        num_bins=num_bins,
        norm=CalibrationErrorNorm.INF,
    )


# --------------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------------


def chi_squared(
    pred_mean: Array,
    pred_std: Array,
    target: Array,
    *,
    averaged: bool = True,
    **kwargs: Kwargs,
) -> Float:
    r"""Estimate the q-value for predictions.

    The $\chi^2$-value is a measure of the squared error normalized by the predicted
    variance.

    Mathematically:

    $$
    \chi^2_{\text{Avg}}
    = \frac{1}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}.
    $$

    Args:
        pred_mean: Array of predicted means.
        pred_std: Array of predicted standard deviations.
        target: Array of ground truth labels.
        averaged: Whether to return the mean or sum of the q-values.
        **kwargs: Additional arguments (ignored).

    Returns:
        The estimated q-value.
    """
    del kwargs
    assert pred_mean.shape == pred_std.shape == target.shape, (
        f"arrays must have the same shape: {pred_mean.shape}, "
        f"{pred_std.shape}, {target.shape}"
    )
    val = jnp.power(pred_mean - target, 2) / jnp.power(pred_std, 2)
    return jnp.mean(val) if averaged else jnp.sum(val)


def chi_squared_zero(**predictions: Kwargs) -> Float:
    r"""Computes a calibration metric for a given set of predictions.

    The calculated metric is the ratio between the error of the prediction and
    the variance of the output uncertainty.

    Args:
        **predictions: Keyword arguments representing the model predictions,
            typically including mean, variance, and target.

    Returns:
        The calibration metric value.
    """
    return jnp.abs(chi_squared(**predictions) - 1)


def estimate_rmse(pred_mean: Array, target: Array, **kwargs: Kwargs) -> Float:
    r"""Estimate the root mean squared error (RMSE) for predictions.

    Mathematically:

    $$
    \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}.
    $$

    Args:
        pred_mean: Array of predicted means.
        target: Array of ground truth labels.
        **kwargs: Additional arguments (ignored).

    Returns:
        The RMSE value.
    """
    del kwargs
    return jnp.sqrt(jnp.mean(jnp.power(pred_mean - target, 2)))


def crps_gaussian(
    pred_mean: Array,
    pred_std: Array,
    target: Array,
    *,
    scaled: bool = True,
    **kwargs: Kwargs,
) -> Float:
    """The negatively oriented continuous ranked probability score for Gaussians.

    Negatively oriented means a smaller value is more desirable.

    Args:
        pred_mean: 1D array of the predicted means for the held out dataset.
        pred_std: 1D array of he predicted standard deviations for the held out dataset.
        target: 1D array of the true labels in the held out dataset.
        scaled: Whether to scale the score by size of held out set.
        **kwargs: Additional arguments (ignored).

    Returns:
        The crps for the heldout set.

    Raises:
        ValueError: pred_mean, pred_std, and target have incompatible shapes.
    """
    del kwargs

    # Ensure input arrays are 1D and of the same shape
    if not (pred_mean.shape == pred_std.shape == target.shape):
        msg = f"arrays must have the same shape : {pred_mean.shape}, {pred_std.shape}, {target.shape}"
        raise ValueError(msg)

    # Compute crps
    pred_std_flat = pred_std.flatten()
    pred_norm = (target.flatten() - pred_mean.flatten()) / pred_std_flat
    term_1 = 1 / jnp.sqrt(jnp.pi)
    term_2 = 2 * jax.scipy.stats.norm.pdf(pred_norm, loc=0, scale=1)
    term_3 = pred_norm * (2 * jax.scipy.stats.norm.cdf(pred_norm, loc=0, scale=1) - 1)

    crps_list = -1 * pred_std_flat * (term_1 - term_2 - term_3)
    crps = jnp.sum(crps_list)

    # Potentially scale so that sum becomes mean
    if scaled:
        crps = crps / len(crps_list)

    return crps


def nll_gaussian(
    pred_mean: Array,
    pred_std: Array,
    target: Array,
    *,
    scaled: bool = True,
    **kwargs: Kwargs,
) -> Float:
    r"""Compute the negative log-likelihood (NLL) for a Gaussian distribution.

    The NLL quantifies how well the predictive distribution fits the data,
    assuming a Gaussian distribution characterized by `pred` (mean) and `pred_std`
    (standard deviation).

    Mathematically:

    $$
    \text{NLL} = - \sum_{i=1}^n \log \left( \frac{1}{\sqrt{2\pi \sigma_i^2}}
    \exp \left( -\frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} \right) \right).
    $$

    Args:
        pred_mean: Array of predicted means for the dataset.
        pred_std: Array of predicted standard deviations for the dataset.
        target: Array of ground truth labels for the dataset.
        scaled: Whether to normalize the NLL by the number of samples (default: True).
        **kwargs: Additional arguments (ignored).

    Returns:
        The computed NLL value.

    Raises:
        ValueError: pred_mean, pred_std, and target have incompatible shapes.
    """
    del kwargs

    # Ensure input arrays are 1D and of the same shape
    if not (pred_mean.shape == pred_std.shape == target.shape):
        msg = f"arrays must have the same shape: {pred_mean.shape}, {pred_std.shape}, {target.shape}"
        raise ValueError(msg)

    # Compute residuals
    residuals = pred_mean - target

    # Compute negative log likelihood
    nll_list = jax.scipy.stats.norm.logpdf(residuals, scale=pred_std)
    nll = -1 * jnp.sum(nll_list)

    # Scale the result by the number of data points if `scaled` is True
    if scaled:
        nll /= math.prod(pred_mean.shape)

    return nll


DEFAULT_REGRESSION_METRICS_DICT = {
    "rmse": estimate_rmse,
    "chi^2": chi_squared,
    "nll": nll_gaussian,
    "crps": crps_gaussian,
}

DEFAULT_REGRESSION_METRICS = [
    apply_fns(
        estimate_rmse,
        chi_squared,
        nll_gaussian,
        crps_gaussian,
        names=["rmse", "chi^2", "nll", "crps"],
        pred_mean="pred_mean",
        pred_std="pred_std",
        target="target",
    )
]

# --------------------------------------------------------------------------------
# Low-rank output covariance metrics (FSP-style)
# --------------------------------------------------------------------------------


def compute_diagonal(pred: dict) -> Array:
    """Return predictive variance from results dict."""
    return pred["pred_var"]


def compute_trace(pred: dict, **kwargs: Kwargs) -> Array:
    """Trace from predictive variance (sum of diagonal)."""
    axis = kwargs.get("axis", -1)
    return jnp.sum(compute_diagonal(pred), axis=axis)


def mean_eigenvalue(pred_cov_low_rank_terms, **kwargs):
    del kwargs
    # Extract values
    eig_vals = pred_cov_low_rank_terms.S  # Lambda

    return jnp.mean(jnp.abs(eig_vals))


def low_rank_mahalanobis_distance_inverse_covariance(
    pred_mean: Array,
    target: Array,
    pred_cov_low_rank_terms: LowRankTerms,
    observation_noise: Array,
    **kwargs: Kwargs,
) -> Float:
    """Compute sqrt((μ–t)ᵀ Σ⁻¹ (μ–t)) for
    Σ = U diag(S²) Uᵀ + σ² I using Woodbury.

    Here, `S` are the singular values of the scale, so the covariance
    eigenvalues are `S²`.
    """
    del kwargs
    U = pred_cov_low_rank_terms.U  # (D, k)
    S = pred_cov_low_rank_terms.S  # (k,)
    sigma = jnp.exp(observation_noise)

    v = (pred_mean - target).reshape(-1)
    w = U.T @ v  # (k,)

    # Σ⁻¹ = (1/σ²)[I – U diag(S²/(S²+σ²)) Uᵀ]
    frac = (S**2) / (S**2 + sigma**2)
    quad = (jnp.dot(v, v) - jnp.sum(frac * (w**2))) / (sigma**2)
    return jnp.sqrt(jnp.maximum(quad, 0.0))


def low_rank_log_determinant(
    pred_cov_low_rank_terms: LowRankTerms, observation_noise: Array, **kwargs: Kwargs
) -> Float:
    """Compute log|Σ| for Σ = U diag(S²) Uᵀ + σ² I."""
    del kwargs
    U = pred_cov_low_rank_terms.U
    S = pred_cov_low_rank_terms.S
    sigma = jnp.exp(observation_noise)
    n, k = U.shape
    eigenspace_logdet = jnp.sum(jnp.log(S**2 + sigma**2))
    complement_logdet = (n - k) * jnp.log(sigma**2)
    return eigenspace_logdet + complement_logdet


def low_rank_nlpd_per_input(
    pred_mean: Array,
    target: Array,
    pred_cov_low_rank_terms: LowRankTerms,
    observation_noise: Array,
    **kwargs: Kwargs,
) -> dict[str, Float]:
    """Negative log predictive density (vector form) using low-rank Σ.

    Returns a dict with `nlpd_per_input` and `mahalanobis_distance`.
    """
    del kwargs

    log_det_term = low_rank_log_determinant(
        pred_cov_low_rank_terms=pred_cov_low_rank_terms,
        observation_noise=observation_noise,
    )
    mah = low_rank_mahalanobis_distance_inverse_covariance(
        pred_mean=pred_mean,
        target=target,
        pred_cov_low_rank_terms=pred_cov_low_rank_terms,
        observation_noise=observation_noise,
    ) / jnp.exp(observation_noise)
    nlpd = 0.5 * (log_det_term + mah**2 + jnp.log(2 * jnp.pi))

    return {"nlpd_per_input": nlpd, "mahalanobis_distance": mah}


def low_rank_marginal_nlpd_per_input(
    pred_mean: Array,
    pred_var: Array,
    target: Array,
    observation_noise: Array,
    **kwargs: Kwargs,
) -> dict[str, Float]:
    """Marginal NLPD using only the predictive variance and isotropic noise."""
    del kwargs
    mse_term = jnp.mean((pred_mean - target) ** 2)
    trace_term = jnp.mean(pred_var)
    sigma = jnp.exp(observation_noise)
    log_term = 0.5 * jnp.log(2 * jnp.pi * sigma**2) / pred_var.shape[-1]
    marginal_nlpd = 0.5 * (trace_term + mse_term / sigma**2 + log_term)

    return {"marginal_nlpd_per_input": marginal_nlpd, "mse_per_input": mse_term}


def cov_low_rank_approximation(results: dict, aux: dict, **kwargs: Kwargs):
    """Approximate a dense predictive covariance with LowRankTerms via Lanczos.

    Adds `low_rank_terms` to results if not present, and also returns
    `pred_cov_low_rank_terms` for convenience. Eigenvalues are clipped and
    square-rooted so that `S` represents scale singular values (matching the
    convention used in low-rank pushforward), i.e., Σ ≈ U diag(S²) Uᵀ.
    """
    cov_pred = results.get("pred_cov")
    if cov_pred is None:
        return results, aux

    if kwargs.get("fsp"):
        results["pred_cov_low_rank_terms"] = aux["pred_cov_low_rank_terms"]
        return results, aux

    cov_rank = kwargs.get("cov_rank", kwargs.get("rank", 100))
    lr = lanczos_lowrank(cov_pred, rank=int(cov_rank))

    # Clip negatives and convert eigenvalues (λ) to singular values (S = sqrt(λ)).
    S_cov = jnp.where(lr.S > 0.0, lr.S, 0.0)
    lr_fixed = LowRankTerms(
        U=lr.U, S=jnp.sqrt(S_cov), scalar=jnp.asarray(0.0, S_cov.dtype)
    )

    results.setdefault("low_rank_terms", lr_fixed)
    results["pred_cov_low_rank_terms"] = lr_fixed
    return results, aux


LOW_RANK_REGRESSION_METRICS_DICT = {
    "rmse": estimate_rmse,
    "mahalanobis_distance": low_rank_mahalanobis_distance_inverse_covariance,
    "log_determinant": low_rank_log_determinant,
    "nlpd": low_rank_nlpd_per_input,
    "marginal_nlpd": low_rank_marginal_nlpd_per_input,
}

LOW_RANK_REGRESSION_METRICS = [
    apply_fns(
        estimate_rmse,
        low_rank_mahalanobis_distance_inverse_covariance,
        low_rank_log_determinant,
        low_rank_nlpd_per_input,
        low_rank_marginal_nlpd_per_input,
        names=[
            "rmse",
            "mahalanobis_distance",
            "log_determinant",
            "nlpd",
            "marginal_nlpd",
        ],
        pred_mean="pred_mean",
        pred_cov_low_rank_terms="low_rank_terms",
        observation_noise="observation_noise",
        pred_var="pred_var",
        target="target",
    )
]
