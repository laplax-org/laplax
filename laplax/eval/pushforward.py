# pushforward.py

"""Pushforward Functions for Weight Space Uncertainty.

This module provides functions to propagate uncertainty in weight space to
output uncertainty. It includes methods for ensemble-based Monte Carlo
predictions and linearized approximations for uncertainty estimation, as well as to
create the posterior_gp_kernel.
"""

import math

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.cov import Posterior
from laplax.eval.predictives import (
    laplace_bridge,
    mean_field_0_predictive,
    mean_field_1_predictive,
    mean_field_2_predictive,
)
from laplax.eval.utils import finalize_fns
from laplax.types import (
    Any,
    Array,
    Callable,
    DistState,
    Float,
    InputArray,
    Int,
    KeyType,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
    PriorArguments,
)
from laplax.util.ops import precompute_list
from laplax.util.tree import add

# -------------------------------------------------------------------------
# Utilities - General
# -------------------------------------------------------------------------


def set_get_weight_sample(key, mean_params, scale_mv, num_samples, **kwargs):
    """Creates a function to sample weights from a Gaussian distribution.

    This function generates weight samples from a Gaussian distribution
    characterized by the mean and the scale matrix-vector product function.
    It supports precomputation of samples for efficiency and assumes a fixed
    number of required samples.

    Args:
        key: PRNG key for generating random samples.
        mean_params: Mean of the weight-space Gaussian distribution.
        scale_mv: Function for the scale matrix-vector product.
        num_samples: Number of weight samples to generate.
        **kwargs: Additional arguments, including:
            - `set_get_weight_sample_precompute`: Controls whether samples are
              precomputed.

    Returns:
        Callable: A function that generates a specific weight sample by index.
    """
    keys = jax.random.split(key, num_samples)

    def get_weight_sample(idx):
        return util.tree.normal_like(
            keys[idx],
            mean=mean_params,
            scale_mv=scale_mv,
            rank=kwargs.get("rank"),  # If scale_mv assumes lower rank.
        )

    return precompute_list(
        get_weight_sample,
        jnp.arange(num_samples),
        precompute=kwargs.get(
            "set_get_weight_sample_precompute", kwargs.get("precompute")
        ),
        **kwargs,
    )


def special_pred_act(
    results: dict[str, Array],
    aux: dict[str, Any],
    linearized: bool,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    var_pred_dict = {
        "laplace_bridge": laplace_bridge,
        "mean_field_0": mean_field_0_predictive,
        "mean_field_1": mean_field_1_predictive,
    }
    cov_pred_dict = {"mean_field_2": mean_field_2_predictive}

    if "pred_mean" not in results:
        pred_mean_fn = lin_pred_mean if linearized else nonlin_pred_mean
        results, aux = pred_mean_fn(results, aux, **kwargs)

    pred_mean = results["pred_mean"]

    special_pred_type = kwargs.get("special_pred_type", "mean_field_0")

    if special_pred_type in var_pred_dict:
        pred_fn = var_pred_dict[special_pred_type]

        if "pred_var" not in results:  # Fall back to `lin_pred_var`
            results, aux = lin_pred_var(results, aux, **kwargs)

        pred_var = results["pred_var"]

        pred = pred_fn(
            pred_mean, pred_var, use_correction=kwargs.get("use_correction", True)
        )
    else:  # special_pred_type in cov_pred_dict
        pred_fn = cov_pred_dict[special_pred_type]

        if "pred_cov" not in results:
            pred_cov = lin_pred_cov if linearized else nonlin_pred_cov
            results, aux = pred_cov(results, aux)

        pred_cov = results["pred_cov"]

        pred = pred_fn(pred_mean, pred_cov)

    results["special_pred_act"] = pred

    return results, aux


# -------------------------------------------------------------------------
# Posterior state to distribution state
# -------------------------------------------------------------------------


def get_dist_state(
    mean_params: Params,
    model_fn: ModelFn,
    posterior_state: PosteriorState,
    *,
    linearized: bool = False,
    num_samples: int = 0,
    key: KeyType | None = None,
    **kwargs,
) -> DistState:
    """Construct the distribution state for uncertainty propagation.

    The distribution state contains information needed to propagate uncertainty
    from the posterior over weights to predictions. It forms the state for both
    linearized and ensemble-based Monte Carlo approaches.

    Args:
        mean_params: Mean of the posterior (model parameters).
        model_fn: The model function to evaluate.
        posterior_state: The posterior distribution state.
        linearized: Whether to consider a linearized approximation.
        num_samples: Number of weight samples for Monte Carlo methods.
        key: PRNG key for generating random samples.
        **kwargs: Additional arguments, including:
            - `set_get_weight_sample_precompute`.

    Returns:
        DistState: A dictionary containing functions and parameters for uncertainty
        propagation.
    """
    dist_state = {
        "posterior_state": posterior_state,
        "num_samples": num_samples,
    }

    if linearized:
        # Create pushforward functions
        def pf_jvp(input: InputArray, vector: Params) -> PredArray:
            return jax.jvp(
                lambda p: model_fn(input=input, params=p),
                (mean_params,),
                (vector,),
            )[1]

        def pf_vjp(input: InputArray, vector: PredArray) -> Params:
            out, vjp_fun = jax.vjp(
                lambda p: model_fn(input=input, params=p), mean_params
            )
            return vjp_fun(vector.reshape(out.shape))

        batched_computation = kwargs.get("batched_computation", False)
        if not batched_computation:
            dist_state["vjp"] = pf_vjp
            dist_state["jvp"] = pf_jvp
        else:
            dist_state["vjp"] = jax.vmap(pf_vjp, in_axes=(0, 0), out_axes=0)
            dist_state["jvp"] = jax.vmap(pf_jvp, in_axes=(0, None), out_axes=0)

    if num_samples > 0:
        weight_sample_mean = (
            util.tree.zeros_like(mean_params) if linearized else mean_params
        )

        # Create weight sample function
        get_weight_samples = set_get_weight_sample(
            key,
            mean_params=weight_sample_mean,
            scale_mv=posterior_state.scale_mv(posterior_state.state),
            num_samples=num_samples,
            rank=posterior_state.rank,
            **kwargs,
        )
        dist_state["get_weight_samples"] = get_weight_samples

    return dist_state


# -------------------------------------------------------------------------
# Utilities - Ensemble pushforward
# -------------------------------------------------------------------------


def nonlin_setup(
    results: dict[str, Array],
    aux: dict[str, Any],
    input: InputArray,
    dist_state: DistState,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Prepare ensemble-based Monte Carlo predictions.

    This function generates predictions for multiple weight samples and stores
    them in the auxiliary dictionary.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data, including the model function.
        input: Input data for prediction.
        dist_state: Distribution state containing weight sampling functions.
        **kwargs: Additional arguments, including:
            - `nonlin_setup_batch_size`: Controls batch size for computing predictions.

    Returns:
        tuple: Updated `results` and `aux`.
    """

    def compute_pred_ptw(idx: int) -> PredArray:
        weight_sample = dist_state["get_weight_samples"](idx)
        return aux["model_fn"](input=input, params=weight_sample)

    aux["pred_ensemble"] = jax.lax.map(
        compute_pred_ptw,
        jnp.arange(dist_state["num_samples"]),
        batch_size=kwargs.get(
            "nonlin_setup_batch_size", kwargs.get("weight_batch_size")
        ),
    )

    return results, aux


def nonlin_pred_mean(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute the mean of ensemble predictions.

    This function calculates the mean of prediction ensemble generated from
    multiple weight samples in an ensemble-based Monte Carlo approach.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the prediction ensemble.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    del kwargs

    pred_ensemble = aux["pred_ensemble"]
    results["pred_mean"] = util.tree.mean(pred_ensemble, axis=0)
    return results, aux


def nonlin_pred_cov(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute the covariance of ensemble predictions.

    This function calculates the empirical covariance of the ensemble of predictions.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the prediction ensemble.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    del kwargs

    pred_ensemble = aux["pred_ensemble"]

    results["pred_cov"] = util.tree.cov(
        pred_ensemble.reshape(pred_ensemble.shape[0], -1), rowvar=False
    )
    return results, aux


def nonlin_pred_var(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute the variance of ensemble predictions.

    This function calculates the empirical variance of the ensemble of predictions.
    If the covariance is already available, it extracts the diagonal.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the prediction ensemble.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    del kwargs

    if "pred_cov" in results:
        pred_cov = results["pred_cov"]
        pred_var = (
            jnp.diagonal(pred_cov)
            if pred_cov.ndim >= 2
            else jnp.reshape(pred_cov, (-1,))
        )
        pred_var = pred_var.reshape(aux["pred_ensemble"].shape[1:])
        results["pred_var"] = pred_var
    else:
        pred_ensemble = aux["pred_ensemble"]
        results["pred_var"] = util.tree.var(pred_ensemble, axis=0)
    return results, aux


def nonlin_pred_std(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute the standard deviation of ensemble predictions.

    This function calculates the empirical standard deviation of the ensemble of
    predictions. If the variance is already available, then it takes the square root.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the prediction ensemble.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    del kwargs

    if "pred_var" in results:
        results["pred_std"] = jnp.sqrt(results["pred_var"])
    else:
        pred_ensemble = aux["pred_ensemble"]
        results["pred_std"] = util.tree.std(pred_ensemble, axis=0)
    return results, aux


def nonlin_samples(
    results: dict[str, Array],
    aux: dict[str, Any],
    num_samples: int = 5,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Select samples from ensemble.

    This function selects a subset of samples from the ensemble of predictions.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the prediction ensemble.
        num_samples: Number of samples to select.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    del kwargs

    pred_ensemble = aux["pred_ensemble"]
    results["samples"] = util.tree.tree_slice(pred_ensemble, 0, num_samples)
    return results, aux


def nonlin_special_pred_act(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Apply special predictive methods to nonlinear Laplace for classification.

    This function applies special predictive methods (Laplace Bridge, Mean Field-0,
    Mean Field-1, or Mean Field-2) to nonlinear Laplace for classification. These
    methods transform the predictions into probability space using specific formulations
    rather than Monte Carlo sampling.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing prediction information.
        **kwargs: Additional arguments, including:
            - `special_pred_type`: Type of special prediction ("laplace_bridge",
              "mean_field_0", "mean_field_1", or "mean_field_2")
            - `use_correction`: Whether to apply correction term for applicable methods.

    Returns:
        tuple: Updated `results` and `aux`.
    """
    return special_pred_act(results, aux, linearized=False, **kwargs)


def nonlin_mc_pred_act(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute Monte Carlo predictions for nonlinear Laplace classification.

    This function generates Monte Carlo predictions for classification by averaging
    softmax probabilities across different weight samples. If samples are not already
    available, it generates them first.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing prediction information.
        **kwargs: Additional arguments passed to sample generation.

    Returns:
        tuple: Updated `results` and `aux`.
    """
    if "samples" not in results:
        results, aux = nonlin_samples(results=results, aux=aux, **kwargs)

    results["mc_pred_act"] = jnp.mean(
        jax.nn.softmax(results["samples"], axis=1), axis=0
    )

    return results, aux


DEFAULT_NONLIN_FINALIZE_FNS = [
    nonlin_setup,
    nonlin_pred_mean,
    nonlin_pred_cov,
    nonlin_pred_var,
    nonlin_pred_std,
    nonlin_samples,
]

# -------------------------------------------------------------------------
# Utilities - Linearized pushforward
# -------------------------------------------------------------------------


def set_output_mv(
    posterior_state: Posterior,
    input: InputArray,
    jvp: Callable[[InputArray, Params], PredArray],
    vjp: Callable[[InputArray, PredArray], Params],
):
    """Create matrix-vector product functions for output covariance and scale.

    This function propagates uncertainty from weight space to output space by
    constructing matrix-vector product functions for the output covariance and
    scale matrices. These functions utilize the posterior's covariance and scale
    operators in conjunction with Jacobian-vector products (JVP) and
    vector-Jacobian products (VJP).

    Args:
        posterior_state: The posterior state containing covariance and scale operators.
        input: Input data for the model.
        jvp: Function for computing Jacobian-vector products.
        vjp: Function for computing vector-Jacobian products.

    Returns:
        dict: A dictionary with:
            - `cov_mv`: Function for the output covariance matrix-vector product.
            - `jac_mv`: Function for the JVP with a fixed input.
            - `low_rank_terms`: Low-rank terms from the posterior state.
    """
    cov_mv = posterior_state.cov_mv(posterior_state.state)

    def output_cov_mv(vec: PredArray) -> PredArray:
        return jvp(input, cov_mv(vjp(input, vec)[0]))

    def output_jac_mv(vec: PredArray) -> PredArray:
        return jvp(input, vec)

    low_rank_terms = posterior_state.low_rank_terms

    return {
        "cov_mv": output_cov_mv,
        "jac_mv": output_jac_mv,
        "low_rank_terms": low_rank_terms,
    }


def lin_setup(
    results: dict[str, Array],
    aux: dict[str, Any],
    input: InputArray,
    dist_state: DistState,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Prepare linearized pushforward functions for uncertainty propagation.

    This function sets up matrix-vector product functions for the output covariance
    and scale matrices in a linearized pushforward framework. It verifies the
    validity of input components (posterior state, JVP, and VJP) and stores the
    resulting functions in the auxiliary dictionary.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data to store matrix-vector product functions.
        input: Input data for the model.
        dist_state: Distribution state containing posterior state, JVP, and VJP
            functions.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """  # noqa: DOC501
    del kwargs

    jvp = dist_state["jvp"]
    vjp = dist_state["vjp"]
    posterior_state = dist_state["posterior_state"]

    if not isinstance(posterior_state, Posterior):
        msg = "posterior state is not a Posterior type"
        raise TypeError(msg)

    if not isinstance(jvp, Callable):
        msg = "JVP is not a JVPType"
        raise TypeError(msg)

    if not isinstance(vjp, Callable):
        msg = "VJP is not a VJPType"
        raise TypeError(msg)

    mv = set_output_mv(posterior_state, input, jvp, vjp)
    aux["cov_mv"] = mv["cov_mv"]
    aux["jac_mv"] = mv["jac_mv"]
    results["low_rank_terms"] = mv["low_rank_terms"]

    return results, aux


def lin_pred_mean(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Restore the linearized predictions.

    This function extracts the prediction from the results dictionary and
    stores it.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data (ignored).
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.

    Note:
        This function is used for the linearized mean prediction.
    """
    del kwargs

    results["pred_mean"] = results["map"]
    return results, aux


def lin_pred_var(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute and store the variance of the linearized predictions.

    This function calculates the variance of predictions by extracting the diagonal
    of the output covariance matrix.

    Args:
        results: Dictionary containing computed results.
        aux: Auxiliary data, including covariance matrix functions.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """
    cov = results.get("pred_cov", aux["cov_mv"])

    if "pred_mean" not in results:
        results, aux = lin_pred_mean(results, aux, **kwargs)

    pred_mean = results["pred_mean"]

    pred_var = util.mv.diagonal(
        cov,
        layout=math.prod(pred_mean.shape),
        mv_jittable=kwargs.get("mv_jittable", True),
        low_rank=False,
    )
    results["pred_var"] = pred_var.reshape(pred_mean.shape)

    return results, aux


def lin_pred_std(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute and store the standard deviation of the linearized predictions.

    This function calculates the standard deviation by taking the square root
    of the predicted variance.

    Args:
        results: Dictionary containing computed results.
        aux: Auxiliary data (ignored).
        **kwargs: Additional arguments.

    Returns:
        tuple: Updated `results` and `aux`.
    """
    if "pred_var" not in results:  # Fall back to `lin_pred_var`
        results, aux = lin_pred_var(results, aux, **kwargs)

    var = results["pred_var"]
    results["pred_std"] = util.tree.sqrt(jnp.abs(var))  # abs to avoid numerical issues
    return results, aux


def lin_pred_cov(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute and store the covariance of the linearized predictions.

    This function computes the full output covariance matrix in dense form
    using the covariance matrix-vector product function.

    Args:
        results: Dictionary containing computed results.
        aux: Auxiliary data containing covariance matrix-vector product functions.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.

    Raises:
        TypeError: If the covariance matrix-vector product function is invalid.
    """  # noqa: DOC502
    if "pred_mean" not in results:
        results, aux = lin_pred_mean(results, aux, **kwargs)

    pred_mean = results["pred_mean"]
    cov_mv = aux["cov_mv"]

    results["pred_cov"] = util.mv.to_dense(cov_mv, layout=pred_mean)
    return results, aux


def lin_pred_lsqrt_low_rank_cov(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    r"""Attach low-rank predictive info and cheap diagonal.

    - Keeps `low_rank_terms` (computed in lin_setup when low_rank=True).
    - Computes `pred_var` from the low-rank factors without densifying.
      For Σ = U diag(S²) Uᵀ + σ² I with U ∈ R^{D \times k},
      diag(Σ) = σ² 1 + Σ_j (S_j U[:, j])².

    Returns:
        tuple: Updated `results` and `aux`.
    """
    if "pred_mean" not in results:
        results, aux = lin_pred_mean(results, aux, **kwargs)

    if "low_rank_terms" in results and "observation_noise" in results:
        lr = results["low_rank_terms"]
        sigma = jnp.exp(results["observation_noise"])
        # sum over columns: (S * U)^2 for low-rank diag
        U_scaled = lr.U * lr.S
        pred_var = jnp.sum(U_scaled * U_scaled, axis=1) + sigma**2
        pred_mean = results["pred_mean"]
        results["pred_var"] = pred_var.reshape(pred_mean.shape)

    return results, aux


def lin_samples(
    results: dict[str, Array],
    aux: dict[str, Any],
    dist_state: DistState,
    **kwargs,
):
    """Generate and store samples from the linearized distribution.

    This function computes samples in the output space by applying the scale
    matrix to weight samples generated from the posterior distribution.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing the scale matrix function.
        dist_state: Distribution state containing sampling functions and sample count.
        **kwargs: Additional arguments, including:
            - `lin_samples_batch_size`: Batch size for computing samples.

    Returns:
        tuple: Updated `results` and `aux`.

    Raises:
        TypeError: If the scale matrix or sampling functions are invalid.
    """  # noqa: DOC502
    if "pred_mean" not in results:
        results, aux = lin_pred_mean(results, aux, **kwargs)

    # Unpack arguments
    jac_mv = aux["jac_mv"]
    get_weight_samples = dist_state["get_weight_samples"]
    num_samples = dist_state["num_samples"]

    # Compute samples
    results["samples"] = jax.lax.map(
        lambda i: add(results["pred_mean"], jac_mv(get_weight_samples(i))),
        jnp.arange(num_samples),
        batch_size=kwargs.get(
            "lin_samples_batch_size", kwargs.get("weight_batch_size")
        ),
    )
    return results, aux


def lin_special_pred_act(
    results: dict[str, Array],
    aux: dict[str, Any],
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Apply special predictive methods to linearized Laplace for classification.

    This function applies special predictive methods (Laplace Bridge, Mean Field-0,
    Mean Field-1, or Mean Field-2) to linearized Laplace for classification. These
    methods transform the predictions into probability space using specific formulations
    rather than Monte Carlo sampling.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing prediction information.
        **kwargs: Additional arguments, including:
            - `special_pred_type`: Type of special prediction ("laplace_bridge",
              "mean_field_0", "mean_field_1", or "mean_field_2")
            - `use_correction`: Whether to apply correction term for applicable methods.

    Returns:
        tuple: Updated `results` and `aux`.
    """
    return special_pred_act(results, aux, linearized=True, **kwargs)


def lin_mc_pred_act(
    results: dict[str, Array], aux: dict[str, Any], **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute Monte Carlo predictions for linear Laplace classification.

    This function generates Monte Carlo predictions for classification by averaging
    softmax probabilities across different weight samples. If samples are not already
    available, it generates them first.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data containing prediction information.
        **kwargs: Additional arguments passed to sample generation.

    Returns:
        tuple: Updated `results` and `aux`.
    """
    if "samples" not in results:
        results, aux = lin_samples(results=results, aux=aux, **kwargs)

    results["mc_pred_act"] = jnp.mean(
        jax.nn.softmax(results["samples"], axis=1), axis=0
    )

    return results, aux


DEFAULT_LIN_FINALIZE_FNS = [
    lin_setup,
    lin_pred_mean,
    lin_pred_cov,
    lin_pred_lsqrt_low_rank_cov,
    lin_pred_var,
    lin_pred_std,
    lin_samples,
]

# A leaner finalize chain for FSP-style metrics that avoids dense covariance.
DEFAULT_LIN_FSP_FINALIZE_FNS = [
    lin_setup,
    lin_pred_mean,
    lin_pred_lsqrt_low_rank_cov,
    lin_pred_std,
]


# -------------------------------------------------------------------------
# Pushforward functions
# -------------------------------------------------------------------------


def set_prob_predictive(
    model_fn: ModelFn,
    mean_params: Params,
    dist_state: DistState,
    pushforward_fns: list[Callable],
    **kwargs,
) -> Callable[[InputArray], dict[str, Array]]:
    """Create a probabilistic predictive function.

    This function generates a predictive callable that computes uncertainty-aware
    predictions using a set of pushforward functions. The generated function can
    evaluate mean predictions and propagate uncertainty from the posterior over
    weights to output space.

    Args:
        model_fn: The model function to evaluate, which takes input and parameters.
        mean_params: The mean of the posterior distribution over model parameters.
        dist_state: The distribution state for uncertainty propagation, containing
            functions and parameters related to the posterior.
        pushforward_fns: A list of pushforward functions, such as mean, variance, and
            covariance.
        **kwargs: Additional arguments passed to the pushforward functions.

    Returns:
        Callable: A function that takes an input array and returns a dictionary
        of predictions and uncertainty metrics.
    """

    def prob_predictive(input: InputArray) -> dict[str, Array]:
        pred_map = model_fn(input=input, params=mean_params)
        aux = {"model_fn": model_fn, "mean_params": mean_params}

        results = {"map": pred_map}

        return finalize_fns(
            fns=pushforward_fns,
            results=results,
            dist_state=dist_state,
            aux=aux,
            input=input,
            **kwargs,
        )

    return prob_predictive


def set_nonlin_pushforward(
    model_fn: ModelFn,
    mean_params: Params,
    posterior_fn: Callable[[PriorArguments, Int], Posterior],
    prior_arguments: PriorArguments,
    *,
    key: KeyType,
    loss_scaling_factor: Float = 1.0,
    pushforward_fns: list = DEFAULT_NONLIN_FINALIZE_FNS,
    num_samples: int = 100,
    **kwargs,
):
    """Construct a Monte Carlo pushforward predictive function.

    This function creates a probabilistic predictive callable that computes
    ensemble-based Monte Carlo (MC) predictions and propagates uncertainty
    from weight space to output space using sampling.

    Args:
        model_fn: The model function to evaluate, which takes input and parameters.
        mean_params: The mean of the posterior distribution over model parameters.
        posterior_fn: A callable that generates the posterior state from prior
            arguments.
        prior_arguments: Arguments for defining the prior distribution.
        key: PRNG key for generating random samples.
        loss_scaling_factor: Factor by which the user-provided loss function is scaled.
            Defaults to 1.0.
        pushforward_fns: A list of Monte Carlo pushforward functions
            (default: `DEFAULT_MC_FUNCTIONS`).
        num_samples: Number of weight samples for Monte Carlo predictions.
        **kwargs: Additional arguments passed to the pushforward functions.

    Returns:
        Callable: A probabilistic predictive function that computes predictions
        and uncertainty metrics using Monte Carlo sampling.
    """
    # Create weight sample function
    posterior_state = posterior_fn(prior_arguments, loss_scaling_factor)

    # Posterior state to dist_state
    dist_state = get_dist_state(
        mean_params,
        model_fn,
        posterior_state,
        linearized=False,
        num_samples=num_samples,
        key=key,
        **kwargs,
    )

    # Set prob predictive
    prob_predictive = set_prob_predictive(
        model_fn=model_fn,
        mean_params=mean_params,
        dist_state=dist_state,
        pushforward_fns=pushforward_fns,
        **kwargs,
    )

    return prob_predictive


def set_lin_pushforward(
    model_fn: ModelFn,
    mean_params: Params,
    posterior_fn: Callable[[PriorArguments, Int], Posterior],
    prior_arguments: PriorArguments,
    loss_scaling_factor: Float = 1.0,
    pushforward_fns: list = DEFAULT_LIN_FINALIZE_FNS,
    **kwargs,
) -> Callable:
    """Construct a linearized pushforward predictive function.

    This function generates a probabilistic predictive callable that computes
    predictions and propagates uncertainty using a linearized approximation of
    the model function.

    Args:
        model_fn: The model function to evaluate, which takes input and parameters.
        mean_params: The mean of the posterior distribution over model parameters.
        posterior_fn: A callable that generates the posterior state from prior
            arguments.
        prior_arguments: Arguments for defining the prior distribution.
        loss_scaling_factor: Factor by which the user-provided loss function is scaled.
            Defaults to 1.0.
        pushforward_fns: A list of linearized pushforward functions
            (default: `DEFAULT_LIN_FINALIZE`).
        **kwargs: Additional arguments passed to the pushforward functions, including:
            - `n_samples`: Number of samples for approximating uncertainty metrics.
            - `batched_computation`: Whether to use batched computation.
            - `key`: PRNG key for generating random samples.

    Returns:
        Callable: A probabilistic predictive function that computes predictions
        and uncertainty metrics using a linearized approximation.
    """
    posterior_state = posterior_fn(prior_arguments, loss_scaling_factor)

    dist_state = get_dist_state(
        mean_params,
        model_fn,
        posterior_state,
        linearized=True,
        **kwargs,
    )

    prob_predictive = set_prob_predictive(
        model_fn=model_fn,
        mean_params=mean_params,
        dist_state=dist_state,
        pushforward_fns=pushforward_fns,
        **kwargs,
    )

    return prob_predictive


def set_posterior_gp_kernel(
    model_fn: ModelFn,
    mean: Params,
    posterior_fn: Callable[[PriorArguments, Int], Posterior],
    prior_arguments: PriorArguments,
    loss_scaling_factor: Float = 1.0,
    **kwargs,
) -> tuple[Callable, DistState]:
    """Construct a kernel matrix-vector product function for a posterior GP.

    This function generates a callable for the kernel matrix-vector product (MVP)
    in a posterior GP framework. The kernel MVP is constructed using the posterior
    state and propagates uncertainty in weight space to output space via linearization.
    The resulting kernel MVP can optionally return a dense matrix representation.

    Args:
        model_fn: The model function to evaluate, which takes input and parameters.
        mean: The mean of the posterior distribution over model parameters.
        posterior_fn: A callable that generates the posterior state from prior
            arguments.
        prior_arguments: Arguments for defining the prior distribution.
        loss_scaling_factor: Factor by which the user-provided loss function is scaled.
            Defaults to 1.0.
        **kwargs: Additional arguments, including:
            - `dense`: Whether to return a dense kernel matrix instead of the MVP.
            - `output_layout`: The layout of the dense kernel matrix (required if
                `dense` is True).

    Returns:
        tuple: A kernel MVP callable or a dense kernel matrix function, and the
        distribution state containing posterior information.

    Raises:
        ValueError: If `dense` is True but `output_layout` is not specified.
    """
    posterior_state = posterior_fn(prior_arguments, loss_scaling_factor)

    dist_state = get_dist_state(
        mean,
        model_fn,
        posterior_state,
        linearized=True,
        num_samples=0,
        **kwargs,
    )

    def kernel_mv(
        vec: PredArray, x1: InputArray, x2: InputArray, dist_state: dict[str, Any]
    ) -> PredArray:
        cov_mv = dist_state["posterior_state"].cov_mv(
            dist_state["posterior_state"].state
        )
        return dist_state["jvp"](x1, cov_mv(dist_state["vjp"](x2, vec)[0]))

    if kwargs.get("dense"):
        output_layout = kwargs.get("output_layout")
        if output_layout:
            return lambda x1, x2: util.mv.to_dense(
                lambda v: kernel_mv(v, x1, x2, dist_state), layout=output_layout
            ), dist_state
        msg = "function should return a dense matrix, but no output layout is specified"
        raise ValueError(msg)

    return kernel_mv, dist_state
