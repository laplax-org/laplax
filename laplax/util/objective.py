# /laplax/util/objective.py

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from laplax.types import Array, Float, KernelFn, ModelFn, Params

_LL_RHO_KEY = "ll_rho"
_BASE_KEY = "base"


@jax.jit
def compute_gaussian_log_likelihood(
    f_hat: ModelFn, y: Array, ll_scale: Array | float, n_samples: int
) -> Array:
    """Compute log-likelihood with Gaussian likelihood.

    Args:
        f_hat: Model predictions (batch_size, output_dim)
        y: Target values (batch_size, output_dim)
        ll_scale: Likelihood scale (standard deviation)
        n_samples: Total number of training samples (for scaling)

    Returns:
        Log-likelihood value (scalar)
    """
    # Sum over batch/output dims, then scale by N/batch_size
    batch_ll = jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).sum()

    batch_size = f_hat.shape[0]
    return batch_ll * (n_samples / batch_size)


@partial(jax.jit, static_argnames=("normalize",))
def compute_rkhs_norm(
    f_hat: Array,
    prior_mean: Array,
    prior_cov: Array,
    jitter: float = 1e-4,
    normalize: str = "mean",
) -> Array:
    """Compute a dense RKHS norm over context points.

    The context axis is assumed to be the first dimension. All remaining dimensions
    are flattened into a single output axis before solving with the kernel Cholesky.

    Parameters
    ----------
    f_hat : Array
        Model predictions with shape ``(M, ...)``.
    prior_mean : Array
        Prior mean with the same shape as ``f_hat``.
    prior_cov : Array
        Context kernel matrix with shape ``(M, M)``.
    jitter : float, optional
        Stabilizing diagonal jitter added to ``prior_cov``.

    Returns:
    Array
        Scalar RKHS norm ``sum_q ||L^{-1}(f_q - m_q)||_2^2`` where
        ``L L^T = prior_cov + jitter I``.
    """
    diff = f_hat - prior_mean
    diff_flat = diff.reshape(diff.shape[0], -1)  # (M, Q)
    eye = jnp.eye(prior_cov.shape[0], dtype=prior_cov.dtype)
    chol = jnp.linalg.cholesky(prior_cov + jitter * eye)
    alpha = jsp.linalg.solve_triangular(chol, diff_flat, lower=True)  # (M, Q)

    sq = jnp.sum(alpha**2)  # = sum_q f_q^T K^{-1} f_q

    if normalize == "mean":
        sq = sq / alpha.size  # alpha.size = M * Q

    return sq


def create_loss_nll(
    model_fn: ModelFn,
    dataset_size: int | None = None,
):
    """Create a Gaussian negative log-likelihood loss factory.

    Parameters
    ----------
    model_fn : ModelFn
        Model function with Laplax signature ``model_fn(input=..., params=...)``.
    dataset_size : int | None, optional
        Total training set size used for minibatch scaling. If ``None``, the
        current batch size is used.

    Returns:
    Callable
        Loss callable with signature
        ``loss_nll(data, params, scale) -> scalar``.
    """

    def loss_nll(
        data: Mapping[str, Array], params: Params, scale: Float | Array | None = None
    ) -> Array:
        if scale is None:
            msg = "`scale` must be provided for Gaussian NLL."
            raise ValueError(msg)
        preds = model_fn(input=data["input"], params=params)
        n_data = data["input"].shape[0] if dataset_size is None else dataset_size
        return -compute_gaussian_log_likelihood(preds, data["target"], scale, n_data)

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: Array,
    prior_cov_kernel,
    *,
    has_batch_dim: bool = True,
    jitter: float = 1e-4,
):
    """Create an RKHS regularization loss factory.

    Parameters
    ----------
    model_fn : ModelFn
        Model function with Laplax signature ``model_fn(input=..., params=...)``.
    prior_mean : Array
        Prior mean at context points with shape ``(M, ...)``.
    prior_cov_kernel : Callable
        Kernel callable. For batched mode it is called as
        ``prior_cov_kernel(context, grid)`` if ``grid`` is provided in the mapping,
        otherwise as ``prior_cov_kernel(context, context)``.
    has_batch_dim : bool, optional
        If ``True``, expects a mapping context input with key ``"context"`` (and
        optionally ``"grid"``). If ``False``, expects context points directly.
    jitter : float, optional
        Jitter passed to ``compute_rkhs_norm``.

    Returns:
    Callable
        Loss callable with signature ``loss_reg(context_points, params) -> scalar``.
    """
    def loss_reg(context_points: Mapping[str, Array] | Array, params: Params) -> Array:
        context = context_points["context"] if has_batch_dim else context_points
        f_context = model_fn(input=context, params=params)
        if has_batch_dim and isinstance(context_points, Mapping):
            grid = context_points.get("grid", context)
            prior_cov = prior_cov_kernel(context, grid)
        else:
            prior_cov = prior_cov_kernel(context, context)
        return compute_rkhs_norm(
            f_hat=f_context,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            jitter=jitter,
        )

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    dataset_size: int,
    prior_mean: Array,
    prior_cov_kernel,
    *,
    has_batch_dim: bool = True,
    jitter: float = 1e-4,
    regularizer_scale: float = 0.5,
):
    """Create an FSP objective combining NLL and RKHS regularization.

    Parameters
    ----------
    model_fn : ModelFn
        Model function with Laplax signature ``model_fn(input=..., params=...)``.
    dataset_size : int
        Total number of training samples for minibatch NLL scaling.
    prior_mean : Array
        Prior mean at context points.
    prior_cov_kernel : Callable
        Prior covariance kernel callable.
    has_batch_dim : bool, optional
        Passed to ``create_loss_reg``.
    jitter : float, optional
        Jitter passed to ``create_loss_reg`` and RKHS norm computation.
    regularizer_scale : float, optional
        Multiplicative factor for the regularizer term.

    Returns:
    Callable
        Objective callable with signature
        ``objective(data, context_points, params, scale) -> scalar``.
    """
    loss_nll = create_loss_nll(model_fn=model_fn, dataset_size=dataset_size)
    loss_reg = create_loss_reg(
        model_fn=model_fn,
        prior_mean=prior_mean,
        prior_cov_kernel=prior_cov_kernel,
        has_batch_dim=has_batch_dim,
        jitter=jitter,
    )

    def objective(
        data: Mapping[str, Array],
        context_points,
        params: Params,
        scale: Float | Array | None = None,
    ) -> Array:
        nll_term = loss_nll(data=data, params=params, scale=scale)
        reg_term = loss_reg(context_points=context_points, params=params)
        return nll_term + regularizer_scale * reg_term

    return objective


@partial(jax.jit, static_argnames=("normalize",))
def compute_rkhs_energy_from_chol(
    f_hat: Array,
    prior_mean: Array,
    prior_cov_chol: Array,
    *,
    normalize: Literal["sum", "mean"] = "mean",
) -> Array:
    """Compute RKHS 'energy' using a *precomputed* Cholesky factor.

    This matches fsp_switchable.py:
      alpha = L^{-1} (f - m)
      energy = mean(alpha^2)   (or sum(alpha^2))

    Parameters
    ----------
    f_hat : Array
        Predictions at context points, shape (M, ...).
    prior_mean : Array
        Prior mean at context points, same shape as f_hat.
    prior_cov_chol : Array
        Lower-triangular Cholesky L of (K + jitter I), shape (M, M).
    normalize : {"sum","mean"}
        If "mean", divides by alpha.size (M * Q). For scalar output this is 1/M.

    Returns:
    -------
    Array
        Scalar energy.
    """
    diff = f_hat - prior_mean
    diff_flat = diff.reshape(diff.shape[0], -1)  # (M, Q)
    alpha = jsp.linalg.solve_triangular(prior_cov_chol, diff_flat, lower=True)  # (M, Q)
    sq = jnp.sum(alpha**2)
    if normalize == "mean":
        sq = sq / alpha.size
    return sq


def create_loss_reg_from_chol(
    model_fn: ModelFn,
    prior_mean: Array,
    prior_cov_chol: Array,
    *,
    has_batch_dim: bool = True,
    normalize: Literal["sum", "mean"] = "mean",
):
    """RKHS regularizer factory using a precomputed Cholesky.

    Args:
        model_fn: Model function.
        prior_mean: Prior mean at context points.
        prior_cov_chol: Prior covariance Cholesky factor.
        has_batch_dim: Whether the context points have a batch dimension.
        normalize: Whether to normalize the energy.

    Returns:
        Loss function.

    """
    def loss_reg(context_points: Mapping[str, Array] | Array, params: Params) -> Array:
        context = context_points["context"] if has_batch_dim else context_points
        f_context = model_fn(input=context, params=params)
        return compute_rkhs_energy_from_chol(
            f_hat=f_context,
            prior_mean=prior_mean,
            prior_cov_chol=prior_cov_chol,
            normalize=normalize,
        )

    return loss_reg


def create_fsp_objective_from_chol(
    model_fn: ModelFn,
    dataset_size: int,
    prior_mean: Array,
    prior_cov_chol: Array,
    *,
    has_batch_dim: bool = True,
    normalize: Literal["sum", "mean"] = "mean",
    regularizer_scale: float = 0.5,
):
    """FSP objective that matches fsp_switchable.py when normalize="mean".

    objective = NLL + regularizer_scale * energy
    where energy = mean((L^{-1}(f-m))^2)

    Args:
        model_fn: Model function.
        dataset_size: Total number of training samples.
        prior_mean: Prior mean at context points.
        prior_cov_chol: Prior covariance Cholesky factor.
        has_batch_dim: Whether the context points have a batch dimension.
        normalize: Whether to normalize the energy.
        regularizer_scale: Regularizer scale.

    Returns:
        Objective function.

    """
    loss_nll = create_loss_nll(model_fn=model_fn, dataset_size=dataset_size)
    loss_reg = create_loss_reg_from_chol(
        model_fn=model_fn,
        prior_mean=prior_mean,
        prior_cov_chol=prior_cov_chol,
        has_batch_dim=has_batch_dim,
        normalize=normalize,
    )

    def objective(data, context_points, params, scale=None) -> Array:
        nll_term = loss_nll(data=data, params=params, scale=scale)
        reg_term = loss_reg(context_points=context_points, params=params)
        return nll_term + regularizer_scale * reg_term

    return objective


@partial(jax.jit, static_argnames=("model_fn", "prior_fn", "n_samples"))
def n_gaussian_log_posterior_objective(
    params: Params,
    model_fn: ModelFn,
    x_batch: Array,
    y_batch: Array,
    x_context: Array,
    prior_fn: KernelFn,
    n_samples: int,
    ll_scale: float | Array = 1.0,
) -> tuple[Array, dict]:
    """Negative log-posterior objective with Gaussian likelihood.

    Args:
        params: Model parameters.
        model_fn: Function (input, params) -> output.
        x_batch: Feature batch.
        y_batch: Label batch.
        x_context: Context features for RKHS norm.
        prior_fn: Function (x) -> (mean, cov).
        n_samples: Total number of training samples.
        ll_scale: Likelihood scale/std dev.

    Returns:
        (neg_log_posterior, metrics_dict)
    """
    # 1. Likelihood term
    f_hat = model_fn(input=x_batch, params=params)
    log_likelihood = compute_gaussian_log_likelihood(
        f_hat, y_batch, ll_scale, n_samples
    )

    # 2. Prior term (RKHS norm)
    # Usually we don't need rng for deterministic models, but keeping for compatibility
    f_hat_context = model_fn(input=x_context, params=params)
    prior_mean, prior_cov = prior_fn(x_context)

    sq_rkhs_norm = compute_rkhs_norm(f_hat_context, prior_mean, prior_cov)

    # Log-posterior = LL - 0.5 * ||f||^2
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm

    return -log_posterior, {
        "log_likelihood": log_likelihood,
        "log_posterior": log_posterior,
        "sq_rkhs_norm": sq_rkhs_norm,
    }


def add_ll_rho(
    params: Params,
    init_ll_rho: Float | Array = 0.0,
    *,
    key: str = _LL_RHO_KEY,
) -> Params:
    """Add a learnable likelihood scale parameter (`ll_rho`) to a parameter pytree.

    This function creates a new parameter structure that contains a scalar `ll_rho`
    leaf which can be optimized jointly with the model parameters.

    The convention is:

    1) If `params` is a dict-like mapping, we insert `params[key] = ll_rho` and
       return a shallow copy.

    2) If `params` is *not* a mapping, we wrap it as:

       `{"base": params, key: ll_rho}`

    Parameters
    ----------
    params :
        Model parameters. Typically a dict/pytree that `model_fn` understands.
    init_ll_rho :
        Initial value of the likelihood *pre-activation* parameter.
        The likelihood scale is computed as `sigma = softplus(ll_rho)`.
        This value will be converted to a scalar `jax.Array` with dtype float64.
    key :
        Dictionary key under which to store `ll_rho`. Default is `"ll_rho"`.

    Returns:
    -------
    Params
        A new parameter pytree containing the added scalar leaf `ll_rho`.

    Notes:
    -----
    - `ll_rho` is stored as a scalar `jax.Array`. Use `softplus(ll_rho)` to obtain
      a strictly positive likelihood scale (standard deviation).
    - For best results, keep `params` as a mapping (dict-like). Then `base_params`
      are simply `params` without the `ll_rho` entry.

    Examples:
    --------
    >>> import jax.numpy as jnp
    >>> params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    >>> params2 = add_ll_rho(params, init_ll_rho=0.0)
    >>> "ll_rho" in params2
    True
    """
    ll_rho = jnp.asarray(init_ll_rho, dtype=jnp.float64)

    if isinstance(params, Mapping):
        out = dict(params)
        out[key] = ll_rho
        return out  # type: ignore[return-value]

    return {_BASE_KEY: params, key: ll_rho}  # type: ignore[return-value]


def split_params_ll_rho(
    params: Params, *, key: str = _LL_RHO_KEY
) -> tuple[Params, Array]:
    """Split a parameter pytree into `(base_params, ll_rho)`.

    This is the inverse companion of :func:`add_ll_rho`. It supports both
    conventions produced by `add_ll_rho`:

    - Mapping params: `{"w": ..., "b": ..., "ll_rho": ...}`
      returns `base_params = {"w": ..., "b": ...}`

    - Wrapped params: `{"base": <Params>, "ll_rho": ...}`
      returns `base_params = <Params>`

    Args:
        params: Parameter pytree that contains an `ll_rho` entry.
            Must be a mapping (dict-like) produced by `add_ll_rho`.
        key: Key under which `ll_rho` is stored. Default is `"ll_rho"`.

    Returns:
        base_params: Parameters for the underlying model function (the part passed
            to `base_model_fn`). This is either:
            - the original mapping without the `ll_rho` key, or
            - the object stored in `params["base"]`.
        ll_rho: Scalar pre-activation parameter for the likelihood scale.

    Raises:
        TypeError: If `params` is not a mapping.
        KeyError: If `key` is not present in `params`.

    Examples:
        >>> import jax.numpy as jnp
        >>> p = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,)), "ll_rho": jnp.array(0.0)}
        >>> base, ll_rho = split_params_ll_rho(p)
        >>> sorted(base.keys())
        ['b', 'w']
    """
    if not isinstance(params, Mapping):
        msg = (
            "Expected `params` to be a Mapping (dict-like). "
            "Call `add_ll_rho(params, ...)` first."
        )
        raise TypeError(msg)

    if key not in params:
        msg = f"params missing '{key}'. Call add_ll_rho(params, ...) first."
        raise KeyError(msg)

    ll_rho = params[key]
    base = params.get(_BASE_KEY, None)

    if base is None:
        base_params = dict(params)
        base_params.pop(key)
        return base_params, ll_rho

    return base, ll_rho


@dataclass(frozen=True)
class FSPModelFn:
    """Wrapper around a Laplax-style `model_fn` that adds a learnable likelihood scale.

    This wrapper assumes you represent parameters as a dict-like mapping containing
    a scalar `ll_rho` entry, created via :func:`add_ll_rho`.

    The wrapped object behaves like a `model_fn`:

    - `fsp_model(input=x, params=params)` forwards to `base_model_fn` using
      `base_params` (i.e. params without `ll_rho`).

    Additionally it provides:

    - `fsp_model.sigma(params)` which returns `softplus(ll_rho)` (optionally clipped).

    Parameters
    ----------
    base_model_fn :
        A Laplax-style model function with signature
        `base_model_fn(input=<Array>, params=<Params>) -> Array`.
    ll_rho_key :
        Key to read `ll_rho` from `params`. Default is `"ll_rho"`.
    sigma_min :
        Lower bound for the returned likelihood scale (standard deviation).
        This can avoid numerical issues for very small sigma values.

    Notes:
    -----
    The typical Gaussian likelihood uses:
    `sigma = softplus(ll_rho)`.

    Examples:
    --------
    >>> # Suppose model_fn(input=..., params=...) exists
    >>> fsp_model = FSPModelFn(base_model_fn=model_fn)
    >>> params = add_ll_rho(base_params, init_ll_rho=0.0)
    >>> y = fsp_model(input=x, params=params)
    >>> sigma = fsp_model.sigma(params)
    """

    base_model_fn: ModelFn
    ll_rho_key: str = _LL_RHO_KEY
    sigma_min: float = 1e-6

    def __call__(self, *, input: Array, params: Params) -> Array:
        """Forward pass through the underlying model function.

        Parameters
        ----------
        input :
            Input array passed to the underlying model.
        params :
            Parameter pytree containing `ll_rho` (and the underlying model parameters).

        Returns:
        -------
        Array
            Model output from `base_model_fn`.
        """
        base_params, _ = split_params_ll_rho(params, key=self.ll_rho_key)
        return self.base_model_fn(input=input, params=base_params)

    def sigma(self, params: Params) -> Array:
        """Compute the positive likelihood scale parameter.

        Parameters
        ----------
        params :
            Parameter pytree containing `ll_rho`.

        Returns:
        -------
        Array
            A scalar array `sigma = clip(softplus(ll_rho), a_min=sigma_min)`.

        Notes:
        -----
        This returns a *standard deviation* (not variance).
        """
        _, ll_rho = split_params_ll_rho(params, key=self.ll_rho_key)
        sigma = jax.nn.softplus(ll_rho)
        return jnp.clip(sigma, a_min=self.sigma_min)


def fsp_wrapper(
    model_fn: ModelFn,
    *,
    ll_rho_key: str = _LL_RHO_KEY,
    sigma_min: Float = 1e-6,
) -> FSPModelFn:
    """Wrap a Laplax-style `model_fn` with an FSP-compatible likelihood scale parameter.

    The returned object behaves like a `model_fn(input=..., params=...)`, but expects
    `params` to contain an additional scalar entry `ll_rho` (by default under key
    `"ll_rho"`), and provides `sigma(params)`.

    Parameters
    ----------
    model_fn :
        Base model function with signature `model_fn(input, params) -> output`.
    ll_rho_key :
        Name of the key under which `ll_rho` is stored in `params`.
    sigma_min :
        Lower bound applied to `sigma(params)` for stability.

    Returns:
    -------
    FSPModelFn
        A wrapper object with:
        - `__call__(input, params)` forwarding to `model_fn` using base params
        - `sigma(params)` returning `softplus(ll_rho)` (clipped)

    Examples:
    --------
    >>> fsp_model = fsp_wrapper(model_fn)
    >>> params = add_ll_rho(base_params, init_ll_rho=0.0)
    >>> sigma = fsp_model.sigma(params)
    >>> loss, metrics = n_gaussian_log_posterior_objective(
    ...     params=params,
    ...     model_fn=fsp_model,
    ...     x_batch=x,
    ...     y_batch=y,
    ...     x_context=xc,
    ...     prior_fn=prior_fn,
    ...     n_samples=n_samples,
    ...     ll_scale=sigma,
    ... )
    """
    return FSPModelFn(
        base_model_fn=model_fn, ll_rho_key=ll_rho_key, sigma_min=sigma_min
    )
