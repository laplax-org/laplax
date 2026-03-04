"""Tests for FSP objective builders."""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp_linalg
import numpy as np
import optax
import pytest
from flax import linen as nn

from laplax.types import Array, Params
from laplax.util.objective import (
    add_ll_rho,
    compute_gaussian_log_likelihood,
    compute_rkhs_energy_from_chol,
    compute_rkhs_norm,
    create_fsp_objective,
    create_fsp_objective_from_chol,
    create_loss_nll,
    create_loss_reg,
    fsp_wrapper,
    n_gaussian_log_posterior_objective,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers for periodic objective test
# ---------------------------------------------------------------------------


def kernel_periodic_1d(x, y, variance=1.0, lengthscale=1.0, period=1.0):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    d = jnp.abs(x[:, None, 0] - y[None, :, 0])
    s = jnp.sin(jnp.pi * d / period)
    return variance * jnp.exp(-(2.0 * s**2) / (lengthscale**2))


def prior_fn_periodic(x):
    # Mean zero, Periodic covariance
    mean = jnp.zeros((x.shape[0], 1))
    cov = kernel_periodic_1d(
        x, x, period=2.0
    )  # Period 2.0 matches [-1, 1] range length
    return mean, cov


class SmallMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def test_fsp_objective_periodicity():
    """Test that training with FSP objective enforces periodicity on extrapolation."""
    key = jr.PRNGKey(42)
    key_init, key_data, key_train = jr.split(key, 3)

    # Data: sin(pi * x) on [-1, 1]. Period is 2.
    x_train = jr.uniform(key_data, (50, 1), minval=-1.0, maxval=1.0)
    y_train = jnp.sin(jnp.pi * x_train)

    model = SmallMLP()
    params = model.init(key_init, jnp.zeros((1, 1)))

    # Context points for regularization
    x_context = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, opt_st):
        loss, _aux = n_gaussian_log_posterior_objective(
            params=p,
            model_fn=lambda input, params: model.apply(params, input),
            x_batch=x_train,
            y_batch=y_train,
            x_context=x_context,
            prior_fn=prior_fn_periodic,
            n_samples=50,
            ll_scale=0.1,
        )
        grads = jax.grad(
            lambda p_: n_gaussian_log_posterior_objective(
                params=p_,
                model_fn=lambda input, params: model.apply(params, input),
                x_batch=x_train,
                y_batch=y_train,
                x_context=x_context,
                prior_fn=prior_fn_periodic,
                n_samples=50,
                ll_scale=0.1,
            )[0]
        )(p)

        updates, opt_st = optimizer.update(grads, opt_st)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    # Train
    for _ in range(500):
        key_train, _k = jr.split(key_train)
        params, opt_state, _loss = step(params, opt_state)

    # Check periodicity
    # Check if f(1.5) approx f(-0.5) (since period is 2)
    x_test = jnp.array([[1.5]])
    x_target = jnp.array([[-0.5]])

    pred_test = model.apply(params, x_test)
    pred_target = model.apply(params, x_target)

    diff = jnp.abs(pred_test - pred_target)
    # print(f"Periodicity diff: {diff}")

    assert diff < 0.5, "FSP Regularizer failed to enforce approximate periodicity"


def linear_model_fn(*, input: Array, params: Params) -> Array:
    """Simple linear regression model: f(x) = x @ w + b.

    Args:
        input: Input points.
        params: Model parameters.

    Returns:
        Model output.
    """
    w = params["w"]
    b = params["b"]
    return input @ w + b


def rbf_prior_fn(x_context: Array, *, jitter: float = 1e-4):
    """Zero-mean RBF GP prior for 1D inputs (sufficient for tests).

    Args:
        x_context: Input context points.
        jitter: Jitter to add to the covariance matrix.

    Returns:
        Mean and covariance of the prior.
    """
    x = x_context.reshape(-1, x_context.shape[-1])
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    d2 = x2 - 2.0 * (x @ x.T) + x2.T
    K = jnp.exp(-0.5 * d2 / (0.5**2))
    K = K + jitter * jnp.eye(K.shape[0])
    m = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
    return m, K


def test_add_ll_rho_inserts_scalar_leaf():
    """`add_ll_rho` should insert a scalar `ll_rho` into a dict-like params structure.

    Args:
        base_params: Base parameters.
        init_ll_rho: Initial likelihood scale parameter.




    """
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    assert "ll_rho" in params
    assert params["ll_rho"].shape == ()


def test_wrapper_forwards_model_fn_and_sigma_positive():
    """FSP wrapper should forward predictions and produce a positive sigma."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)

    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.array([[2.0], [3.0]])
    y_wrapped = fsp_model(input=x, params=params)
    y_base = linear_model_fn(input=x, params=base_params)

    assert jnp.allclose(y_wrapped, y_base)
    assert float(fsp_model.sigma(params)) > 0.0


def test_ll_rho_receives_gradient_and_updates():
    """`ll_rho` should receive gradients and update under an optimizer step."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = jnp.zeros_like(x)

    def loss_fn(p):
        f = fsp_model(input=x, params=p)
        sigma = fsp_model.sigma(p)
        # use your log-likelihood scaling (N/batch_size) via the provided helper
        return -compute_gaussian_log_likelihood(f, y, sigma, n_samples=x.shape[0])

    ll_before = params["ll_rho"]
    grads = jax.grad(loss_fn)(params)
    assert "ll_rho" in grads
    assert jnp.isfinite(grads["ll_rho"])

    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    updates, opt_state = opt.update(grads, opt_state)
    params2 = optax.apply_updates(params, updates)

    assert not jnp.allclose(ll_before, params2["ll_rho"])


def test_objective_runs_with_learned_sigma():
    """The full negative log-posterior objective should run using `sigma(params)`."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    y = jnp.sin(2 * jnp.pi * x)
    xc = jnp.linspace(-2.0, 2.0, 20).reshape(-1, 1)

    loss, metrics = n_gaussian_log_posterior_objective(
        params=params,
        model_fn=fsp_model,
        x_batch=x,
        y_batch=y,
        x_context=xc,
        prior_fn=lambda z: rbf_prior_fn(z, jitter=1e-4),
        n_samples=x.shape[0],
        ll_scale=fsp_model.sigma(params),
    )

    assert jnp.isfinite(loss)
    assert "log_likelihood" in metrics
    assert "sq_rkhs_norm" in metrics
    assert jnp.isfinite(metrics["sq_rkhs_norm"])


def test_compute_rkhs_norm_handles_structured_outputs():
    kernel = jnp.array([
        [1.5, 0.2, 0.1],
        [0.2, 1.2, 0.3],
        [0.1, 0.3, 1.1],
    ])
    f_hat = jnp.arange(12.0).reshape(3, 2, 2)
    prior_mean = jnp.zeros_like(f_hat)
    val_mean = compute_rkhs_norm(
        f_hat, prior_mean, kernel, jitter=0.0, normalize="mean"
    )
    val_sum = compute_rkhs_norm(f_hat, prior_mean, kernel, jitter=0.0, normalize="sum")

    diff = (f_hat - prior_mean).reshape(3, -1)
    inv_kernel = jnp.linalg.inv(kernel)
    expected_sum = jnp.sum(diff * (inv_kernel @ diff))
    expected_mean = expected_sum / diff.size
    assert jnp.allclose(val_sum, expected_sum, atol=1e-6)
    assert jnp.allclose(val_mean, expected_mean, atol=1e-6)


def test_create_loss_nll_matches_direct_formula():
    params = {"w": jnp.array([[2.0]]), "b": jnp.array([0.5])}
    data = {
        "input": jnp.array([[0.0], [1.0], [2.0], [3.0]]),
        "target": jnp.array([[0.4], [2.7], [4.2], [6.6]]),
    }
    scale = jnp.array(0.3)
    loss_nll = create_loss_nll(linear_model_fn, dataset_size=20)
    val = loss_nll(data, params, scale)

    preds = linear_model_fn(input=data["input"], params=params)
    expected = -compute_gaussian_log_likelihood(
        preds, data["target"], scale, n_samples=20
    )
    assert jnp.allclose(val, expected)


def test_create_loss_reg_matches_manual_quadratic():
    params = {"w": jnp.array([[1.0]]), "b": jnp.array([0.0])}
    context = jnp.array([[-1.0], [0.0], [1.0]])
    prior_mean = jnp.zeros((3, 1))

    def kernel_fn(x1, x2):
        del x1, x2
        return jnp.array([
            [1.0, 0.1, 0.0],
            [0.1, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ])

    loss_reg = create_loss_reg(
        model_fn=linear_model_fn,
        prior_mean=prior_mean,
        prior_cov_kernel=kernel_fn,
        has_batch_dim=False,
        jitter=0.0,
    )
    val = loss_reg(context, params)

    f_context = linear_model_fn(input=context, params=params)
    expected = compute_rkhs_norm(
        f_hat=f_context,
        prior_mean=prior_mean,
        prior_cov=kernel_fn(context, context),
        jitter=0.0,
    )
    assert jnp.allclose(val, expected, atol=1e-6)


def test_create_fsp_objective_combines_nll_and_regularizer():
    params = {"w": jnp.array([[1.5]]), "b": jnp.array([0.2])}
    data = {
        "input": jnp.array([[-1.0], [0.0], [1.0], [2.0]]),
        "target": jnp.array([[-1.1], [0.1], [1.2], [3.1]]),
    }
    context_dict = {
        "context": jnp.array([[-1.0], [0.0], [1.0]]),
        "grid": jnp.array([[-1.0], [0.0], [1.0]]),
    }
    prior_mean = jnp.zeros((3, 1))

    def kernel_fn(_x1, _x2):
        return jnp.array([
            [1.0, 0.1, 0.0],
            [0.1, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ])

    objective = create_fsp_objective(
        model_fn=linear_model_fn,
        dataset_size=12,
        prior_mean=prior_mean,
        prior_cov_kernel=kernel_fn,
        has_batch_dim=True,
        jitter=0.0,
        regularizer_scale=0.5,
    )
    val = objective(data, context_dict, params, scale=jnp.array(0.25))

    nll = create_loss_nll(linear_model_fn, dataset_size=12)(
        data, params, jnp.array(0.25)
    )
    reg = create_loss_reg(
        model_fn=linear_model_fn,
        prior_mean=prior_mean,
        prior_cov_kernel=kernel_fn,
        has_batch_dim=True,
        jitter=0.0,
    )(context_dict, params)
    expected = nll + 0.5 * reg
    assert jnp.allclose(val, expected, atol=1e-6)


def _make_spd(key: jax.Array, n: int, eps: float = 1e-2) -> jnp.ndarray:
    a = jr.normal(key, (n, n), dtype=jnp.float64)
    k = a @ a.T
    k = k + eps * jnp.eye(n, dtype=jnp.float64)
    return k


def _manual_energy_from_kernel(
    f_hat: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_cov: jnp.ndarray,
    jitter: float,
    normalize: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (energy, chol) where chol = chol(prior_cov + jitter I)."""
    m = prior_cov.shape[0]
    chol = jnp.linalg.cholesky(prior_cov + jitter * jnp.eye(m, dtype=prior_cov.dtype))
    diff = (f_hat - prior_mean).reshape(m, -1)
    alpha = jsp_linalg.solve_triangular(chol, diff, lower=True)
    sq = jnp.sum(alpha**2)
    if normalize == "mean":
        sq = sq / alpha.size
    return sq, chol


def _rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, *, variance=1.0, lengthscale=1.0):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    sq = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sq / (lengthscale**2))


def _linear_model_fn_fsp(*, input: jnp.ndarray, params: dict) -> jnp.ndarray:
    return input @ params["w"] + params["b"]


def _fixture_fsp(*, key=0, m=23, d=2, q=3, jitter=1e-4):
    k = jr.PRNGKey(key)
    k1, k2, k3, k4 = jr.split(k, 4)

    c = jnp.linspace(-1.0, 1.0, m).reshape(m, 1)
    xb = jr.normal(k1, (11, d))
    yb = jr.normal(k2, (11, q))

    params = {
        "w": jr.normal(k3, (d, q)),
        "b": jr.normal(k4, (q,)),
    }

    prior_mean = jnp.zeros((m, q))
    k_mat = _rbf_kernel(c, c, variance=1.2, lengthscale=0.7)
    chol = jnp.linalg.cholesky(k_mat + jitter * jnp.eye(m))
    c_model = jnp.concatenate([c, c], axis=1)
    return c_model, xb, yb, params, prior_mean, k_mat, chol, jitter


def test_compute_rkhs_norm_sum_vs_mean_factor():
    key = jr.PRNGKey(0)
    m, q = 7, 3
    k1, k2, k3 = jr.split(key, 3)

    prior_cov = _make_spd(k1, m)
    f_hat = jr.normal(k2, (m, q), dtype=jnp.float64)
    prior_mean = jr.normal(k3, (m, q), dtype=jnp.float64)

    jitter = 1e-4
    val_sum = compute_rkhs_norm(
        f_hat, prior_mean, prior_cov, jitter=jitter, normalize="sum"
    )
    val_mean = compute_rkhs_norm(
        f_hat, prior_mean, prior_cov, jitter=jitter, normalize="mean"
    )

    val_sum = float(val_sum)
    val_mean = float(val_mean)

    assert np.isfinite(val_sum)
    assert np.isfinite(val_mean)
    np.testing.assert_allclose(val_sum, val_mean * (m * q), rtol=1e-5, atol=1e-5)


def test_compute_rkhs_norm_matches_manual_energy():
    key = jr.PRNGKey(1)
    m, q = 9, 2
    k1, k2 = jr.split(key, 2)

    prior_cov = _make_spd(k1, m)
    f_hat = jr.normal(k2, (m, q), dtype=jnp.float64)
    prior_mean = jnp.zeros((m, q), dtype=jnp.float64)

    jitter = 5e-4
    expected, _ = _manual_energy_from_kernel(
        f_hat=f_hat,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        jitter=jitter,
        normalize="mean",
    )

    got = compute_rkhs_norm(
        f_hat, prior_mean, prior_cov, jitter=jitter, normalize="mean"
    )
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-5, atol=1e-5)


def test_compute_rkhs_energy_from_chol_matches_manual():
    key = jr.PRNGKey(2)
    m, q = 8, 4
    k1, k2 = jr.split(key, 2)

    prior_cov = _make_spd(k1, m)
    f_hat = jr.normal(k2, (m, q), dtype=jnp.float64)
    prior_mean = jnp.zeros((m, q), dtype=jnp.float64)

    jitter = 1e-3
    expected, chol = _manual_energy_from_kernel(
        f_hat=f_hat,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        jitter=jitter,
        normalize="mean",
    )

    got = compute_rkhs_energy_from_chol(
        f_hat=f_hat,
        prior_mean=prior_mean,
        prior_cov_chol=chol,
        normalize="mean",
    )

    np.testing.assert_allclose(float(got), float(expected), rtol=1e-10, atol=1e-10)


def test_create_loss_reg_dense_is_mean_normalized():
    def model_fn(*, input, params):
        return input * params["w"] + params["b"]

    key = jr.PRNGKey(3)
    m = 11
    c = jnp.linspace(-1.0, 1.0, m, dtype=jnp.float64).reshape(-1, 1)

    params = {
        "w": jnp.array(0.7, dtype=jnp.float64),
        "b": jnp.array(-0.1, dtype=jnp.float64),
    }
    prior_mean = jnp.zeros((m, 1), dtype=jnp.float64)

    k_mat = _make_spd(key, m)
    jitter = 1e-4

    def prior_cov_kernel(x1, x2):
        assert x1.shape[0] == m
        assert x2.shape[0] == m
        return k_mat

    loss_reg = create_loss_reg(
        model_fn=model_fn,
        prior_mean=prior_mean,
        prior_cov_kernel=prior_cov_kernel,
        has_batch_dim=True,
        jitter=jitter,
    )

    f_c = model_fn(input=c, params=params)
    expected, _ = _manual_energy_from_kernel(
        f_hat=f_c,
        prior_mean=prior_mean,
        prior_cov=k_mat,
        jitter=jitter,
        normalize="mean",
    )

    got = loss_reg({"context": c}, params)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-5, atol=1e-5)


def test_create_fsp_objective_from_chol_matches_switchable_scaling():
    def model_fn(*, input, params):
        return input * params["w"] + params["b"]

    x = jnp.array([[-0.5], [0.5]], dtype=jnp.float64)
    y = jnp.array([[1.0], [-1.0]], dtype=jnp.float64)
    data = {"input": x, "target": y}
    dataset_size = x.shape[0]
    sigma = jnp.array(0.3, dtype=jnp.float64)

    c = jnp.array([[-0.5], [0.5], [1.5]], dtype=jnp.float64)
    m = c.shape[0]
    context = {"context": c}

    params = {
        "w": jnp.array(0.2, dtype=jnp.float64),
        "b": jnp.array(0.0, dtype=jnp.float64),
    }
    prior_mean = jnp.zeros((m, 1), dtype=jnp.float64)

    key = jr.PRNGKey(4)
    k_mat = _make_spd(key, m)
    jitter = 1e-4
    chol = jnp.linalg.cholesky(k_mat + jitter * jnp.eye(m, dtype=jnp.float64))

    obj = create_fsp_objective_from_chol(
        model_fn=model_fn,
        dataset_size=dataset_size,
        prior_mean=prior_mean,
        prior_cov_chol=chol,
        has_batch_dim=True,
        normalize="mean",
        regularizer_scale=0.5,
    )

    preds = model_fn(input=x, params=params)
    nll = -compute_gaussian_log_likelihood(preds, y, sigma, n_samples=dataset_size)

    f_c = model_fn(input=c, params=params)
    energy = compute_rkhs_energy_from_chol(f_c, prior_mean, chol, normalize="mean")

    expected = nll + 0.5 * energy
    got = obj(data, context, params, sigma)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-5, atol=1e-5)


def test_periodic_kernel_induces_periodic_bias_in_tiny_linear_layer():
    def model_fn(*, input, params):
        return input * params["w"] + params["b"]

    def kernel_periodic_1d_local(x, y, *, variance=1.0, lengthscale=0.7, period=1.0):
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        d = jnp.abs(x[:, None, 0] - y[None, :, 0])
        s = jnp.sin(jnp.pi * d / period)
        return variance * jnp.exp(-(2.0 * s**2) / (lengthscale**2))

    def kernel_matern52_1d(x, y, *, variance=1.0, lengthscale=0.15):
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        r = jnp.abs(x[:, None, 0] - y[None, :, 0])
        sqrt5 = jnp.sqrt(5.0)
        val = sqrt5 * r / lengthscale
        return variance * (1.0 + val + (val**2) / 3.0) * jnp.exp(-val)

    x0 = jnp.array([[0.25]], dtype=jnp.float64)
    x1 = jnp.array([[1.25]], dtype=jnp.float64)
    x_train = jnp.concatenate([x0, x1], axis=0)
    y_train = jnp.array([[1.0], [-1.0]], dtype=jnp.float64)
    data = {"input": x_train, "target": y_train}
    dataset_size = x_train.shape[0]

    c = x_train
    context = {"context": c}
    prior_mean = jnp.zeros((c.shape[0], 1), dtype=jnp.float64)

    jitter = 1e-2
    k_p = kernel_periodic_1d_local(c, c)
    k_m = kernel_matern52_1d(c, c)

    l_p = jnp.linalg.cholesky(k_p + jitter * jnp.eye(c.shape[0], dtype=jnp.float64))
    l_m = jnp.linalg.cholesky(k_m + jitter * jnp.eye(c.shape[0], dtype=jnp.float64))

    sigma = jnp.array(0.25, dtype=jnp.float64)
    obj_p = create_fsp_objective_from_chol(
        model_fn=model_fn,
        dataset_size=dataset_size,
        prior_mean=prior_mean,
        prior_cov_chol=l_p,
        has_batch_dim=True,
        normalize="mean",
        regularizer_scale=0.5,
    )
    obj_m = create_fsp_objective_from_chol(
        model_fn=model_fn,
        dataset_size=dataset_size,
        prior_mean=prior_mean,
        prior_cov_chol=l_m,
        has_batch_dim=True,
        normalize="mean",
        regularizer_scale=0.5,
    )

    params0 = {
        "w": jnp.array(0.0, dtype=jnp.float64),
        "b": jnp.array(0.0, dtype=jnp.float64),
    }

    tx = optax.adam(0.15)

    def _train(obj, params_init, steps: int = 400):
        opt_state = tx.init(params_init)

        @jax.jit
        def step(params, opt_state):
            def loss_fn(p):
                return obj(data, context, p, sigma)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state2 = tx.update(grads, opt_state, params)
            params2 = optax.apply_updates(params, updates)
            return params2, opt_state2, loss

        params = params_init
        for _ in range(steps):
            params, opt_state, _ = step(params, opt_state)
        return params

    params_p = _train(obj_p, params0)
    params_m = _train(obj_m, params0)

    pred_p = model_fn(input=x_train, params=params_p).reshape(-1)
    pred_m = model_fn(input=x_train, params=params_m).reshape(-1)

    diff_p = float(jnp.abs(pred_p[0] - pred_p[1]))
    diff_m = float(jnp.abs(pred_m[0] - pred_m[1]))
    assert diff_p < 0.75 * diff_m
    assert diff_m > 1.0


def test_rkhs_energy_default_is_mean():
    c_model, _, _, params, prior_mean, _, chol, _ = _fixture_fsp()
    f_c = _linear_model_fn_fsp(input=c_model, params=params)

    got_default = compute_rkhs_energy_from_chol(
        f_hat=f_c, prior_mean=prior_mean, prior_cov_chol=chol
    )
    got_mean = compute_rkhs_energy_from_chol(
        f_hat=f_c, prior_mean=prior_mean, prior_cov_chol=chol, normalize="mean"
    )
    got_sum = compute_rkhs_energy_from_chol(
        f_hat=f_c, prior_mean=prior_mean, prior_cov_chol=chol, normalize="sum"
    )

    assert jnp.allclose(got_default, got_mean, rtol=0, atol=0)
    diff = (f_c - prior_mean).reshape(f_c.shape[0], -1)
    alpha = jsp_linalg.solve_triangular(chol, diff, lower=True)
    assert jnp.abs(got_sum - (got_mean * alpha.size)) < 1e-5


def test_compute_rkhs_norm_mean_sum_relation():
    c_model, _, _, params, prior_mean, k_mat, chol, jitter = _fixture_fsp()
    f_c = _linear_model_fn_fsp(input=c_model, params=params)

    dense_mean = compute_rkhs_norm(
        f_hat=f_c,
        prior_mean=prior_mean,
        prior_cov=k_mat,
        jitter=jitter,
        normalize="mean",
    )
    dense_sum = compute_rkhs_norm(
        f_hat=f_c,
        prior_mean=prior_mean,
        prior_cov=k_mat,
        jitter=jitter,
        normalize="sum",
    )

    diff = (f_c - prior_mean).reshape(f_c.shape[0], -1)
    alpha = jsp_linalg.solve_triangular(chol, diff, lower=True)
    assert jnp.allclose(dense_sum, dense_mean * alpha.size, rtol=1e-6, atol=1e-6)


def test_fsp_objective_from_chol_matches_switchable_convention():
    c_model, xb, yb, params, prior_mean, _, chol, _ = _fixture_fsp()
    sigma = jnp.array(0.35)
    n = 123

    data = {"input": xb, "target": yb}
    context = {"context": c_model}

    obj = create_fsp_objective_from_chol(
        model_fn=_linear_model_fn_fsp,
        dataset_size=n,
        prior_mean=prior_mean,
        prior_cov_chol=chol,
        has_batch_dim=True,
        normalize="mean",
        regularizer_scale=0.5,
    )

    got = obj(data, context, params, sigma)

    preds = _linear_model_fn_fsp(input=xb, params=params)
    ll = compute_gaussian_log_likelihood(preds, yb, sigma, n)
    nll = -ll

    f_c = _linear_model_fn_fsp(input=c_model, params=params)
    diff = (f_c - prior_mean).reshape(f_c.shape[0], -1)
    alpha = jsp_linalg.solve_triangular(chol, diff, lower=True)
    energy_mean = jnp.mean(alpha**2)

    expected = nll + 0.5 * energy_mean
    assert jnp.allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_dense_objective_matches_chol_objective():
    c_model, xb, yb, params, prior_mean, _, chol, jitter = _fixture_fsp()
    sigma = jnp.array(0.35)
    n = 123

    data = {"input": xb, "target": yb}

    def kernel_fn(x, y):
        x1 = x[:, :1]
        y1 = y[:, :1]
        return _rbf_kernel(x1, y1, variance=1.2, lengthscale=0.7)

    obj_dense = create_fsp_objective(
        model_fn=_linear_model_fn_fsp,
        dataset_size=n,
        prior_mean=prior_mean,
        prior_cov_kernel=kernel_fn,
        has_batch_dim=True,
        jitter=jitter,
        regularizer_scale=0.5,
    )
    obj_chol = create_fsp_objective_from_chol(
        model_fn=_linear_model_fn_fsp,
        dataset_size=n,
        prior_mean=prior_mean,
        prior_cov_chol=chol,
        has_batch_dim=True,
        normalize="mean",
        regularizer_scale=0.5,
    )

    got_dense = obj_dense(data, {"context": c_model}, params, sigma)
    got_chol = obj_chol(data, {"context": c_model}, params, sigma)
    assert jnp.allclose(got_dense, got_chol, rtol=1e-5, atol=1e-5)
