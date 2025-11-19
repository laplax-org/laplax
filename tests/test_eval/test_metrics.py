import jax
import jax.numpy as jnp
import numpy as np

from laplax.curv.utils import LowRankTerms
from laplax.eval.metrics import (
    chi_squared,
    cov_low_rank_approximation,
    crps_gaussian,
    estimate_rmse,
    low_rank_log_determinant,
    low_rank_mahalanobis_distance_inverse_covariance,
    low_rank_marginal_nlpd_per_input,
    low_rank_nlpd_per_input,
    nll_gaussian,
)


def _make_psd_from_factors(U: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    return U @ (jnp.square(S) * U.T)


def _orthonormal(n: int, k: int, key: jax.Array) -> jnp.ndarray:
    A = jax.random.normal(key, (n, k))
    Q, _ = jnp.linalg.qr(A)
    return Q[:, :k]


def test_cov_low_rank_approximation_fsp_passthrough():
    # Simulate an FSP result: we already have low_rank_terms, no pred_cov.
    key = jax.random.PRNGKey(0)
    n, k = 20, 5
    U = _orthonormal(n, k, key)
    S = jnp.linspace(1.0, 2.0, k)
    lr = LowRankTerms(U=U, S=S, scalar=jnp.asarray(0.0, S.dtype))

    results_in = {"low_rank_terms": lr}
    results_out, _ = cov_low_rank_approximation(results_in.copy(), aux={})

    # Should leave existing low_rank_terms untouched and not add pred_cov_low_rank_terms
    np.testing.assert_allclose(results_out["low_rank_terms"].U, U, atol=1e-6)
    np.testing.assert_allclose(results_out["low_rank_terms"].S, S, atol=1e-6)
    assert "pred_cov_low_rank_terms" not in results_out


def test_cov_low_rank_approximation_from_dense_rank_exact():
    # Construct an exactly rank-r PSD matrix and verify recovery with cov_rank=r
    key = jax.random.PRNGKey(1)
    n, r = 30, 6
    U_true = _orthonormal(n, r, key)
    evals = jnp.linspace(2.0, 4.0, r)  # covariance eigenvalues
    # In our convention S are scale singular values, so S^2 = evals
    S_true = jnp.sqrt(evals)
    Sigma = _make_psd_from_factors(U_true, S_true)

    results, _ = cov_low_rank_approximation({"pred_cov": Sigma}, aux={}, cov_rank=r)
    lr = results["pred_cov_low_rank_terms"]

    # Reconstruct covariance
    Sigma_hat = _make_psd_from_factors(lr.U, lr.S)
    err = jnp.linalg.norm(Sigma - Sigma_hat, ord="fro")
    rel_err = err / jnp.linalg.norm(Sigma, ord="fro")
    assert rel_err < 1e-3


def test_cov_low_rank_approximation_error_decreases_with_rank():
    key = jax.random.PRNGKey(2)
    n, r = 40, 8
    U_true = _orthonormal(n, r, key)
    S_true = jnp.linspace(0.5, 2.0, r)
    Sigma = _make_psd_from_factors(U_true, S_true)

    res_small, _ = cov_low_rank_approximation({"pred_cov": Sigma}, aux={}, cov_rank=3)
    lr_small = res_small["pred_cov_low_rank_terms"]
    Sigma_small = _make_psd_from_factors(lr_small.U, lr_small.S)
    err_small = float(jnp.linalg.norm(Sigma - Sigma_small, ord="fro"))

    res_large, _ = cov_low_rank_approximation({"pred_cov": Sigma}, aux={}, cov_rank=6)
    lr_large = res_large["pred_cov_low_rank_terms"]
    Sigma_large = _make_psd_from_factors(lr_large.U, lr_large.S)
    err_large = float(jnp.linalg.norm(Sigma - Sigma_large, ord="fro"))

    assert err_large <= err_small + 1e-6


def test_low_rank_metrics_match_dense():
    key = jax.random.PRNGKey(3)
    n, k = 25, 5
    keyU, keyv = jax.random.split(key)
    U = _orthonormal(n, k, keyU)
    S = jnp.linspace(0.8, 1.3, k)
    sigma = 0.3

    # Build dense covariance
    Sigma = _make_psd_from_factors(U, S) + (sigma**2) * jnp.eye(n)

    # Random mean/target
    v = jax.random.normal(keyv, (n,))
    pred_mean = v
    target = jnp.zeros_like(v)

    # Exact quadratic form and logdet
    x = jnp.linalg.solve(Sigma, v)
    quad_exact = float(jnp.dot(v, x))
    sign, logdet_exact = jnp.linalg.slogdet(Sigma)
    assert sign > 0

    # Low-rank metrics
    lr = LowRankTerms(U=U, S=S, scalar=jnp.asarray(0.0, S.dtype))
    md_lr = low_rank_mahalanobis_distance_inverse_covariance(
        pred_mean=pred_mean,
        target=target,
        pred_cov_low_rank_terms=lr,
        observation_noise=jnp.log(sigma),
    )
    logdet_lr = low_rank_log_determinant(lr, jnp.log(sigma))

    np.testing.assert_allclose(md_lr**2, quad_exact, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(logdet_lr, float(logdet_exact), atol=1e-5, rtol=1e-5)

    # NLPD comparison
    out = low_rank_nlpd_per_input(
        pred_mean=pred_mean,
        target=target,
        pred_cov_low_rank_terms=lr,
        observation_noise=jnp.log(sigma),
    )
    nlpd_lr = out["nlpd_per_input"]
    nlpd_exact = 0.5 * (logdet_exact + quad_exact + jnp.log(2 * jnp.pi))
    np.testing.assert_allclose(nlpd_lr, float(nlpd_exact), atol=1e-5, rtol=1e-5)


def test_low_rank_marginal_nlpd_uses_pred_var():
    key = jax.random.PRNGKey(4)
    n, k = 20, 4
    U = _orthonormal(n, k, key)
    S = jnp.linspace(0.5, 1.0, k)
    sigma = 0.2
    pred_var = jnp.sum((U * S) ** 2, axis=1) + sigma**2
    pred_mean = jax.random.normal(key, (n,))
    target = jax.random.normal(key, (n,))

    out = low_rank_marginal_nlpd_per_input(
        pred_mean=pred_mean,
        pred_var=pred_var,
        target=target,
        observation_noise=jnp.log(sigma),
    )
    # Verify keys and finite values
    assert "marginal_nlpd_per_input" in out
    assert "mse_per_input" in out
    assert jnp.isfinite(out["marginal_nlpd_per_input"]).item()


def test_default_metrics_simple():
    # RMSE
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 1.0, 4.0])
    rmse = estimate_rmse(pred_mean=y_pred, target=y_true)
    np.testing.assert_allclose(rmse, jnp.sqrt(jnp.mean((y_pred - y_true) ** 2)))

    # chi^2
    std = jnp.array([1.0, 2.0, 0.5])
    chi = chi_squared(pred_mean=y_pred, pred_std=std, target=y_true)
    np.testing.assert_allclose(chi, jnp.mean(((y_pred - y_true) ** 2) / (std**2)))

    # NLL Gaussian
    nll = nll_gaussian(pred_mean=y_pred, pred_std=std, target=y_true, scaled=False)
    resid = y_pred - y_true
    expected_nll = -jnp.sum(jax.scipy.stats.norm.logpdf(resid, scale=std))
    np.testing.assert_allclose(nll, expected_nll)

    # CRPS Gaussian - compare to formula used internally at zero residuals.
    zeros = jnp.zeros_like(y_true)
    std1 = jnp.ones_like(y_true) * 0.7
    crps = crps_gaussian(pred_mean=zeros, pred_std=std1, target=zeros, scaled=False)
    term_1 = 1 / jnp.sqrt(jnp.pi)
    term_2 = 2 * jax.scipy.stats.norm.pdf(0.0)
    term_3 = 0.0
    crps_expected = -jnp.sum(std1 * (term_1 - term_2 - term_3))
    np.testing.assert_allclose(crps, crps_expected)
