"""Tests for laplax.curv.fsp — Function Space Prior Laplace Approximation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flax import linen as nn

# Optional GPJax import
try:
    import gpjax as gpx

    GPJAX_AVAILABLE = True
except ImportError:
    GPJAX_AVAILABLE = False

from laplax.curv.fsp import (
    _compute_full_prior_inverse_sqrt,
    _compute_M_memory_efficient,
    _create_concatenated_model_jvp,
    _fsp_state_to_cov,
    _fsp_state_to_scale,
    _lanczos_init_full,
    _model_vjp,
    create_fsp_curvature,
    create_fsp_posterior,
)
from laplax.curv.ggn import _create_loss_fn
from laplax.enums import LossFn, LowRankMethod


@pytest.fixture(autouse=True)
def _enable_x64():
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Small network so tests run fast."""

    hidden: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class TinyMLPMultiOutput(nn.Module):
    """Multi-output variant for testing output_dim > 1."""

    hidden: int = 16
    out_dim: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class SmallMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def _model_fn(input, params, apply_fn):
    return apply_fn(params, input)


def _rbf_kernel(x, y, lengthscale=1.0, variance=1.0):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


def kernel_periodic_1d(x, y, variance=1.0, lengthscale=1.0, period=1.0):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    d = jnp.abs(x[:, None, 0] - y[None, :, 0])
    s = jnp.sin(jnp.pi * d / period)
    return variance * jnp.exp(-(2.0 * s**2) / (lengthscale**2))


class SimpleDataLoader:
    """Minimal iterable dataloader wrapping a list of batches."""

    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


@pytest.fixture
def setup_1d():
    """Standard 1-D regression setup with non-trivial data.

    Returns:
        Tuple of (model_fn, params, data, loss_fn).
    """
    key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)

    n_train = 20
    x_train = jr.uniform(k1, (n_train, 1), minval=-1.0, maxval=1.0)
    y_train = jnp.sin(2 * jnp.pi * x_train) + 0.1 * jr.normal(k2, (n_train, 1))

    model = TinyMLP(hidden=16)
    params = model.init(jr.PRNGKey(42), jnp.zeros((1, 1)))

    def model_fn(input, params):
        return model.apply(params, input)

    dataloader = SimpleDataLoader([{"input": x_train, "target": y_train}])

    def kernel_fn(x, y):
        return _rbf_kernel(x, y, lengthscale=0.5, variance=1.0)

    return {
        "model_fn": model_fn,
        "params": params,
        "x_train": x_train,
        "y_train": y_train,
        "dataloader": dataloader,
        "kernel_fn": kernel_fn,
        "key": jr.PRNGKey(99),
    }


@pytest.fixture
def setup_multioutput():
    """Multi-output regression setup.

    Returns:
        Tuple of (model_fn, params, data, loss_fn).
    """
    key = jr.PRNGKey(1)
    k1, k2 = jr.split(key)
    n_train = 15
    in_dim, out_dim = 2, 3

    x_train = jr.normal(k1, (n_train, in_dim))
    y_train = jr.normal(k2, (n_train, out_dim))

    model = TinyMLPMultiOutput(hidden=16, out_dim=out_dim)
    params = model.init(jr.PRNGKey(7), jnp.zeros((1, in_dim)))

    def model_fn(input, params):
        return model.apply(params, input)

    dataloader = SimpleDataLoader([{"input": x_train, "target": y_train}])

    def kernel_fn(x, y):
        return _rbf_kernel(x, y, lengthscale=1.0, variance=1.0)

    return {
        "model_fn": model_fn,
        "params": params,
        "x_train": x_train,
        "y_train": y_train,
        "dataloader": dataloader,
        "kernel_fn": kernel_fn,
        "key": jr.PRNGKey(11),
        "out_dim": out_dim,
    }


# ---------------------------------------------------------------------------
# Unit tests: _fsp_state_to_scale / _fsp_state_to_cov
# ---------------------------------------------------------------------------


def test_scale_mv_shape():
    P, R = 10, 3
    S = jr.normal(jr.PRNGKey(0), (P, R))
    state = {"scale_sqrt": S}
    mv = _fsp_state_to_scale(state)
    v = jnp.ones(R)
    out = mv(v)
    assert out.shape == (P,)


def test_cov_mv_shape():
    P, R = 10, 3
    S = jr.normal(jr.PRNGKey(0), (P, R))
    state = {"scale_sqrt": S}
    mv = _fsp_state_to_cov(state)
    v = jnp.ones(P)
    out = mv(v)
    assert out.shape == (P,)


def test_cov_equals_SST():
    """Cov mv should equal S @ S^T @ v."""
    P, R = 8, 4
    S = jr.normal(jr.PRNGKey(1), (P, R))
    state = {"scale_sqrt": S}
    mv = _fsp_state_to_cov(state)

    v = jr.normal(jr.PRNGKey(2), (P,))
    expected = S @ (S.T @ v)
    result = mv(v)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_cov_is_psd():
    """S S^T must be positive semi-definite."""
    P, R = 6, 3
    S = jr.normal(jr.PRNGKey(3), (P, R))
    cov_mat = S @ S.T
    eigvals = jnp.linalg.eigvalsh(cov_mat)
    assert jnp.all(eigvals >= -1e-5)


def test_cov_symmetry():
    """S S^T should be symmetric."""
    P, R = 6, 3
    S = jr.normal(jr.PRNGKey(4), (P, R))
    cov_mat = S @ S.T
    assert jnp.allclose(cov_mat, cov_mat.T, atol=1e-7)


def test_scale_identity():
    """If S = I, scale_mv should be identity."""
    P = 5
    S = jnp.eye(P)
    state = {"scale_sqrt": S}
    mv = _fsp_state_to_scale(state)
    v = jr.normal(jr.PRNGKey(5), (P,))
    assert jnp.allclose(mv(v), v, atol=1e-7)


# ---------------------------------------------------------------------------
# Unit tests: JVP / VJP helpers
# ---------------------------------------------------------------------------


def test_concatenated_model_jvp_shape(setup_1d):
    s = setup_1d
    x = s["x_train"][:8]
    result = _create_concatenated_model_jvp(s["model_fn"], s["params"], x, num_chunks=2)
    # Should be flattened: n_points * output_dim
    expected_len = x.shape[0] * 1  # output_dim = 1
    assert result.shape == (expected_len,)


def test_lanczos_init_is_unit_vector(setup_1d):
    s = setup_1d
    x = s["x_train"][:8]
    b = _lanczos_init_full(s["model_fn"], s["params"], x, num_chunks=2)
    assert jnp.allclose(jnp.linalg.norm(b), 1.0, atol=1e-5)


def test_lanczos_init_shape(setup_1d):
    s = setup_1d
    x = s["x_train"][:8]
    b = _lanczos_init_full(s["model_fn"], s["params"], x, num_chunks=2)
    assert b.shape == (x.shape[0],)  # 8 points * 1 output


def test_model_vjp_shape(setup_1d):
    s = setup_1d
    x = s["x_train"][:4]
    # vs: same shape as model output per point
    vs = jnp.ones((4, 1))
    result = _model_vjp(s["model_fn"], s["params"], x, vs, batch_size=1)
    # Result should be pytree with same structure as params, but with leading batch dim
    _flat_params, _ = jax.flatten_util.ravel_pytree(s["params"])
    result_leaves = jax.tree.leaves(result)
    # Each leaf should have shape (4, *original_shape)
    for leaf in result_leaves:
        assert leaf.shape[0] == 4


def test_vjp_jvp_adjoint_property(setup_1d):
    """<Jv, w> should equal <v, J^T w> (adjoint property)."""
    s = setup_1d
    key = jr.PRNGKey(77)
    x = s["x_train"][:3]

    # Random parameter-space vector
    flat_p, unravel = jax.flatten_util.ravel_pytree(s["params"])
    v_flat = jr.normal(key, flat_p.shape, dtype=flat_p.dtype)
    v_tree = unravel(v_flat)

    # Random output-space vector
    out_dtype = s["model_fn"](input=x[0], params=s["params"]).dtype
    w = jr.normal(jr.PRNGKey(78), (3, 1), dtype=out_dtype)

    # Forward: Jv
    def fwd(p):
        return jax.vmap(lambda xi: s["model_fn"](input=xi, params=p))(x)

    _, jv = jax.jvp(fwd, (s["params"],), (v_tree,))

    # Backward: J^T w
    jtw = _model_vjp(s["model_fn"], s["params"], x, w, batch_size=1)
    jtw_flat = jnp.concatenate([
        jnp.sum(l, axis=0).ravel() for l in jax.tree.leaves(jtw)
    ])

    lhs = jnp.sum(jv * w)
    rhs = jnp.dot(v_flat, jtw_flat)
    assert jnp.allclose(lhs, rhs, atol=1e-4), f"|{lhs} - {rhs}| = {abs(lhs - rhs)}"


# ---------------------------------------------------------------------------
# Unit tests: _compute_full_prior_inverse_sqrt
# ---------------------------------------------------------------------------


def test_output_shape_1d(setup_1d):
    s = setup_1d
    n_ctx = 12
    x_ctx = jnp.linspace(-1, 1, n_ctx).reshape(-1, 1)
    k_inv_sqrt, rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=s["kernel_fn"],
        model_fn=s["model_fn"],
        params=s["params"],
        x_context=x_ctx,
        output_shape=(n_ctx, 1),
        n_chunks_eff=2,
        jitter=1e-4,
    )
    assert k_inv_sqrt.shape[0] == n_ctx
    assert k_inv_sqrt.shape[1] == 1  # output_dim
    assert k_inv_sqrt.shape[2] == rank
    assert rank > 0


def test_output_shape_multioutput(setup_multioutput):
    s = setup_multioutput
    n_ctx = 10
    x_ctx = jr.normal(jr.PRNGKey(0), (n_ctx, 2))
    k_inv_sqrt, rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=s["kernel_fn"],
        model_fn=s["model_fn"],
        params=s["params"],
        x_context=x_ctx,
        output_shape=(n_ctx, s["out_dim"]),
        n_chunks_eff=2,
        jitter=1e-4,
    )
    assert k_inv_sqrt.shape == (n_ctx, s["out_dim"], rank)
    assert rank > 0


def test_rank_bounded_by_dim(setup_1d):
    """Rank of K^{-1/2} factor can't exceed n_context * output_dim."""
    s = setup_1d
    n_ctx = 8
    x_ctx = jnp.linspace(-1, 1, n_ctx).reshape(-1, 1)
    _, rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=s["kernel_fn"],
        model_fn=s["model_fn"],
        params=s["params"],
        x_context=x_ctx,
        output_shape=(n_ctx, 1),
        n_chunks_eff=2,
        jitter=1e-4,
    )
    assert rank <= n_ctx * 1  # n_ctx * output_dim


# ---------------------------------------------------------------------------
# Unit tests: _compute_M_memory_efficient
# ---------------------------------------------------------------------------


def test_M_shape(setup_1d):
    s = setup_1d
    n_ctx = 10
    x_ctx = jnp.linspace(-1, 1, n_ctx).reshape(-1, 1)

    k_inv_sqrt, rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=s["kernel_fn"],
        model_fn=s["model_fn"],
        params=s["params"],
        x_context=x_ctx,
        output_shape=(n_ctx, 1),
        n_chunks_eff=2,
        jitter=1e-4,
    )

    M = _compute_M_memory_efficient(
        s["model_fn"], s["params"], x_ctx, k_inv_sqrt, n_chunks_eff=2
    )
    flat_p, _ = jax.flatten_util.ravel_pytree(s["params"])
    # M should be (n_params, rank)
    assert M.shape == (flat_p.shape[0], rank)


def test_M_chunks_consistency(setup_1d):
    """M should be the same regardless of chunk count (just affects memory)."""
    s = setup_1d
    n_ctx = 10
    x_ctx = jnp.linspace(-1, 1, n_ctx).reshape(-1, 1)

    k_inv_sqrt, _ = _compute_full_prior_inverse_sqrt(
        kernel_fn=s["kernel_fn"],
        model_fn=s["model_fn"],
        params=s["params"],
        x_context=x_ctx,
        output_shape=(n_ctx, 1),
        n_chunks_eff=2,
        jitter=1e-4,
    )

    M1 = _compute_M_memory_efficient(
        s["model_fn"], s["params"], x_ctx, k_inv_sqrt, n_chunks_eff=1
    )
    M2 = _compute_M_memory_efficient(
        s["model_fn"], s["params"], x_ctx, k_inv_sqrt, n_chunks_eff=5
    )
    assert jnp.allclose(M1, M2, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration tests: create_fsp_curvature
# ---------------------------------------------------------------------------


def _run_curvature(setup, n_context=15, max_rank=None, **extra):
    s = setup
    return create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=n_context,
        context_selection="random",
        n_chunks=2,
        max_rank=max_rank,
        **extra,
    )


def test_returns_low_rank_terms(setup_1d):
    lr = _run_curvature(setup_1d, n_context=10)
    assert hasattr(lr, "U")
    assert hasattr(lr, "S")


def test_U_columns_match_S_length(setup_1d):
    lr = _run_curvature(setup_1d, n_context=10)
    assert lr.U.shape[1] == lr.S.shape[0]


def test_U_rows_match_param_count(setup_1d):
    lr = _run_curvature(setup_1d, n_context=10)
    flat_p, _ = jax.flatten_util.ravel_pytree(setup_1d["params"])
    assert lr.U.shape[0] == flat_p.shape[0]


def test_eigenvalues_nonnegative(setup_1d):
    lr = _run_curvature(setup_1d, n_context=10)
    assert jnp.all(lr.S >= -1e-8), f"Negative eigenvalues: {lr.S[lr.S < 0]}"


def test_rank_bounded_by_context(setup_1d):
    n_ctx = 10
    lr = _run_curvature(setup_1d, n_context=n_ctx)
    # Rank can't exceed n_context * output_dim
    assert lr.S.shape[0] <= n_ctx * 1


def test_max_rank_respected(setup_1d):
    lr = _run_curvature(setup_1d, n_context=15, max_rank=3)
    assert lr.S.shape[0] <= 3


def test_multioutput(setup_multioutput):
    lr = _run_curvature(setup_multioutput, n_context=10)
    flat_p, _ = jax.flatten_util.ravel_pytree(setup_multioutput["params"])
    assert lr.U.shape[0] == flat_p.shape[0]
    assert lr.S.shape[0] > 0


def test_unsupported_method_raises(setup_1d):
    s = setup_1d
    with pytest.raises(ValueError, match="Unsupported FSP algorithm"):
        create_fsp_curvature(
            model_fn=s["model_fn"],
            params=s["params"],
            data=s["dataloader"],
            loss_fn=LossFn.MSE,
            key=s["key"],
            kernel_fn=s["kernel_fn"],
            low_rank_method="eigendecomposition",  # not LANCZOS
            jitter=1e-4,
            ggn_factor=1.0,
            n_context_points=10,
            context_selection="random",
            n_chunks=2,
            max_rank=None,
        )


def test_tuple_data_format(setup_1d):
    """Dataloader yielding (x, y) tuples instead of dicts."""
    s = setup_1d
    tuple_loader = SimpleDataLoader([(s["x_train"], s["y_train"])])
    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=tuple_loader,
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    assert lr.S.shape[0] > 0


def test_multiple_batches(setup_1d):
    """Data split across multiple batches should still work."""
    s = setup_1d
    half = s["x_train"].shape[0] // 2
    multi_loader = SimpleDataLoader([
        {"input": s["x_train"][:half], "target": s["y_train"][:half]},
        {"input": s["x_train"][half:], "target": s["y_train"][half:]},
    ])
    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=multi_loader,
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    assert lr.S.shape[0] > 0


# ---------------------------------------------------------------------------
# Integration tests: create_fsp_posterior
# ---------------------------------------------------------------------------


def _make_posterior_fn(setup, n_context=12, **extra):
    s = setup
    return create_fsp_posterior(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=n_context,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
        **extra,
    )


def test_returns_callable(setup_1d):
    pf = _make_posterior_fn(setup_1d)
    assert callable(pf)


def test_posterior_has_required_fields(setup_1d):
    posterior = _make_posterior_fn(setup_1d)({"prior_prec": 1.0})
    assert posterior.rank > 0
    assert "scale_sqrt" in posterior.state
    assert posterior.cov_mv is not None
    assert posterior.scale_mv is not None
    assert posterior.low_rank_terms is not None


def test_scale_sqrt_shape(setup_1d):
    posterior = _make_posterior_fn(setup_1d)({"prior_prec": 1.0})
    flat_p, _ = jax.flatten_util.ravel_pytree(setup_1d["params"])
    ss = posterior.state["scale_sqrt"]
    assert ss.shape[0] == flat_p.shape[0]
    assert ss.shape[1] == posterior.rank


def test_prior_prec_scales_covariance(setup_1d):
    """Doubling prior_prec should halve the covariance eigenvalues."""
    pf = _make_posterior_fn(setup_1d)
    post1 = pf({"prior_prec": 1.0})
    post2 = pf({"prior_prec": 2.0})

    ss1 = post1.state["scale_sqrt"]
    ss2 = post2.state["scale_sqrt"]

    # Cov = S S^T, so cov eigenvalues scale as 1/prior_prec
    # ||S||_F^2 is trace of cov
    trace1 = jnp.sum(ss1**2)
    trace2 = jnp.sum(ss2**2)

    ratio = trace1 / trace2
    assert jnp.allclose(ratio, 2.0, atol=0.1), f"Trace ratio {ratio} != 2.0"


def test_rank_consistent(setup_1d):
    posterior = _make_posterior_fn(setup_1d)({"prior_prec": 1.0})
    assert posterior.rank == posterior.state["scale_sqrt"].shape[1]


def test_cov_mv_returns_pytree(setup_1d):
    """cov_mv should accept and return param-shaped pytrees."""
    s = setup_1d
    posterior = _make_posterior_fn(s)({"prior_prec": 1.0})
    # Build a random pytree matching params structure
    flat_p, unravel = jax.flatten_util.ravel_pytree(s["params"])
    v_tree = unravel(jr.normal(jr.PRNGKey(0), flat_p.shape))
    result = posterior.cov_mv(posterior.state)(v_tree)
    # Result should be pytree with same structure as params
    result_flat, _ = jax.flatten_util.ravel_pytree(result)
    assert result_flat.shape == flat_p.shape


def test_scale_mv_returns_pytree(setup_1d):
    s = setup_1d
    posterior = _make_posterior_fn(s)({"prior_prec": 1.0})
    # scale_mv input has rank dimensions
    rank = posterior.rank
    v = jr.normal(jr.PRNGKey(1), (rank,))
    result = posterior.scale_mv(posterior.state)(v)
    flat_r, _ = jax.flatten_util.ravel_pytree(result)
    flat_p, _ = jax.flatten_util.ravel_pytree(s["params"])
    assert flat_r.shape == flat_p.shape


def test_multioutput_posterior(setup_multioutput):
    posterior = _make_posterior_fn(setup_multioutput, n_context=8)({"prior_prec": 1.0})
    assert posterior.rank > 0
    flat_p, _ = jax.flatten_util.ravel_pytree(setup_multioutput["params"])
    assert posterior.state["scale_sqrt"].shape[0] == flat_p.shape[0]


# ---------------------------------------------------------------------------
# Mathematical property tests
# ---------------------------------------------------------------------------


def test_curvature_U_orthonormal_columns(setup_1d):
    """U from SVD should have orthonormal columns."""
    s = setup_1d
    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    gram = lr.U.T @ lr.U
    assert jnp.allclose(gram, jnp.eye(gram.shape[0]), atol=1e-5)


def test_eigenvalues_descending(setup_1d):
    """S should be in descending order (from the final SVD)."""
    s = setup_1d
    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    diffs = jnp.diff(lr.S)
    assert jnp.all(diffs <= 1e-8), "Eigenvalues not in descending order"


def test_posterior_cov_symmetric_via_mv(setup_1d):
    """v^T Cov w == w^T Cov v for random v, w."""
    s = setup_1d
    pf = create_fsp_posterior(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
    )
    posterior = pf({"prior_prec": 1.0})
    flat_p, unravel = jax.flatten_util.ravel_pytree(s["params"])
    P = flat_p.shape[0]

    v = unravel(jr.normal(jr.PRNGKey(10), (P,)))
    w = unravel(jr.normal(jr.PRNGKey(11), (P,)))

    cov_v = posterior.cov_mv(posterior.state)(v)
    cov_w = posterior.cov_mv(posterior.state)(w)

    cov_v_flat, _ = jax.flatten_util.ravel_pytree(cov_v)
    cov_w_flat, _ = jax.flatten_util.ravel_pytree(cov_w)
    v_flat, _ = jax.flatten_util.ravel_pytree(v)
    w_flat, _ = jax.flatten_util.ravel_pytree(w)

    lhs = jnp.dot(v_flat, cov_w_flat)
    rhs = jnp.dot(w_flat, cov_v_flat)
    assert jnp.allclose(lhs, rhs, atol=1e-5), f"|{lhs} - {rhs}| = {abs(lhs - rhs)}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_data_point():
    """FSP should handle a dataset with a single observation."""
    key = jr.PRNGKey(55)
    model = TinyMLP(hidden=8)
    params = model.init(key, jnp.zeros((1, 1)))

    def model_fn(input, params):
        return model.apply(params, input)

    x = jnp.array([[0.5]])
    y = jnp.array([[1.0]])
    loader = SimpleDataLoader([{"input": x, "target": y}])

    def kernel_fn(x, y):
        return _rbf_kernel(x, y)

    lr = create_fsp_curvature(
        model_fn=model_fn,
        params=params,
        data=loader,
        loss_fn=LossFn.MSE,
        key=key,
        kernel_fn=kernel_fn,
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-3,
        ggn_factor=1.0,
        n_context_points=5,
        context_selection="random",
        n_chunks=1,
        max_rank=None,
    )
    assert lr.S.shape[0] >= 0  # might be 0 or small, but shouldn't crash


def test_high_jitter_shrinks_rank(setup_1d):
    """Very high jitter should dominate the kernel and reduce effective rank."""
    s = setup_1d
    lr_low = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    lr_high = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e2,  # huge jitter
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    # With massive jitter, the trace of the covariance should be smaller
    trace_low = jnp.sum(lr_low.S)
    trace_high = jnp.sum(lr_high.S)
    assert trace_high <= trace_low + 1e-3


# ---------------------------------------------------------------------------
# Loss function tests (since fsp.py uses _create_loss_fn from ggn)
# ---------------------------------------------------------------------------


def test_mse_curvature(setup_1d):
    s = setup_1d
    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=LossFn.MSE,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    assert lr.S.shape[0] > 0


def test_callable_loss_fn(setup_1d):
    """Custom callable loss should work."""
    s = setup_1d

    def my_loss(pred, target):
        return jnp.mean((pred - target) ** 2)

    lr = create_fsp_curvature(
        model_fn=s["model_fn"],
        params=s["params"],
        data=s["dataloader"],
        loss_fn=my_loss,
        key=s["key"],
        kernel_fn=s["kernel_fn"],
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=10,
        context_selection="random",
        n_chunks=2,
        max_rank=None,
    )
    assert lr.S.shape[0] > 0


# ---------------------------------------------------------------------------
# Regression tests integrated
# ---------------------------------------------------------------------------


def test_fsp_posterior_creation_shapes():
    """Test create_fsp_posterior_fn shapes and rank."""
    key = jr.PRNGKey(0)
    x_data = jnp.zeros((10, 1))
    y_data = jnp.zeros((10, 1))

    model = SmallMLP()
    params = model.init(key, x_data)

    dataloader = SimpleDataLoader([{"input": x_data, "target": y_data}])

    posterior_fn = create_fsp_posterior(
        model_fn=lambda input, params: model.apply(params, input),
        params=params,
        data=dataloader,
        loss_fn=LossFn.MSE,
        key=key,
        kernel_fn=lambda x, y: kernel_periodic_1d(x, y, period=2.0),
        ggn_factor=1.0,
        low_rank_method=LowRankMethod.LANCZOS,
        n_context_points=20,
        context_selection="sobol",
        n_chunks=4,
        max_rank=None,
    )

    posterior = posterior_fn({"prior_prec": 1.0})

    assert posterior.rank > 0
    assert posterior.rank <= 20
    assert "scale_sqrt" in posterior.state
    # Check shape of scale_sqrt: (n_params, rank)
    flat, _ = jax.flatten_util.ravel_pytree(params)
    assert posterior.state["scale_sqrt"].shape[0] == flat.shape[0]
    assert posterior.state["scale_sqrt"].shape[1] == posterior.rank


def test_create_fsp_curvature_shapes():
    """Test create_fsp_curvature function directly."""
    key = jr.PRNGKey(1)
    model = SmallMLP()
    params = model.init(key, jnp.zeros((1, 1)))
    x_data = jnp.zeros((10, 1))
    y_data = jnp.zeros((10, 1))

    def kernel_fn(x, y):
        return jnp.dot(x, y.T) + jnp.eye(x.shape[0]) * 1e-4

    loss_fn_callable = _create_loss_fn(LossFn.MSE)

    # Returns LowRankTerms
    est_terms = create_fsp_curvature(
        model_fn=lambda input, params: model.apply(params, input),
        params=params,
        data=SimpleDataLoader([{"input": x_data, "target": y_data}]),
        loss_fn=loss_fn_callable,
        key=key,
        kernel_fn=kernel_fn,
        low_rank_method=LowRankMethod.LANCZOS,
        jitter=1e-4,
        ggn_factor=1.0,
        n_context_points=30,
        context_selection="sobol",
        context_bounds=(-1.0, 1.0),
        n_chunks=4,
        max_rank=None,
    )

    rank = est_terms.S.shape[0]
    assert rank > 0
    assert est_terms.U.shape[1] == rank


def test_compute_full_prior_inverse_sqrt_shape():
    """Unit test for Lanczos based prior inversion."""
    key = jr.PRNGKey(1)
    model = SmallMLP()
    params = model.init(key, jnp.zeros((1, 1)))
    x_context = jnp.linspace(-1, 1, 30).reshape(-1, 1)

    def kernel_fn(x, y):
        # Identity like kernel for easy rank
        return jnp.dot(x, y.T) + jnp.eye(x.shape[0]) * 1e-4

    k_inv_sqrt, rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=kernel_fn,
        model_fn=lambda input, params: model.apply(params, input),
        params=params,
        x_context=x_context,
        output_shape=(30, 1),
        n_chunks_eff=2,
    )

    # Output shape should be (N, Output_dim, Rank)
    # output_dim=1
    assert k_inv_sqrt.shape[0] == 30
    assert k_inv_sqrt.shape[1] == 1
    assert rank > 0
    assert k_inv_sqrt.shape[2] == rank


@pytest.mark.skipif(not GPJAX_AVAILABLE, reason="gpjax not installed")
def test_fsp_with_gpjax_kernel():
    """Test FSP working with a GPJax kernel wrapper."""
    key = jr.PRNGKey(2)
    x_data = jnp.zeros((10, 1))
    y_data = jnp.zeros((10, 1))

    model = SmallMLP()
    params = model.init(key, x_data)

    # Define GPJax kernel wrapper
    gpx_kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)

    def kernel_fn(x, y):
        # gpjax cross_covariance computes K(x, y)
        return gpx_kernel.cross_covariance(x, y)

    posterior_fn = create_fsp_posterior(
        model_fn=lambda input, params: model.apply(params, input),
        params=params,
        data=SimpleDataLoader([{"input": x_data, "target": y_data}]),
        loss_fn=LossFn.MSE,
        key=key,
        kernel_fn=kernel_fn,
        ggn_factor=1.0,
        low_rank_method=LowRankMethod.LANCZOS,
        n_context_points=20,
        context_selection="sobol",
        n_chunks=4,
        max_rank=None,
    )

    posterior = posterior_fn({"prior_prec": 1.0})
    assert posterior.rank > 0
    assert posterior.rank <= 20


def test_fsp_with_custom_rbf_kernel_implementation():
    """Test FSP with a custom manually implemented RBF kernel."""
    key = jr.PRNGKey(3)
    x_data = jnp.zeros((10, 1))
    y_data = jnp.zeros((10, 1))

    model = SmallMLP()
    params = model.init(key, x_data)

    def custom_rbf_kernel(x, y):
        # Manual RBF implementation
        # k(x,y) = sigma^2 * exp(-|x-y|^2 / 2l^2)
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        # Using squared euclidean distance
        dist_sq = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return 1.0 * jnp.exp(-0.5 * dist_sq / (1.0**2))

    posterior_fn = create_fsp_posterior(
        model_fn=lambda input, params: model.apply(params, input),
        params=params,
        data=SimpleDataLoader([{"input": x_data, "target": y_data}]),
        loss_fn=LossFn.MSE,
        key=key,
        kernel_fn=custom_rbf_kernel,
        ggn_factor=1.0,
        low_rank_method=LowRankMethod.LANCZOS,
        n_context_points=20,
        context_selection="sobol",
        n_chunks=4,
        max_rank=None,
    )

    posterior = posterior_fn({"prior_prec": 1.0})
    assert posterior.rank > 0
    assert posterior.rank <= 20
