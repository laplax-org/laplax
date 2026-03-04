# /tests/test_curv/test_ggn.py

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

from laplax.api import GGN
from laplax.curv.ggn import (
    create_ggn_mv,
    create_ggn_mv_fsp_without_data,
)
from laplax.enums import LossFn
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.loader import input_target_split
from laplax.util.mv import to_dense

from .cases.rosenbrock import RosenbrockCase

# ---------------------------------------------------------------
# GGN - Rosenbrock
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def test_ggn_rosenbrock(rosenbrock):
    # Setup ggn_mv
    ggn_mv = create_ggn_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        num_curv_samples=1,
        num_total_samples=1,
    )

    # Compute the GGN
    ggn_calc = jax.lax.map(ggn_mv, jnp.eye(2))

    # Compare with the manual GGN
    ggn_manual = rosenbrock.ggn_manual
    assert jnp.allclose(ggn_calc, ggn_manual)


# ---------------------------------------------------------------
# FSP-based GGN Tests
# ---------------------------------------------------------------


@pytest.mark.parametrize("loss_fn", [LossFn.MSE, LossFn.CROSS_ENTROPY])
def test_ggn_fsp_loss_functions(loss_fn):
    """Test FSP GGN with different loss functions."""
    key = jax.random.split(jax.random.key(42), 4)
    D_in, D_out, N = 3, 2, 5

    X = jax.random.normal(key[0], (N, D_in))
    if loss_fn == LossFn.MSE:
        y = jax.random.normal(key[1], (N, D_out))
    else:
        y = jax.random.randint(key[1], (N,), 0, D_out)

    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    # FSP GGN
    ggn_mv_fsp = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=1.0,
        vmap_over_data=True,
    )

    data = {"input": X, "target": y}
    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec, data)

    # Check that result has correct shape
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )

    # Check that result is not all zeros
    assert jax.tree.all(jax.tree.map(lambda x: jnp.any(jnp.abs(x) > 1e-8), result_fsp))


@pytest.mark.parametrize("vmap_over_data", [True, False])
def test_ggn_fsp_vmap_over_data(vmap_over_data):
    """Test FSP GGN with different vmap_over_data settings."""
    key = jax.random.split(jax.random.key(43), 3)
    D_in, D_out, N = 4, 3, 6

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    # FSP GGN
    ggn_mv_fsp = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=vmap_over_data,
    )

    data = {"input": X, "target": y}
    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec, data)

    # Check that result has correct shape
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )


def test_ggn_fsp_with_dataloader():
    """Test FSP GGN with dataloader through API."""
    key = jax.random.split(jax.random.key(44), 3)
    D_in, D_out, N, batch_size = 3, 2, 10, 3

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    class ReusableDataloader:
        def __iter__(self):
            for i in range(0, N, batch_size):
                yield (X[i : i + batch_size], y[i : i + batch_size])

    ggn_mv_fsp = GGN(
        model_fn=model_fn,
        params=params,
        data=ReusableDataloader(),
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
        transform=input_target_split,
    )

    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec)

    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )
    assert jax.tree.all(jax.tree.map(lambda x: jnp.any(jnp.abs(x) > 1e-8), result_fsp))


def test_ggn_fsp_factor_scaling():
    """Test that FSP GGN correctly applies factor scaling."""
    key = jax.random.split(jax.random.key(45), 3)
    D_in, D_out, N = 2, 2, 4

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    factor = 2.5

    ggn_mv_fsp_1 = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
    )

    ggn_mv_fsp_2 = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=factor,
        vmap_over_data=True,
    )

    data = {"input": X, "target": y}
    test_vec = jax.tree.map(jnp.ones_like, params)

    result_1 = ggn_mv_fsp_1(test_vec, data)
    result_2 = ggn_mv_fsp_2(test_vec, data)

    # Check that result_2 is factor times result_1
    assert jax.tree.all(
        jax.tree.map(
            lambda x, y: jnp.allclose(x * factor, y, atol=1e-6),
            result_1,
            result_2,
        )
    )


def test_ggn_fsp_custom_loss():
    """Test FSP GGN with custom loss function."""
    key = jax.random.split(jax.random.key(46), 3)
    D_in, D_out, N = 2, 2, 4

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    def custom_loss(pred, target):
        return jnp.mean((pred - target) ** 3)

    # FSP GGN with custom loss
    ggn_mv_fsp = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=custom_loss,
        factor=1.0,
        vmap_over_data=True,
    )

    data = {"input": X, "target": y}
    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec, data)

    # Check that result has correct shape
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )

    # Check that result is not all zeros
    assert jax.tree.all(jax.tree.map(lambda x: jnp.any(jnp.abs(x) > 1e-8), result_fsp))


def test_ggn_fsp_api_single_batch():
    """Test FSP GGN through API with single batch."""
    key = jax.random.split(jax.random.key(47), 3)
    D_in, D_out, N = 3, 2, 5

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    data = {"input": X, "target": y}

    # FSP GGN through API
    ggn_mv_fsp = GGN(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec)

    # Check that result has correct shape
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )

    # Check that result is not all zeros
    assert jax.tree.all(jax.tree.map(lambda x: jnp.any(jnp.abs(x) > 1e-8), result_fsp))


def test_ggn_fsp_single_sample_batch():
    """Test FSP GGN with single sample batch (edge case)."""
    key = jax.random.split(jax.random.key(48), 3)
    D_in, D_out = 3, 2

    X = jax.random.normal(key[0], (1, D_in))
    y = jax.random.normal(key[1], (1, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    # FSP GGN
    ggn_mv_fsp = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
    )

    data = {"input": X, "target": y}
    test_vec = jax.tree.map(jnp.ones_like, params)

    result_fsp = ggn_mv_fsp(test_vec, data)

    # Check that result has correct shape
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )


def test_ggn_fsp_linear_regression_ground_truth():
    """Test FSP GGN against manual GGN computation for linear regression.

    Note: This tests the GGN component computation, not the full FSP-Laplace
    precision matrix which includes the GP prior term.
    """
    key = jax.random.split(jax.random.key(50), 3)
    D_in, D_out, N = 5, 3, 10

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    # Manual GGN computation (ground truth for the GGN component)
    xxT = jnp.einsum("ni,nj->ij", X, X)
    G_manual = jnp.kron(2 * jnp.eye(D_out), xxT)

    # FSP GGN (computes the GGN component using lax.map)
    ggn_mv_fsp = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data={"input": X, "target": y},
        loss_fn=LossFn.MSE,
        num_total_samples=N,
        vmap_over_data=True,
    )

    # Convert to FSP version
    flatten, unflatten = create_pytree_flattener(params)
    ggn_mv_fsp_flat = wrap_function(ggn_mv_fsp, input_fn=unflatten, output_fn=flatten)

    # Compute dense GGN matrix
    num_params = D_out * D_in
    G_fsp = to_dense(ggn_mv_fsp_flat, layout=num_params)

    # Reshape to match manual computation
    G_fsp_reshaped = G_fsp.swapaxes(0, 1).reshape(-1, D_out * D_in)

    # Compare results (this validates the GGN component computation)
    np.testing.assert_allclose(G_manual, G_fsp_reshaped, atol=5e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_ggn_fsp_batch_consistency(batch_size):
    """Test that FSP GGN processes batches correctly with lax.map."""
    key = jax.random.split(jax.random.key(51), 3)
    D_in, D_out, N = 3, 2, 10

    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input.T

    # FSP GGN
    ggn_mv_fsp = create_ggn_mv_fsp_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
    )

    # Process data in batches
    test_vec = jax.tree.map(jnp.ones_like, params)
    result_fsp = jax.tree.map(jnp.zeros_like, params)

    for i in range(0, N, batch_size):
        batch_data = {
            "input": X[i : i + batch_size],
            "target": y[i : i + batch_size],
        }
        result_fsp = jax.tree.map(
            operator.add, result_fsp, ggn_mv_fsp(test_vec, batch_data)
        )

    # Check that result has correct shape and is non-zero
    assert jax.tree.all(
        jax.tree.map(lambda x, p: x.shape == p.shape, result_fsp, params)
    )
    assert jax.tree.all(jax.tree.map(lambda x: jnp.any(jnp.abs(x) > 1e-8), result_fsp))
