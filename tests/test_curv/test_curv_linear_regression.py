import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from laplax.curv.ggn import create_ggn_mv
from laplax.curv.hessian import create_hessian_mv
from laplax.enums import LossFn
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import to_dense
from laplax.util.tree import get_size


def test_ggn_linear_regression():
    """Test the GGN of linear regression."""
    D_in, D_out = 5, 3
    N = 10
    key = jax.random.split(jax.random.key(0), 3)
    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input

    # Compute manual GGN (ground truth)
    xxT = jnp.einsum("ni,nj->ij", X, X)
    G_manual = jnp.kron(2 * jnp.eye(D_out), xxT)

    # Compute the GGN using package
    ggn_mv = create_ggn_mv(
        model_fn,
        params=params,
        data={"input": X, "target": y},
        loss_fn=LossFn.MSE,
        num_total_samples=N,
    )

    G_calc = to_dense(ggn_mv, layout=params).swapaxes(0, 1).reshape(-1, D_out * D_in)

    # Compare results
    np.testing.assert_allclose(G_manual, G_calc, atol=5 * 1e-6)


def test_ggn_linear_regression_2():
    D_in, D_out, N = 5, 3, 10
    D_flat = D_in * D_out
    model = nnx.Linear(D_in, D_out, use_bias=False, rngs=nnx.Rngs(0))
    graph_def, state = nnx.split(model)

    # Model function
    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    # Set up data
    key = jax.random.split(jax.random.key(0), 3)
    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))

    # Manually calculate GGN
    xxT = jnp.einsum("ni,nj->ij", X, X)
    G_manual = jnp.kron(2 * jnp.eye(D_out), xxT)

    # Calculate the GGN and flatten
    ggn_mv = create_ggn_mv(
        model_fn,
        state,
        data={"input": X, "target": y},
        loss_fn=LossFn.MSE,
        num_total_samples=N,
    )
    flatten, unflatten = create_pytree_flattener(state)
    ggn_mv = wrap_function(ggn_mv, input_fn=unflatten, output_fn=flatten)
    num_params = get_size(state)
    G = to_dense(ggn_mv, layout=num_params)

    # Compare results
    G_manual = (
        G_manual.reshape(D_out, D_in, D_out, D_in)
        .swapaxes(0, 1)
        .swapaxes(-1, -2)
        .reshape(D_flat, D_flat)
    )
    np.testing.assert_allclose(G_manual, G, atol=5 * 1e-6)


def test_hessian_linear_regression():
    # Set up model, data, and parameters
    D_in, D_out = 5, 3
    N = 10
    key = jax.random.split(jax.random.key(0), 3)
    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))
    params = jax.random.normal(key[2], (D_out, D_in))

    def model_fn(input, params):
        return params @ input

    # Compute manual Hessian (ground truth)
    xxT = jnp.einsum("ni,nj->ij", X, X)
    hessian_manual = jnp.kron(2 * jnp.eye(D_out), xxT)

    # Compute the Hessian using package
    hessian_mv = create_hessian_mv(
        model_fn=model_fn,
        params=params,
        data={"input": X, "target": y},
        loss_fn=LossFn.MSE,
        num_total_samples=N,
        has_batch=True,
    )

    hessian_calc = (
        to_dense(hessian_mv, layout=params).swapaxes(0, 1).reshape(-1, D_out * D_in)
    )

    # Compare results
    np.testing.assert_allclose(hessian_manual, hessian_calc, atol=5 * 1e-6)


def test_hessian_linear_regression_2():
    D_in, D_out, N = 5, 3, 10
    D_flat = D_in * D_out
    model = nnx.Linear(D_in, D_out, use_bias=False, rngs=nnx.Rngs(0))
    graph_def, state = nnx.split(model)

    # Model function
    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    # Set up data
    key = jax.random.split(jax.random.key(0), 3)
    X = jax.random.normal(key[0], (N, D_in))
    y = jax.random.normal(key[1], (N, D_out))

    # Manually calculate Hessian
    xxT = jnp.einsum("ni,nj->ij", X, X)
    hessian_manual = jnp.kron(2 * jnp.eye(D_out), xxT)

    # Calculate the Hessian and flatten
    hessian_mv = create_hessian_mv(
        model_fn=model_fn,
        params=state,
        data={"input": X, "target": y},
        loss_fn=LossFn.MSE,
        num_total_samples=N,
        has_batch=True,
    )
    flatten, unflatten = create_pytree_flattener(state)
    hessian_mv = wrap_function(hessian_mv, input_fn=unflatten, output_fn=flatten)
    num_params = get_size(state)
    H = to_dense(hessian_mv, layout=num_params)

    # Compare results
    hessian_manual = (
        hessian_manual.reshape(D_out, D_in, D_out, D_in)
        .swapaxes(0, 1)
        .swapaxes(-1, -2)
        .reshape(D_flat, D_flat)
    )
    np.testing.assert_allclose(hessian_manual, H, atol=5 * 1e-6)
