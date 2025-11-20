import jax
import jax.numpy as jnp

from laplax.curv.fisher import (
    create_empirical_fisher_mv,
    create_MC_fisher_mv,
    sample_likelihood,
)
from laplax.enums import LossFn
from laplax.util.flatten import full_flatten


def test_emp_fisher_on_quadratic_fn():
    def fn(input, params):
        return jnp.array(params["a"] * input**2 + params["b"] * input + params["c"])

    data = {
        "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3, 1),
        "target": jnp.array([1.25, -0.11, 0.79]).reshape(3, 1),
    }

    best_params = {"a": jnp.array(1.5), "b": jnp.array(-0.5), "c": jnp.array(-0.25)}

    fisher_mv = create_empirical_fisher_mv(
        model_fn=fn,
        params=best_params,
        data=data,
        loss_fn=LossFn.MSE,
        vmap_over_data=True,
    )

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(fisher_mv({"a": 1.0, "b": 0.0, "c": 0.0}))
    fisher_row_2 = full_flatten(fisher_mv({"a": 0.0, "b": 1.0, "c": 0.0}))
    fisher_row_3 = full_flatten(fisher_mv({"a": 0.0, "b": 0.0, "c": 1.0}))
    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3))

    def df_dparams(input, params):
        del params
        df_da = input.item() ** 2
        df_db = input.item()
        df_dc = 1
        return jnp.array([[df_da, df_db, df_dc]])

    def dc_df(f, y):  # For MSE Loss
        return jnp.atleast_2d(2 * (f - y))

    jacs = [df_dparams(x, best_params) for x in data["input"]]
    grads = [
        dc_df(fn(x, best_params), y)
        for x, y in zip(data["input"], data["target"], strict=True)
    ]

    fisher_manual = jnp.mean(
        jnp.array([
            jac.T @ grad @ grad.T @ jac for jac, grad in zip(jacs, grads, strict=True)
        ]),
        axis=0,
    )
    assert jnp.allclose(fisher_laplax, fisher_manual)


def test_emp_fisher_on_quadratic_fn_2():
    # Carefully crafted example where all shapes are different to simplify debugging
    # n_data = 3
    # fn_output_dim = 2
    # Parameters: PyTree with two elements that are (2,)-tensors -> 4 params

    def fn(input, params):
        return jnp.array([
            params["a"][0] * input**2 + params["b"][0] * input,
            params["a"][1] * input + params["b"][1],
        ]).squeeze()

    data = {
        "input": jnp.array([0.3, 0.7, 0.4]).reshape(3, 1),
        "target": jnp.array([0.3, 0.7, 0.4, 0.5, 0.3, 0.7]).reshape(3, 2),
    }

    best_params = {"a": jnp.array([1.7, 2.3]), "b": jnp.array([-0.5, -1])}

    fisher_mv = create_empirical_fisher_mv(
        model_fn=fn,
        params=best_params,
        data=data,
        loss_fn=LossFn.MSE,
        vmap_over_data=True,
    )

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(
        fisher_mv({"a": jnp.array([1.0, 0.0]), "b": jnp.array([0.0, 0.0])})
    )
    fisher_row_2 = full_flatten(
        fisher_mv({"a": jnp.array([0.0, 1.0]), "b": jnp.array([0.0, 0.0])})
    )
    fisher_row_3 = full_flatten(
        fisher_mv({"a": jnp.array([0.0, 0.0]), "b": jnp.array([1.0, 0.0])})
    )
    fisher_row_4 = full_flatten(
        fisher_mv({"a": jnp.array([0.0, 0.0]), "b": jnp.array([0.0, 1.0])})
    )

    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3, fisher_row_4))

    def df_dparams(input, params):
        del params
        df_da0 = input.item() ** 2
        df_db0 = input.item()
        df_da1 = input.item()
        df_db1 = 1
        return jnp.array([[df_da0, 0.0, df_db0, 0.0], [0.0, df_da1, 0.0, df_db1]])

    def dc_df(f, y):  # For MSE Loss
        return 2 * (f - y)

    jacs = [df_dparams(x, best_params) for x in data["input"]]
    grads = [
        dc_df(fn(x, best_params).squeeze(), y.squeeze())
        for x, y in zip(data["input"], data["target"], strict=True)
    ]

    fisher_manual = jnp.mean(
        jnp.array([
            jac.T @ grad[:, None] @ grad[None, :] @ jac
            for jac, grad in zip(jacs, grads, strict=True)
        ]),
        axis=0,
    )

    assert jnp.allclose(fisher_laplax, fisher_manual)


def test_MSE_samples():
    key = jax.random.key(42)
    f_n = jnp.arange(5, dtype=float)
    samples = sample_likelihood(LossFn.MSE, f_n, 4, key)
    assert samples.shape == (5, 4)


def test_CE_samples():
    key = jax.random.key(42)
    f_n = jnp.arange(10, dtype=float)
    samples = sample_likelihood(LossFn.CROSS_ENTROPY, f_n, 4, key)
    assert samples.shape == (1, 4)


def test_MC_fisher():
    def fn(input, params):
        return jnp.array([
            params["a"][0] * input**3,
            params["a"][1] ** 2 * input,
        ]).squeeze()

    data = {
        "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3, 1),
        "target": jnp.array([-1.5, -0.04, 0.5145, 0.028, 3.295, 0.052]).reshape(3, 2),
    }

    params = {"a": jnp.array([1.5, 0.2])}

    key = jax.random.PRNGKey(42)

    mc_fisher_mv = create_MC_fisher_mv(
        model_fn=fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
        vmap_over_data=True,
        mc_samples=10,
        key=key,
    )

    mc_fisher_row_1 = full_flatten(mc_fisher_mv({"a": jnp.array([1.0, 0.0])}))
    mc_fisher_row_2 = full_flatten(mc_fisher_mv({"a": jnp.array([0.0, 1.0])}))

    mc_fisher = jnp.stack((mc_fisher_row_1, mc_fisher_row_2))

    emp_fisher_mv = create_empirical_fisher_mv(
        model_fn=fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
        vmap_over_data=True,
    )

    emp_fisher_row_1 = full_flatten(emp_fisher_mv({"a": jnp.array([1.0, 0.0])}))
    emp_fisher_row_2 = full_flatten(emp_fisher_mv({"a": jnp.array([0.0, 1.0])}))

    emp_fisher = jnp.stack((emp_fisher_row_1, emp_fisher_row_2))

    assert jnp.linalg.norm(mc_fisher - emp_fisher) < 100  # Very large, see comment
    # Current behaviour: When params are close to best params
    # (i.e. f(x) matches true y), the emp_fisher matrix vanishes,
    # and the mc fisher does this too for small mc_samples.
    # For larger mc_samples, it diverges.
    # If params are sub-optimal, the mc fisher converges to a value
    # that is distinct from that of the emp fisher. This is because
    # the mean of the MC samples for y converges to f(x),
    # which is different from y in this case.
