import jax.numpy as jnp

from laplax.curv.fisher import create_fisher_mv
from laplax.enums import FisherType, LossFn
from laplax.util.flatten import full_flatten


def test_emp_fisher_on_quadratic_fn():
    def fn(input, params):
        return jnp.array(params["a"] * input**2 + params["b"] * input + params["c"])

    data = {
        "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3, 1),
        "target": jnp.array([1.25, -0.11, 0.79]).reshape(3, 1),
    }

    # TODO(Luis Gindorf): Find out why test fails for different parameters
    best_params = {"a": jnp.array(1.0), "b": jnp.array(-0.5), "c": jnp.array(-0.25)}

    fisher_mv = create_fisher_mv(
        FisherType.EMPIRICAL,
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

    fisher_manual = jnp.sum(
        jnp.array([
            jac.T @ grad @ grad.T @ jac for jac, grad in zip(jacs, grads, strict=True)
        ]),
        axis=0,
    )

    assert jnp.allclose(fisher_laplax, fisher_manual)
