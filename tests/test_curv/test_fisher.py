import jax.numpy as jnp
import pytest

from laplax.curv.fisher import create_fisher_mv_without_data
from laplax.enums import FisherType, LossFn
from laplax.util.flatten import full_flatten

def test_emp_fisher_on_quadratic_fn():

    def fn(input, params):
        return jnp.atleast_1d(params["a"] * input**2 + params["b"] * input + params["c"])

    data = {
        "input": jnp.array([-1., 0.7, 1.3]).reshape(3, 1),
        "target": jnp.array([1.25, -0.11, 0.79]).reshape(3, 1),
        }

    # TODO: Find out why test fails for different parameters
    best_params = {"a": jnp.array(1.0), "b": jnp.array(-0.5), "c": jnp.array(-0.25)}


    fisher_mv = create_fisher_mv_without_data(
        FisherType.EMPIRICAL,
        model_fn=fn,
        params=best_params,
        loss_fn=LossFn.MSE,
        factor=1,
        vmap_over_data=True,
        )

    def wrapped_fisher_mv(vec):
        return fisher_mv(vec, data)

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(wrapped_fisher_mv({"a": 1., "b": 0., "c": 0.}))
    fisher_row_2 = full_flatten(wrapped_fisher_mv({"a": 0., "b": 1., "c": 0.}))
    fisher_row_3 = full_flatten(wrapped_fisher_mv({"a": 0., "b": 0., "c": 1.}))
    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3))

    def df_dparams(input, params):
        df_da = input.item()**2
        df_db = input.item()
        df_dc = 1
        return jnp.array([[df_da,df_db,df_dc]])

    def dc_df(f,y): # For MSE Loss
        return jnp.atleast_2d(2*(f - y))

    jacs = [df_dparams(x, best_params) for x in data["input"]]
    grads = [dc_df(fn(x, best_params),y) for x,y in zip(data["input"],data["target"])]

    fisher_manual = jnp.sum(jnp.array([jac.T @ grad @ grad.T @ jac for jac, grad in zip(jacs, grads)]), axis=0)

    assert jnp.allclose(fisher_laplax, fisher_manual)
