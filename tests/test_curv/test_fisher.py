import jax
import jax.numpy as jnp

from laplax.curv.fisher import (
    create_empirical_fisher_mv,
    create_MC_fisher_mv,
    sample_likelihood,
)
from laplax.enums import LossFn
from laplax.util.flatten import full_flatten

import pytest
from .cases.fisher import FisherCase

@pytest.fixture
def case1():
    return FisherCase(
    fn = lambda input, params: jnp.array(params[0] * input**2 + params[1] * input + params[2]),
    data = {
        "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3, 1),
        "target": jnp.array([1.25, -0.11, 0.79]).reshape(3, 1),
        },
    params = jnp.array([1.5, -0.5, -0.25]),
    loss = lambda fn,y: ((fn - y)**2).sum()
)

@pytest.fixture
def case2():
    # Carefully crafted case where all shapes are different to simplify debugging
    # n_data = 3
    # fn_output_dim = 2
    # Parameters: PyTree with two elements that are (2,)-tensors -> 4 params
    return FisherCase(
    fn = lambda input, params: jnp.array([
            params[0] * input**2 + params[1] * input,
            params[2] * input + params[3],
        ]).squeeze(),
    data = {
        "input": jnp.array([0.3, 0.7, 0.4]).reshape(3, 1),
        "target": jnp.array([0.3, 0.7, 0.4, 0.5, 0.3, 0.7]).reshape(3, 2),
    },
    params = jnp.array([1.7, 2.3, -0.5, -1]),
    loss = lambda fn,y: ((fn - y)**2).sum()
)


def test_emp_fisher_on_quadratic_fn(case1):
    
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case1.fn,
        params=case1.params,
        data=case1.data,
        loss_fn=case1.loss,
        vmap_over_data=True,
    )

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(fisher_mv(jnp.array([1., 0., 0.])))
    fisher_row_2 = full_flatten(fisher_mv(jnp.array([0., 1., 0.])))
    fisher_row_3 = full_flatten(fisher_mv(jnp.array([0., 0., 1.])))
    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3))

    assert jnp.allclose(fisher_laplax, case1.fisher_manual)


def test_emp_fisher_on_quadratic_fn_2(case2):
    
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case2.fn,
        params=case2.params,
        data=case2.data,
        loss_fn=case2.loss,
        vmap_over_data=True,
    )

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(fisher_mv(jnp.array([1., 0., 0., 0.])))
    fisher_row_2 = full_flatten(fisher_mv(jnp.array([0., 1., 0., 0.])))
    fisher_row_3 = full_flatten(fisher_mv(jnp.array([0., 0., 1., 0.])))
    fisher_row_4 = full_flatten(fisher_mv(jnp.array([0., 0., 0., 1.])))
    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3, fisher_row_4))
    print(fisher_laplax)
    print(case2.fisher_manual)
    assert jnp.allclose(fisher_laplax, case2.fisher_manual)


def test_emp_fisher_without_data_vmap(case1):
    
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case1.fn,
        params=case1.params,
        data=case1.data,
        loss_fn=case1.loss,
        vmap_over_data=False,
    )

    # Construct full matrix via mvp with one-hot vectors as PyTrees
    fisher_row_1 = full_flatten(fisher_mv(jnp.array([1., 0., 0.])))
    fisher_row_2 = full_flatten(fisher_mv(jnp.array([0., 1., 0.])))
    fisher_row_3 = full_flatten(fisher_mv(jnp.array([0., 0., 1.])))
    fisher_laplax = jnp.stack((fisher_row_1, fisher_row_2, fisher_row_3))

    
    assert jnp.allclose(fisher_laplax, case1.fisher_manual)


def test_MSE_samples():
    key = jax.random.key(42)
    f_n = jnp.arange(5, dtype=float)
    samples = sample_likelihood("mse", f_n, 4, key)
    assert samples.shape == (4,5)


def test_CE_samples():
    key = jax.random.key(42)
    f_n = jnp.arange(10, dtype=float)
    samples = sample_likelihood(LossFn.CROSS_ENTROPY, f_n, 4, key)
    assert samples.shape == (4,1)


def test_BCE_samples():
    key = jax.random.key(42)
    f_n = jnp.array(0.6, dtype=float)
    samples = sample_likelihood(LossFn.BINARY_CROSS_ENTROPY, f_n, 4, key)
    assert samples.shape == (4,1)


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
