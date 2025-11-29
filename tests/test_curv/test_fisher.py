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
import pytest_cases


def case1():
    return FisherCase(
    n = 3,
    o = 1,
    i = 1,
    l = 1,
    p = 3,
    fn = lambda input, params: params[0] * input**2 + params[1] * input + params[2],
    data = {
        "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3,1),
        "target": jnp.array([1.25, -0.11, 0.79]).reshape(3,1),
        },
    params = jnp.array([1.5, -0.5, -0.25]).reshape(3),
    loss = lambda fn,y: ((fn - y)**2).sum(axis=-1)
)

def case2():
    def fn(input, params):
        input = jnp.squeeze(input, axis=-1) # get rid of singleton data dimension
        return jnp.array([
            params[0] * input**2 + params[1] * input,
            params[2] * input + params[3],
        ])
    return FisherCase(
    n = 3,
    o = 2,
    i = 1,
    l = 2,
    p = 4,
    fn = fn,
    data = {
        "input": jnp.array([0.3, 0.7, 0.4]).reshape(3, 1),
        "target": jnp.array([0.3, 0.7, 0.4, 0.5, 0.3, 0.7]).reshape(3, 2),
    },
    params = jnp.array([1.7, 2.3, -0.5, -1]),
    loss = lambda fn,y: ((fn - y)**2).sum(axis=-1)
)

@pytest.mark.parametrize("i", [0, 1])
def cases(i):
    return [case1(), case2()][i]


@pytest_cases.parametrize_with_cases("case", cases=[cases])
def test_emp_fisher(case):
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case.fn,
        params=case.params,
        data=case.data,
        loss_fn=case.loss,
        vmap_over_data=True,
    )
    fisher_laplax = case.construct_fisher(fisher_mv)
    
    assert jnp.allclose(fisher_laplax, case.fisher_manual)

@pytest.fixture
def case_single_datum():
    def fn(input, params):
        input = jnp.squeeze(input, axis=-1) # get rid of singleton data dimension
        return jnp.array([
            params[0] * input**2 + params[1] * input,
            params[2] * input + params[3],
        ])
    return FisherCase(
    n = 1,
    o = 2,
    i = 1,
    l = 2,
    p = 4,
    fn = fn,
    data = {
        "input": jnp.array([0.3]).reshape((1,1)),
        "target": jnp.array([0.3, 0.7]).reshape((1,2)),
    },
    params = jnp.array([1.7, 2.3, -0.5, -1]),
    loss = lambda fn,y: ((fn - y)**2).sum(axis=-1)
)


def test_emp_fisher_single_datum(case_single_datum):
    case = case_single_datum

    fisher_mv = create_empirical_fisher_mv(
        model_fn=case.fn,
        params=case.params,
        data=case.data,
        loss_fn=case.loss,
        vmap_over_data=False,
    )
    fisher_laplax = case.construct_fisher(fisher_mv)
    assert jnp.allclose(fisher_laplax, case.fisher_manual)


@pytest_cases.parametrize_with_cases("case", cases=[cases])
def test_emp_fisher_without_data_vmap(case):
    case.handle_batches = True
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case.fn,
        params=case.params,
        data=case.data,
        loss_fn=case.loss,
        vmap_over_data=False,
    )
    fisher_laplax = case.construct_fisher(fisher_mv)
    assert jnp.allclose(fisher_laplax, case.fisher_manual)


def test_emp_fisher_with_pytree_params():
    # Can not ue FisherCase class here because it only supports array parameters
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


@pytest.fixture
def case_CE():
    def fn(input, params):
        input = input.squeeze(axis=-1)
        return jnp.array([
            params[0] * input + params[1],
            params[2] * input + params[3],
        ])

    def CE(fn, y):
        return (fn[y] - jnp.logaddexp(fn[0], fn[1])).sum(axis=-1)

    return FisherCase(
        n = 3,
        o = 2,
        i = 1,
        l = 1,
        p = 4,
        fn = fn,
        data = {
            "input": jnp.array([-1.0, 0.7, -0.5]).reshape(3, 1),
            "target": jnp.array([0, 1, 0]).reshape(3, 1),
            },
        params = jnp.array([1.0, 0.5, -1.0, 0.5]),
        loss = CE
        )


def test_cross_entropy_loss(case_CE):
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case_CE.fn,
        params=case_CE.params,
        data=case_CE.data,
        loss_fn=case_CE.loss,
        vmap_over_data=True,
    )
    fisher_laplax = case_CE.construct_fisher(fisher_mv)
    
    assert jnp.allclose(fisher_laplax, case_CE.fisher_manual)


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

@pytest.mark.skip
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

    key = jax.random.key(42)

    mc_fisher_mv = create_MC_fisher_mv(
        model_fn=fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
        vmap_over_data=True,
        mc_samples=1,
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
