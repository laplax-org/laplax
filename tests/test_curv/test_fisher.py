import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases
import torch
from curvlinops import EFLinearOperator, FisherMCLinearOperator

from laplax.curv.fisher import (
    create_empirical_fisher_mv,
    create_MC_fisher_mv,
    sample_likelihood,
)
from laplax.curv.ggn import create_ggn_mv
from laplax.enums import LossFn
from laplax.util.flatten import create_pytree_flattener, full_flatten, wrap_function
from laplax.util.mv import to_dense
from tests.conftest import input_target_split_jax

from .cases.fisher import FisherCase


def case1():
    return FisherCase(
        n=3,
        o=1,
        i=1,
        l=1,
        p=3,
        fn=lambda input, params: params[0] * input**2 + params[1] * input + params[2],
        data={
            "input": jnp.array([-1.0, 0.7, 1.3]).reshape(3, 1),
            "target": jnp.array([1.25, -0.11, 0.79]).reshape(3, 1),
        },
        params=jnp.array([1.5, -0.5, -0.25]).reshape(3),
        loss=lambda fn, y: ((fn - y) ** 2).sum(axis=-1),
    )


def case2():
    def fn(input, params):
        input = jnp.squeeze(input, axis=-1)  # get rid of singleton data dimension
        return jnp.array([
            params[0] * input**2 + params[1] * input,
            params[2] * input + params[3],
        ])

    return FisherCase(
        n=3,
        o=2,
        i=1,
        l=2,
        p=4,
        fn=fn,
        data={
            "input": jnp.array([0.3, 0.7, 0.4]).reshape(3, 1),
            "target": jnp.array([0.3, 0.7, 0.4, 0.5, 0.3, 0.7]).reshape(3, 2),
        },
        params=jnp.array([1.7, 2.3, -0.5, -1]),
        loss=lambda fn, y: ((fn - y) ** 2).sum(axis=-1),
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
        num_curv_samples=case.n,
        num_total_samples=1,
    )
    fisher_laplax = case.construct_fisher(fisher_mv)

    assert jnp.allclose(fisher_laplax, case.fisher_manual)


@pytest.fixture
def case_single_datum():
    def fn(input, params):
        input = jnp.squeeze(input, axis=-1)  # get rid of singleton data dimension
        return jnp.array([
            params[0] * input**2 + params[1] * input,
            params[2] * input + params[3],
        ])

    return FisherCase(
        n=1,
        o=2,
        i=1,
        l=2,
        p=4,
        fn=fn,
        data={
            "input": jnp.array([0.3]).reshape((1, 1)),
            "target": jnp.array([0.3, 0.7]).reshape((1, 2)),
        },
        params=jnp.array([1.7, 2.3, -0.5, -1]),
        loss=lambda fn, y: ((fn - y) ** 2).sum(axis=-1),
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
        num_curv_samples=case.n,
        num_total_samples=1,
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
        num_curv_samples=3,
        num_total_samples=1,
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
        n=3,
        o=2,
        i=1,
        l=1,
        p=4,
        fn=fn,
        data={
            "input": jnp.array([-1.0, 0.7, -0.5]).reshape(3, 1),
            "target": jnp.array([0, 1, 0]).reshape(3, 1),
        },
        params=jnp.array([1.0, 0.5, -1.0, 0.5]),
        loss=CE,
    )


def test_cross_entropy_loss(case_CE):
    fisher_mv = create_empirical_fisher_mv(
        model_fn=case_CE.fn,
        params=case_CE.params,
        data=case_CE.data,
        loss_fn=case_CE.loss,
        vmap_over_data=True,
        num_curv_samples=case_CE.n,
        num_total_samples=1,
    )
    fisher_laplax = case_CE.construct_fisher(fisher_mv)

    assert jnp.allclose(fisher_laplax, case_CE.fisher_manual)


KEY = jax.random.key(42)


def test_MSE_samples():
    f_n = jnp.arange(5, dtype=float)
    samples = sample_likelihood("mse", f_n, 4, KEY)
    assert samples.shape == (4, 5)


def test_MSE_samples_batch():
    f_ns = jnp.arange(30, dtype=float).reshape((5, 6))
    samples = sample_likelihood("mse", f_ns, 4, KEY)
    assert samples.shape == (5, 4, 6)


def test_BCE_samples():
    f_n = jnp.array(0.6, dtype=float).reshape(1)
    samples = sample_likelihood(LossFn.BINARY_CROSS_ENTROPY, f_n, 4, KEY)
    assert samples.shape == (4, 1)


def test_BCE_samples_batch():
    f_ns = jax.random.uniform(KEY, (3, 1))
    samples = sample_likelihood(LossFn.BINARY_CROSS_ENTROPY, f_ns, 4, KEY)
    assert samples.shape == (3, 4, 1)


def test_CE_samples():
    f_n = jax.random.uniform(KEY, 10)
    samples = sample_likelihood(LossFn.CROSS_ENTROPY, f_n, 4, KEY)
    assert samples.shape == (4, 1)


def test_CE_samples_batch():
    f_ns = jax.random.uniform(KEY, (6, 10))
    samples = sample_likelihood(LossFn.CROSS_ENTROPY, f_ns, 4, KEY)
    assert samples.shape == (6, 4, 1)


def test_emp_fisher_against_curvlinops(trained_laplace_comparison):
    la_case = trained_laplace_comparison
    params = [p for p in la_case.torch_model.parameters() if p.requires_grad]

    torch_mv = EFLinearOperator(
        la_case.torch_model,
        torch.nn.MSELoss(),
        params,
        [(la_case.X_train, la_case.y_train)],
    )
    torch_curv = torch_mv @ torch.eye(torch_mv.shape[0])

    train_batch = input_target_split_jax(next(iter(la_case.train_loader)))
    jax_mv = create_empirical_fisher_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="mse",
        num_curv_samples=150,
        num_total_samples=1,
        vmap_over_data=False,
    )
    flatten, unflatten = create_pytree_flattener(la_case.params)
    jax_curv = to_dense(
        wrap_function(jax_mv, unflatten, flatten), layout=flatten(la_case.params)
    )
    np.testing.assert_allclose(
        np.sort(jnp.abs(torch_curv).sum(axis=-1))
        / np.sort(jnp.abs(jax_curv).sum(axis=-1)),
        1,
        atol=1e-2,
    )


def test_MC_fisher_against_curvlinops(trained_laplace_comparison):
    la_case = trained_laplace_comparison
    params = [p for p in la_case.torch_model.parameters() if p.requires_grad]

    torch_mv = FisherMCLinearOperator(
        la_case.torch_model,
        torch.nn.MSELoss(),
        params,
        [(la_case.X_train, la_case.y_train)],
        mc_samples=1000,
    )
    torch_curv = torch_mv @ torch.eye(torch_mv.shape[0])

    train_batch = input_target_split_jax(next(iter(la_case.train_loader)))

    jax_mv = create_MC_fisher_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="mse",
        mc_samples=10000,
        num_curv_samples=150,
        num_total_samples=1,
        vmap_over_data=True,
    )
    def jax_mv_with_key(vec):
        return jax_mv(vec, KEY)

    flatten, unflatten = create_pytree_flattener(la_case.params)
    jax_curv = to_dense(
        wrap_function(jax_mv_with_key, unflatten, flatten), layout=flatten(la_case.params)
    )
    np.testing.assert_allclose(
        np.sort(jnp.abs(torch_curv).sum(axis=-1))
        / np.sort(jnp.abs(jax_curv).sum(axis=-1)),
        1,
        atol=1e-2,
    )


def test_MC_fisher_against_curvlinops_BCE(trained_laplace_comparison_classification):
    la_case = trained_laplace_comparison_classification
    params = [p for p in la_case.torch_model.parameters() if p.requires_grad]

    torch_mv = EFLinearOperator(
        la_case.torch_model,
        torch.nn.BCEWithLogitsLoss(),
        params,
        [(la_case.X_train, la_case.y_train)],
    )
    torch_curv = torch_mv @ torch.eye(torch_mv.shape[0])

    train_batch = input_target_split_jax(next(iter(la_case.train_loader)))
    np.testing.assert_allclose(train_batch["target"], la_case.y_train)
    jax_mv = create_empirical_fisher_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="binary_cross_entropy",
        num_curv_samples=150,
        num_total_samples=1,
        vmap_over_data=False,
    )
    flatten, unflatten = create_pytree_flattener(la_case.params)
    jax_curv = to_dense(
        wrap_function(jax_mv, unflatten, flatten), layout=flatten(la_case.params)
    )
    np.testing.assert_allclose(
        la_case.torch_model(la_case.X_train).detach().numpy(),
        la_case.nnx_model_fn(train_batch["input"], la_case.params),
        atol=0.01,
    )

    np.testing.assert_allclose(
        np.sort(jnp.abs(torch_curv).sum(axis=-1))
        / np.sort(jnp.abs(jax_curv).sum(axis=-1)),
        1,
        atol=1e-2,
    )


def test_MC_convergence(trained_laplace_comparison):
    la_case = trained_laplace_comparison
    train_batch = input_target_split_jax(next(iter(la_case.train_loader)))

    GGN_mv = create_ggn_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="mse",
        num_curv_samples=150,
        num_total_samples=1,
        vmap_over_data=False,
    )
    flatten, unflatten = create_pytree_flattener(la_case.params)
    GGN_curv = to_dense(
        wrap_function(GGN_mv, unflatten, flatten), layout=flatten(la_case.params)
    )
    KEY = jax.random.key(742)
    MC_mv = create_MC_fisher_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="mse",
        mc_samples=10000,
        num_curv_samples=150,
        num_total_samples=1,
        vmap_over_data=False,
    )
    def MC_mv_with_key(vec):
        return MC_mv(vec, KEY)

    flatten, unflatten = create_pytree_flattener(la_case.params)
    MC_curv = to_dense(
        wrap_function(MC_mv_with_key, unflatten, flatten), layout=flatten(la_case.params)
    )

    np.testing.assert_allclose(GGN_curv, MC_curv, atol=0.03)
