import jax.numpy as jnp
import numpy as np
import pytest
import torch
from curvlinops import GGNLinearOperator

from laplax.curv.ggn import create_ggn_mv
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import to_dense
from tests.conftest import input_target_split_jax


@pytest.mark.parametrize("la_method", ["full", "diagonal"])
def test_compare_implementations_against_laplace_redux(
    la_method, trained_laplace_comparison
):
    comparison = trained_laplace_comparison
    comparison.la_method = la_method  # override for this test

    # Now set up Laplace with chosen la_method
    comparison.setup_laplace_torch()
    torch_mu_nonlin, torch_std_nonlin = comparison.run_laplace_torch_nonlin()
    torch_mu_lin, torch_std_lin = comparison.run_laplace_torch_lin()

    comparison.setup_laplax()
    laplax_mu_nonlin, laplax_std_nonlin = comparison.run_laplax_nonlin()
    laplax_mu_lin, laplax_std_lin = comparison.run_laplax_lin()

    # Compare results
    mean_diff_nonlin = np.abs(torch_mu_nonlin - laplax_mu_nonlin).mean()
    std_diff_nonlin = np.abs(torch_std_nonlin - laplax_std_nonlin).mean()

    mean_diff_lin = np.abs(torch_mu_lin - laplax_mu_lin).mean()
    std_diff_lin = np.abs(torch_std_lin - laplax_std_lin).mean()

    np.testing.assert_allclose(mean_diff_nonlin, 0, atol=1)
    np.testing.assert_allclose(std_diff_nonlin, 0, atol=1)
    np.testing.assert_allclose(mean_diff_lin, 0, atol=1e-4)
    np.testing.assert_allclose(std_diff_lin, 0, atol=1.5e-3)


def test_ggn_against_curvlinops(trained_laplace_comparison):
    la_case = trained_laplace_comparison

    # Torch GGN (Curvlinops)
    params = [p for p in la_case.torch_model.parameters() if p.requires_grad]
    GGN = GGNLinearOperator(
        la_case.torch_model,
        torch.nn.MSELoss(),
        params,
        [(la_case.X_train, la_case.y_train)],
    )
    torch_ggn = GGN @ torch.eye(GGN.shape[0])

    # Laplax (JAX) GGN
    train_batch = input_target_split_jax(next(iter(la_case.train_loader)))
    ggn_mv = create_ggn_mv(
        la_case.nnx_model_fn,
        la_case.params,
        train_batch,
        loss_fn="mse",
        num_curv_samples=150,
        num_total_samples=1,
    )
    flatten, unflatten = create_pytree_flattener(la_case.params)
    jax_ggn = to_dense(
        wrap_function(ggn_mv, unflatten, flatten), layout=flatten(la_case.params)
    )

    np.testing.assert_allclose(
        np.sort(jnp.abs(torch_ggn).sum(axis=-1))
        / np.sort(jnp.abs(jax_ggn).sum(axis=-1)),
        1,
        atol=1e-2,
    )
