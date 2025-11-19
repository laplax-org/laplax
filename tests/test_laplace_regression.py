import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.utils.data as data_utils
from curvlinops import GGNLinearOperator
from flax import nnx
from laplace import Laplace

from laplax.curv.cov import create_posterior_fn
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_var,
    lin_setup,
    nonlin_pred_mean,
    nonlin_pred_var,
    nonlin_setup,
    set_lin_pushforward,
    set_nonlin_pushforward,
)
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import to_dense


def get_sinusoid_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test


def input_target_split_jax(batch):
    return {
        "input": jnp.asarray(batch[0].numpy()),
        "target": jnp.asarray(batch[1].numpy()),
    }


class LaplaceComparison:
    def __init__(
        self,
        n_epochs=1000,
        seed=711,
        sigma_noise=0.3,
        lr=1e-2,
        *,
        la_method="full",
        debug=False,
    ):
        self.n_epochs = n_epochs
        self.seed = seed
        self.sigma_noise = sigma_noise
        self.lr = lr
        self.debug = debug
        self.num_samples = 1000
        self.la_method = la_method

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)

        # Create data
        (
            self.X_train,
            self.y_train,
            self.train_loader,
            self.X_test,
        ) = get_sinusoid_example(sigma_noise=self.sigma_noise)

        # Build the torch model
        self.torch_model = self._build_torch_model()

        # Convert torch model to NNX model (weights still untrained if we haven't called
        # train)
        self.nnx_model = self._convert_torch_to_nnx()
        self.graph_def, self.params = nnx.split(self.nnx_model)
        self.nnx_model_fn = lambda input, params: nnx.call((self.graph_def, params))(
            input
        )[0]

        # Initialize Laplace placeholders
        self.la_torch = None
        self.ggn_mv = None
        self.get_posterior = None

    def train_model(self):
        """Explicitly train the torch model."""
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr)
        for _ in range(self.n_epochs):
            for X, y in self.train_loader:
                optimizer.zero_grad()
                loss = criterion(self.torch_model(X), y)
                loss.backward()
                optimizer.step()

        # After training, update our NNX model so that it reflects the final
        # trained parameters.
        self._update_nnx_from_torch()

        # Quick check that the models match
        np.testing.assert_allclose(
            self.nnx_model(self.X_train),
            self.torch_model(self.X_train).cpu().detach().numpy(),
            atol=1e-2,
        )

    def _build_torch_model(self):
        """Just build the Torch model (don't train).

        Returns:
            The Torch model.
        """
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1),
        )

    def _update_nnx_from_torch(self):
        """Copy Torch model's trained weights over to the NNX model."""
        self.nnx_model.linear1.kernel.value = jnp.asarray(
            self.torch_model[0].weight.detach().numpy()
        ).T
        self.nnx_model.linear1.bias.value = jnp.asarray(
            self.torch_model[0].bias.detach().numpy()
        )
        self.nnx_model.linear2.kernel.value = jnp.asarray(
            self.torch_model[2].weight.detach().numpy()
        ).T
        self.nnx_model.linear2.bias.value = jnp.asarray(
            self.torch_model[2].bias.detach().numpy()
        )
        # Update our internal reference to params
        self.graph_def, self.params = nnx.split(self.nnx_model)
        self.nnx_model_fn = lambda input, params: nnx.call((self.graph_def, params))(
            input
        )[0]

    @property
    def X_test_jax(self):
        return jnp.asarray(self.X_test.numpy())

    def setup_laplace_torch(self):
        methods = {
            "full": "full",
            "diagonal": "diag",
            "low_rank": "lowrank",
        }

        la = Laplace(
            self.torch_model,
            "regression",
            subset_of_weights="all",
            hessian_structure=methods[self.la_method],
        )
        la.fit(self.train_loader)
        self.la_torch = la
        if self.la_method == "full":
            self.la_torch._compute_scale()  # Precompute scale.  # noqa: SLF001

    def run_laplace_torch_lin(self):
        if self.la_torch is None:
            msg = "You must call `setup_laplace_torch()` first."
            raise RuntimeError(msg)

        f_mu, f_var = self.la_torch(self.X_test)
        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + self.la_torch.sigma_noise.item() ** 2)

        return f_mu, pred_std

    def run_laplace_torch_nonlin(self):
        if self.la_torch is None:
            msg = "You must call `setup_laplace_torch()` first."
            raise RuntimeError(msg)

        if self.la_method == "full":
            self.la_torch._posterior_scale = self.la_torch._posterior_scale.T  # noqa: SLF001
        # TODO(any): Check if this is a bug in laplace-redux.

        f_mu, f_var = self.la_torch(
            self.X_test,
            pred_type="nn",
            link_approx="mc",
            n_samples=self.num_samples,
        )

        if self.la_method == "full":
            self.la_torch._posterior_scale = self.la_torch._posterior_scale.T  # noqa: SLF001
        # TODO(any): Check if this is a bug in laplace-redux.

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + self.la_torch.sigma_noise.item() ** 2)

        return f_mu, pred_std

    def _convert_torch_to_nnx(self):
        class Model(nnx.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, rngs):
                self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
                self.tanh = nnx.tanh
                self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

            def __call__(self, x):
                x = self.linear1(x)
                x = self.tanh(x)
                x = self.linear2(x)
                return x

        rngs = nnx.Rngs(0)
        nnx_model = Model(1, 50, 1, rngs)

        # Copy weights
        nnx_model.linear1.kernel.value = jnp.asarray(
            self.torch_model[0].weight.detach().numpy()
        ).T
        nnx_model.linear1.bias.value = jnp.asarray(
            self.torch_model[0].bias.detach().numpy()
        )
        nnx_model.linear2.kernel.value = jnp.asarray(
            self.torch_model[2].weight.detach().numpy()
        ).T
        nnx_model.linear2.bias.value = jnp.asarray(
            self.torch_model[2].bias.detach().numpy()
        )

        return nnx_model

    def setup_laplax(self):
        train_batch = input_target_split_jax(next(iter(self.train_loader)))

        ggn_mv = create_ggn_mv(
            self.nnx_model_fn,
            self.params,
            train_batch,
            loss_fn="mse",
            num_curv_samples=150,
            num_total_samples=75,
        )
        self.ggn_mv = ggn_mv

        # Setup posterior
        self.get_posterior = create_posterior_fn(
            self.la_method,
            mv=self.ggn_mv,
            layout=self.params,
        )

    def run_laplax_nonlin(self):
        # Create pushforward
        pushforward = set_nonlin_pushforward(
            model_fn=self.nnx_model_fn,
            mean_params=self.params,
            key=jax.random.key(0),
            posterior_fn=self.get_posterior,
            prior_arguments={"prior_prec": 1},
            pushforward_fns=[
                nonlin_setup,
                nonlin_pred_mean,
                nonlin_pred_var,
            ],
            num_samples=self.num_samples,
        )

        results = jax.vmap(pushforward)(self.X_test_jax)
        f_mu = results["pred_mean"].reshape(-1)
        pred_std = jnp.sqrt(results["pred_var"] + 1).reshape(-1)

        return np.array(f_mu), np.array(pred_std)

    def run_laplax_lin(self):
        # Create pushforward
        pushforward = set_lin_pushforward(
            model_fn=self.nnx_model_fn,
            mean_params=self.params,
            key=jax.random.key(0),
            posterior_fn=self.get_posterior,
            prior_arguments={"prior_prec": 1},
            pushforward_fns=[
                lin_setup,
                lin_pred_mean,
                lin_pred_var,
            ],
        )

        results = jax.vmap(pushforward)(self.X_test_jax)
        f_mu = results["pred_mean"].reshape(-1)
        pred_std = jnp.sqrt(results["pred_var"] + 1).reshape(-1)

        return np.array(f_mu), np.array(pred_std)


@pytest.fixture(scope="module")
def trained_laplace_comparison():
    """Build a LaplaceComparison instance, train it once, and return it.

    Returns:
        The LaplaceComparison object.
    """
    comparison = LaplaceComparison(
        n_epochs=1000,
        seed=711,
        sigma_noise=0.3,
        lr=1e-2,
        la_method=None,  # We'll override la_method later if desired
        debug=False,
    )
    comparison.train_model()
    return comparison


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
