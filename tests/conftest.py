# /tests/conftest.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.utils.data as data_utils
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


@pytest.fixture(autouse=True)
def _disable_x64():
    jax.config.update("jax_enable_x64", False)


def get_sinusoid_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test


def get_sinusoid_classification_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise > 0
    y_train = y_train.type(torch.float32)
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
        loss="mse",
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
        if loss == "mse":
            (
                self.X_train,
                self.y_train,
                self.train_loader,
                self.X_test,
            ) = get_sinusoid_example(sigma_noise=self.sigma_noise)
        else:
            (
                self.X_train,
                self.y_train,
                self.train_loader,
                self.X_test,
            ) = get_sinusoid_classification_example(sigma_noise=self.sigma_noise)
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


@pytest.fixture(scope="module")
def trained_laplace_comparison_classification():
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
        loss="bce",
    )
    comparison.train_model()
    return comparison
