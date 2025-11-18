"""FSP Laplace on Truncated Sine Regression.

This example demonstrates FSP (Function-Space Prior) Laplace approximation
for regression on a truncated sine function, showing how FSP captures
uncertainty in extrapolation regions.
"""


import operator

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from laplax.curv import KernelStructure, create_fsp_posterior
from laplax.util.objective import create_fsp_objective

jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Data Generation
# ==============================================================================


def generate_truncated_sine(key, n_samples=100, feature_dim=1, noise_std=0.01):
    """Generate truncated sine data.

    Generates data from a sine function on [-1, -0.5] U [0.5, 1],
    creating a gap in the middle for extrapolation testing.

    Args:
        key: JAX random key
        n_samples: Number of samples
        feature_dim: Dimensionality of input features
        noise_std: Standard deviation of observation noise

    Returns:
        Tuple of (X, y) arrays
    """
    key1, key2, key3 = jax.random.split(key, num=3)

    # Features: split between two regions
    X1 = jax.random.uniform(
        key1, minval=-1.0, maxval=-0.5, shape=(n_samples // 2, feature_dim)
    )
    X2 = jax.random.uniform(
        key2, minval=0.5, maxval=1.0, shape=(n_samples // 2, feature_dim)
    )
    X = jnp.concatenate([X1, X2], axis=0)

    # Targets: sine function + noise
    eps = noise_std * jax.random.normal(key3, shape=(n_samples,))
    y = jnp.sin(2 * jnp.pi * X.mean(axis=-1)) + eps

    return X, y.reshape(-1, 1)


# ==============================================================================
# Model Definition (Flax NNX)
# ==============================================================================


class MLP(nnx.Module):
    """Simple MLP for regression using Flax NNX."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs, dtype=jnp.float64):
        """Initialize MLP.

        Args:
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generator
            dtype: Data type for parameters
        """
        in_dim = 1  # 1D input

        # Build layers - use attributes to avoid tuple storage issue
        for i, hidden_dim in enumerate(hidden_dims):
            setattr(
                self,
                f"hidden_{i}",
                nnx.Linear(in_dim, hidden_dim, rngs=rngs, dtype=dtype),
            )
            in_dim = hidden_dim

        # Output layer (single value for regression)
        self.output_layer = nnx.Linear(in_dim, 1, rngs=rngs, dtype=dtype)
        self.n_hidden = len(hidden_dims)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through MLP.

        Args:
            x: Input array of shape (1,) or (batch, 1)

        Returns:
            Prediction output of shape (1,)
        """
        for i in range(self.n_hidden):
            x = jnp.tanh(getattr(self, f"hidden_{i}")(x))
        x = self.output_layer(x)
        return x  # Don't squeeze - keep shape (1,)


def split_model(model: MLP):
    """Split Flax NNX model into function and parameters.

    Returns:
        tuple[Callable, nnx.Params]: Model function and parameters.
    """
    graphdef, params = nnx.split(model, nnx.Param)

    def model_fn(x, params):
        """Model function that takes input and parameters.

        Returns:
            jax.Array: Model predictions for input ``x``.
        """
        model_copy = nnx.merge(graphdef, params)
        return model_copy(x)

    return model_fn, params


# ==============================================================================
# Training
# ==============================================================================


class NoiseScale(nnx.Module):
    """Trainable noise scale parameter."""

    def __init__(self, initial_value: float = 0.1):
        # Use log scale for better optimization
        self.log_scale = nnx.Param(jnp.log(jnp.array(initial_value)))

    @property
    def scale(self):
        return jnp.exp(self.log_scale.value)


def train_mlp_fsp(
    model,
    noise_scale,
    x_train,
    y_train,
    x_context,
    prior_mean,
    prior_cov_kernel,
    num_epochs=1000,
    learning_rate=0.01,
):
    """Train MLP with FSP objective using create_fsp_objective.

    Returns:
        tuple[MLP, NoiseScale]: Trained model and noise scale.
    """
    # Split model for laplax
    model_fn, params = split_model(model)

    # Create FSP objective using laplax.util.objective
    dataset_size = x_train.shape[0]
    fsp_objective = create_fsp_objective(
        model_fn=model_fn,
        dataset_size=dataset_size,
        prior_mean=prior_mean,
        prior_cov_kernel=prior_cov_kernel,
    )

    # Prepare data in expected format
    data = {"input": x_train, "target": y_train}
    context_points = {"context": x_context, "grid": x_context}

    # Create optimizers for params and noise
    optimizer = optax.adam(learning_rate)
    params_opt_state = optimizer.init(params)
    noise_opt_state = optimizer.init(noise_scale.log_scale.value)

    @jax.jit
    def train_step(params, noise_log_scale, params_opt_state, noise_opt_state):
        """Single training step - JIT compiled for speed.

        Returns:
            tuple: Updated parameters, noise log-scale, optimizer states,
            and scalar loss.
        """

        def loss_fn(params, noise_log_scale):
            noise_val = jnp.exp(noise_log_scale)
            return fsp_objective(data, context_points, params, scale=noise_val)

        # Compute gradients w.r.t. both params and noise
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            params, noise_log_scale
        )
        params_grads, noise_grads = grads

        # Update params
        params_updates, params_opt_state = optimizer.update(
            params_grads, params_opt_state
        )
        params = optax.apply_updates(params, params_updates)

        # Update noise
        noise_updates, noise_opt_state = optimizer.update(noise_grads, noise_opt_state)
        noise_log_scale = optax.apply_updates(noise_log_scale, noise_updates)

        return params, noise_log_scale, params_opt_state, noise_opt_state, loss

    # Training loop
    noise_log_scale = noise_scale.log_scale.value
    for epoch in range(num_epochs):
        params, noise_log_scale, params_opt_state, noise_opt_state, _loss = train_step(
            params, noise_log_scale, params_opt_state, noise_opt_state
        )

        if (epoch + 1) % 200 == 0:
            jnp.exp(noise_log_scale)

    # Update model and noise_scale with trained values
    graphdef, _ = nnx.split(model, nnx.Param)
    model = nnx.merge(graphdef, params)
    noise_scale.log_scale.value = noise_log_scale

    return model, noise_scale


# ==============================================================================
# FSP Posterior with RBF Kernel
# ==============================================================================
def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    """Radial Basis Function (RBF) kernel (squared exponential).

    Returns:
        jax.Array: Kernel matrix evaluated at ``x1`` and ``x2``.
    """
    sqdist = (
        jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    )
    return variance * jnp.exp(-0.5 / lengthscale**2 * sqdist)


def periodic_kernel(x1, x2, lengthscale=1.0, variance=1.0, period=2.0):
    """Periodic kernel function for periodic data like sine waves.

    k(x1, x2) = variance * exp(-||x1 - x2||^2 / (2 * lengthscale^2)).

    Returns:
        jax.Array: Periodic kernel matrix evaluated at ``x1`` and ``x2``.
    """
    dist = jnp.sqrt(jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1))
    sin_term = jnp.sin(jnp.pi * dist / period)
    return variance * jnp.exp(-2 * sin_term**2 / lengthscale**2)


def create_kernel_matrix(x_context, lengthscale=1.0, variance=1.0):
    """Create RBF kernel matrix.

    Returns:
        jax.Array: Kernel matrix evaluated at ``x_context``.
    """
    K = periodic_kernel(x_context, x_context, lengthscale, variance, period=1.0)
    K = K + 1e-6 * jnp.eye(K.shape[0])
    return K


# ==============================================================================
# Main Example
# ==============================================================================


def main():
    """Run FSP Laplace on truncated sine regression."""
    # Generate training data
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    X_train, y_train = generate_truncated_sine(
        data_key, n_samples=100, feature_dim=1, noise_std=0.1
    )

    # Set up context points and kernel for FSP training
    x_context = jnp.linspace(-1.5, 1.5, 200).reshape(-1, 1)

    lengthscale = 1.0
    variance = 1.0
    period = 1.00

    def prior_cov_kernel(x1, x2):
        """Prior covariance kernel for FSP objective.

        Returns:
            jax.Array: Prior covariance matrix for inputs ``x1`` and ``x2``.
        """
        K = periodic_kernel(
            x1, x2, lengthscale=lengthscale, variance=variance, period=period
        )
        K = K + 1e-6 * jnp.eye(K.shape[0])
        return K

    # Train model with FSP objective
    key, model_key = jax.random.split(key)
    model = MLP(hidden_dims=[50, 50], rngs=nnx.Rngs(int(model_key[0])))
    noise_scale = NoiseScale(initial_value=0.1)

    # Prior mean for FSP (zero for now)
    prior_mean = jnp.zeros(x_context.shape[0])

    model, noise_scale = train_mlp_fsp(
        model,
        noise_scale,
        X_train,
        y_train,
        x_context,
        prior_mean,
        prior_cov_kernel,
        num_epochs=100,
        learning_rate=0.01,
    )

    # Create FSP posterior

    def kernel_fn(v):
        """Kernel matrix-vector product.

        Returns:
            jax.Array: Product of the kernel matrix with vector ``v``.
        """
        K = create_kernel_matrix(x_context, lengthscale=lengthscale, variance=variance)
        return K @ v

    # Compute prior variance
    prior_cov = create_kernel_matrix(
        x_context, lengthscale=lengthscale, variance=variance
    )
    prior_variance = jnp.diag(prior_cov)

    # Split model for laplax
    model_fn, trained_params = split_model(model)

    # Create FSP posterior
    posterior = create_fsp_posterior(
        model_fn=model_fn,
        params=trained_params,
        x_context=x_context,
        kernel_structure=KernelStructure.NONE,
        kernel=kernel_fn,
        prior_variance=prior_variance,
        n_chunks=2,
        max_iter=50,
        # Align GGN scaling with Gaussian NLL using the learned noise.
        regression_noise_scale=float(noise_scale.scale),
    )

    # Make predictions with uncertainty

    # Create test grid including extrapolation region
    x_test = jnp.linspace(-1.5, 1.5, 300).reshape(-1, 1)

    # Mean predictions
    jax.vmap(model)(x_test)

    # Sample from posterior
    key, _sample_key = jax.random.split(key)
    n_samples = 50
    samples = []

    for _i in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(operator.add, trained_params, delta_params)

        sample_preds = jax.vmap(
            lambda x, p=sample_params: model_fn(x, p)
        )(x_test)
        samples.append(sample_preds)

    samples = jnp.stack(samples)
    pred_mean = jnp.mean(samples, axis=0).squeeze()
    pred_std = jnp.std(samples, axis=0).squeeze()

    # Visualize results
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True function for reference
    x_true = jnp.linspace(-1.5, 1.5, 1000)
    y_true = jnp.sin(2 * jnp.pi * x_true)

    # Plot 1: Predictions with uncertainty
    ax = axes[0]
    ax.plot(x_true, y_true, "k--", alpha=0.3, label="True function", linewidth=2)
    ax.scatter(X_train, y_train, c="blue", alpha=0.6, s=30, label="Training data")
    ax.plot(x_test, pred_mean, "r-", label="Posterior mean", linewidth=2)
    ax.fill_between(
        x_test.squeeze(),
        pred_mean - 2 * pred_std,
        pred_mean + 2 * pred_std,
        alpha=0.3,
        color="red",
        label="±2 sigma (95% CI)",
    )

    # Highlight extrapolation regions
    ax.axvspan(-1.5, -1.0, alpha=0.1, color="gray", label="Extrapolation")
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="gray")
    ax.axvspan(1.0, 1.5, alpha=0.1, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"FSP Laplace Predictions (rank={posterior.rank})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Uncertainty quantification
    ax = axes[1]
    ax.plot(x_test, pred_std, "b-", linewidth=2, label="Predictive std")
    ax.scatter(X_train, jnp.zeros_like(y_train), c="blue", alpha=0.3, s=20)

    # Highlight extrapolation regions
    ax.axvspan(-1.5, -1.0, alpha=0.1, color="gray", label="Extrapolation")
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="gray")
    ax.axvspan(1.0, 1.5, alpha=0.1, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("Predictive standard deviation")
    ax.set_title("Uncertainty Quantification", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig("fsp_sin_regression.png", dpi=150, bbox_inches="tight")

    # Summary


if __name__ == "__main__":
    main()
