"""FSP vs Standard Laplace on Two Moons (Binary Classification).

This script demonstrates FSP (Function-Space Prior) Laplace approximation for
binary classification on the two-moons dataset. It trains a small MLP using the
FSP objective (cross-entropy + RKHS regularization) with an RBF kernel on context
points, constructs an FSP posterior, and compares predictive uncertainty against
a standard Laplace (low-rank/Lanczos) posterior.

It saves a visualization to `fsp_two_moons_classification.png`.
"""

import operator

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from sklearn.datasets import make_moons

from laplax.api import GGN
from laplax.curv import (
    KernelStructure,
    create_fsp_posterior,
    estimate_curvature,
    set_posterior_fn,
)
from laplax.enums import CurvApprox, LossFn
from laplax.util.flatten import create_pytree_flattener
from laplax.util.objective import create_loss_reg

# ==============================================================================
# Model (Flax NNX)
# ==============================================================================


class MLP(nnx.Module):
    """Simple MLP for binary classification."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        in_dim = 2  # Two moons features

        # Build layers - use attributes to avoid tuple storage issue
        for i, hidden_dim in enumerate(hidden_dims):
            setattr(self, f"hidden_{i}", nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        self.output_layer = nnx.Linear(in_dim, 1, rngs=rngs)
        self.n_hidden = len(hidden_dims)

    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.n_hidden):
            x = jnp.tanh(getattr(self, f"hidden_{i}")(x))
        x = self.output_layer(x)
        return x  # Keep shape (1,) for FSP compatibility


def split_model(model: MLP):
    """Split Flax NNX model into function and parameters.

    Returns:
        tuple[Callable, nnx.Params]: Model function and parameters.
    """
    graphdef, params = nnx.split(model, nnx.Param)

    def model_fn(x, params):
        model_copy = nnx.merge(graphdef, params)
        return model_copy(x)

    return model_fn, params


# ==============================================================================
# Training
# ==============================================================================


def binary_cross_entropy_loss(model, x_batch, y_batch):
    """Binary cross-entropy loss.

    Returns:
        jax.Array: Scalar loss value.
    """
    logits = jax.vmap(model)(x_batch).squeeze(-1)  # Squeeze to match y_batch shape
    return -jnp.mean(
        jax.nn.log_sigmoid(logits) * y_batch
        + jax.nn.log_sigmoid(-logits) * (1 - y_batch)
    )


def train_mlp_fsp(
    model,
    x_train,
    y_train,
    x_context,
    prior_mean,
    prior_cov_kernel,
    num_epochs=100,
    learning_rate=0.1,
    reg_weight=1.0,
):
    """Train MLP with FSP objective using laplax.util.objective.

    Returns:
        MLP: Trained model.
    """
    # Split model for laplax
    model_fn, params = split_model(model)

    # Create regularization loss using laplax.util.objective
    loss_reg = create_loss_reg(
        model_fn=model_fn,
        prior_mean=prior_mean,
        prior_cov_kernel=prior_cov_kernel,
        has_batch_dim=True,  # Use dict format for context points
    )

    # Prepare context points in expected format
    context_points = {"context": x_context, "grid": x_context}
    dataset_size = x_train.shape[0]

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        """Single training step - JIT compiled for speed.

        Returns:
            tuple: Updated parameters, optimizer state, and scalar loss.
        """

        def loss_fn(params):
            # Cross-entropy term
            logits = jax.vmap(model_fn, in_axes=(0, None))(x_train, params).squeeze(-1)
            ce_loss = -jnp.mean(
                jax.nn.log_sigmoid(logits) * y_train
                + jax.nn.log_sigmoid(-logits) * (1 - y_train)
            )
            ce_loss = ce_loss * dataset_size

            # Regularization term using laplax utility
            reg = loss_reg(context_points, params)

            return ce_loss + reg_weight * reg

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update params
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Training loop
    for epoch in range(num_epochs):
        params, opt_state, _loss = train_step(params, opt_state)

        if (epoch + 1) % 20 == 0:
            pass

    # Update model with trained values
    graphdef, _ = nnx.split(model, nnx.Param)
    model = nnx.merge(graphdef, params)

    return model


# ==============================================================================
# Kernels (RBF) for FSP
# ==============================================================================


def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-sq_dist / (2 * lengthscale**2))


def create_kernel_matrix(x_context, lengthscale=1.0, variance=1.0):
    K = rbf_kernel(x_context, x_context, lengthscale, variance)
    return K + 1e-6 * jnp.eye(K.shape[0])


# ==============================================================================
# Main
# ==============================================================================


def main():

    # ------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------
    X, y = make_moons(n_samples=300, noise=0.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # ------------------------------------------------------------------
    # 2) Set up FSP training with RBF kernel
    # ------------------------------------------------------------------

    # Select context points for FSP training
    n_context = 50
    rng = np.random.default_rng(42)
    context_indices = rng.choice(len(X), size=n_context, replace=False)
    x_context = X_jax[context_indices]

    # RBF kernel hyperparameters
    lengthscale = 0.5
    variance = 0.01

    def prior_cov_kernel(x1, x2):
        """Prior covariance kernel for FSP objective.

        Returns:
            jax.Array: Prior covariance matrix for inputs ``x1`` and ``x2``.
        """
        K = rbf_kernel(x1, x2, lengthscale=lengthscale, variance=variance)
        # Add jitter for numerical stability
        K = K + 1e-5 * jnp.eye(K.shape[0])
        return K

    # Prior mean for FSP (zero for now)
    prior_mean = jnp.zeros(x_context.shape[0])

    # ------------------------------------------------------------------
    # 3) Train model with FSP objective
    # ------------------------------------------------------------------
    model = MLP(hidden_dims=[32, 32], rngs=nnx.Rngs(0))
    model = train_mlp_fsp(
        model,
        X_jax,
        y_jax,
        x_context,
        prior_mean,
        prior_cov_kernel,
        num_epochs=100,
        learning_rate=0.1,
        reg_weight=0.01,
    )

    logits = jax.vmap(model)(X_jax).squeeze(-1)
    predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    jnp.mean(predictions == y_jax)

    # ------------------------------------------------------------------
    # 4) FSP posterior
    # ------------------------------------------------------------------
    # Note: Context points and kernel already defined for training

    def kernel_fn(v):
        K = create_kernel_matrix(x_context, lengthscale=lengthscale, variance=variance)
        return K @ v

    # Split for Laplax
    model_fn, trained_params = split_model(model)

    # Wrap model to use two-logit trick for proper CE Hessian in FSP
    def model_fn_two_logit(x, params):
        """Model function with two-logit output for cross-entropy.

        Returns:
            jax.Array: Two-logit outputs for binary classification.
        """
        logit = model_fn(x, params).squeeze()
        return jnp.stack([jnp.zeros_like(logit), logit], axis=-1)

    # Compute prior variance for multi-output model (n_context, n_outputs)
    prior_cov = create_kernel_matrix(
        x_context, lengthscale=lengthscale, variance=variance
    )
    prior_var_single = jnp.diag(prior_cov)  # Shape: (n_context,)
    # Tile for both outputs (same prior for each output dimension)
    prior_variance = jnp.tile(
        prior_var_single[:, None], (1, 2)
    )  # Shape: (n_context, 2)

    posterior = create_fsp_posterior(
        model_fn=model_fn_two_logit,
        params=trained_params,
        x_context=x_context,
        kernel_structure=KernelStructure.NONE,
        kernel=kernel_fn,  # Shared kernel for both outputs
        prior_variance=prior_variance,  # Shape: (n_context, 2)
        independent_outputs=True,  # Process outputs independently
        n_chunks=2,
        max_iter=50,
        is_classification=True,
    )

    # ------------------------------------------------------------------
    # 5) Standard Laplace (Lanczos low-rank curvature)
    # ------------------------------------------------------------------
    flatten_fn, unflatten_fn = create_pytree_flattener(trained_params)

    def model_fn_flat(inp, flat_params):
        params_pytree = unflatten_fn(flat_params)
        logit = model_fn(inp, params_pytree).squeeze()  # Squeeze to scalar
        return jnp.stack([jnp.zeros_like(logit), logit], axis=-1)

    data = {"input": X_jax, "target": y_jax.astype(jnp.int32)}
    ggn_mv = GGN(
        model_fn_flat,
        flatten_fn(trained_params),
        data,
        loss_fn=LossFn.CROSS_ENTROPY,
        vmap_over_data=True,
    )

    max_rank = 50
    curv_estimate = estimate_curvature(
        curv_type=CurvApprox.LANCZOS,
        mv=ggn_mv,
        layout=flatten_fn(trained_params).shape[0],
        rank=max_rank,
        key=jax.random.key(42),
        has_batch=True,
    )

    posterior_fn = set_posterior_fn(
        curv_type=CurvApprox.LANCZOS,
        curv_estimate=curv_estimate,
        layout=flatten_fn(trained_params).shape[0],
    )
    standard_posterior = posterior_fn({"prior_prec": 1.0})

    # ------------------------------------------------------------------
    # 6) Predictions & Sampling
    # ------------------------------------------------------------------
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    # Mean prediction
    mean_logits = jax.vmap(model)(grid_jax).squeeze(-1)
    mean_probs = jax.nn.sigmoid(mean_logits)

    # FSP sampling
    key = jax.random.PRNGKey(42)
    n_samples = 20
    fsp_samples = []
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(operator.add, trained_params, delta_params)
        # Use two-logit model and extract the relevant logit (index 1)
        sample_two_logits = jax.vmap(
            lambda x, p=sample_params: model_fn_two_logit(x, p)
        )(grid_jax)
        sample_logits = sample_two_logits[:, 1]  # Extract second logit
        fsp_samples.append(jax.nn.sigmoid(sample_logits))
    fsp_samples = jnp.stack(fsp_samples)
    fsp_std = jnp.std(fsp_samples, axis=0)

    # Standard Laplace sampling
    key = jax.random.PRNGKey(42)
    standard_samples = []
    param_size = int(flatten_fn(trained_params).shape[0])
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (param_size,))
        delta_params_flat = standard_posterior.scale_mv(standard_posterior.state)(z)
        delta_params = unflatten_fn(delta_params_flat)
        sample_params = jax.tree.map(operator.add, trained_params, delta_params)
        sample_logits = jax.vmap(
            lambda x, p=sample_params: model_fn(x, p)
        )(grid_jax)
        standard_samples.append(jax.nn.sigmoid(sample_logits))
    standard_samples = jnp.stack(standard_samples)
    standard_std = jnp.std(standard_samples, axis=0)

    # ------------------------------------------------------------------
    # 7) Visualization
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Mean predictions (shared)
    ax = axes[0, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.scatter(
        x_context[:, 0],
        x_context[:, 1],
        c="green",
        marker="x",
        s=100,
        label="Context points",
        linewidths=2,
    )
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    plt.colorbar(contour, ax=ax)

    # Row 1: FSP uncertainty and confidence
    ax = axes[0, 1]
    contour = ax.contourf(
        xx, yy, fsp_std.reshape(xx.shape), levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(
        f"FSP Laplace Uncertainty (rank={posterior.rank})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    ax = axes[0, 2]
    fsp_confidence = 1 - 2 * fsp_std
    contour = ax.contourf(
        xx, yy, fsp_confidence.reshape(xx.shape), levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("FSP Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    # Row 2: Standard Laplace uncertainty and confidence
    ax = axes[1, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax)

    ax = axes[1, 1]
    contour = ax.contourf(
        xx, yy, standard_std.reshape(xx.shape), levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(
        f"Standard Laplace Uncertainty (rank={curv_estimate.U.shape[1]})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    ax = axes[1, 2]
    standard_confidence = 1 - 2 * standard_std
    contour = ax.contourf(
        xx,
        yy,
        standard_confidence.reshape(xx.shape),
        levels=20,
        cmap="plasma",
        alpha=0.8,
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Standard Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    plt.tight_layout()
    plt.savefig("fsp_two_moons_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) Summary
    # ------------------------------------------------------------------
    fsp_std - standard_std


if __name__ == "__main__":
    main()
