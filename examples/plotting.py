"""Plotting utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles


def plot_regression_with_uncertainty(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    X_pred=None,
    y_pred=None,
    y_std=None,
    title=None,
):
    """Plot regression data with optional prediction and uncertainty.

    Args:
        X_train: Training input data (shape: n_samples, 1)
        y_train: Training target data (shape: n_samples, 1)
        X_test: Test input data (shape: n_samples, 1), optional
        y_test: Test target data (shape: n_samples, 1), optional
        X_pred: Prediction input data (shape: n_samples, 1), if None uses X_test
        y_pred: Prediction mean (shape: n_samples, 1), optional
        y_std: Prediction standard deviation (shape: n_samples, 1), optional
        title: Plot title, optional

    Returns:
        The figure.
    """
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays if they are JAX arrays
    if hasattr(X_train, "device_buffer"):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        if X_pred is not None and y_pred is not None:
            X_pred = np.array(X_pred)
            y_pred = np.array(y_pred)
            if y_std is not None:
                y_std = np.array(y_std)

    X_train_min = X_train.min()
    X_train_max = X_train.max()

    if X_test is not None:
        X_test_min = X_test.min()
        X_test_max = X_test.max()
    else:
        X_test_min = X_train_min
        X_test_max = X_train_max

    # Plot training data
    plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")

    # Plot test data if provided
    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test data")

    # Plot prediction with uncertainty if provided
    if y_pred is not None:
        # If X_pred is not provided but X_test is, use X_test for predictions
        X_plot = X_pred if X_pred is not None else X_test

        # Only proceed if we have points to plot predictions for
        if X_plot is not None:
            # Sort X for proper line plotting
            sort_idx = np.argsort(X_plot.flatten())
            X_plot_sorted = X_plot[sort_idx]
            y_pred_sorted = y_pred[sort_idx]

            plt.plot(X_plot_sorted, y_pred_sorted, color="red", label="Prediction")

            # Plot uncertainty if provided
            if y_std is not None:
                y_std_sorted = y_std[sort_idx]
                plt.fill_between(
                    X_plot_sorted.flatten(),
                    (y_pred_sorted - 2 * y_std_sorted).flatten(),
                    (y_pred_sorted + 2 * y_std_sorted).flatten(),
                    color="red",
                    alpha=0.2,
                    label="95% confidence interval",
                )

    # Plot true function
    min_point = min(X_train_min, X_test_min)
    max_point = max(X_train_max, X_test_max)
    x_true = np.linspace(min_point, max_point, 1000).reshape(-1, 1)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, color="black", linestyle="--", label="True function")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    else:
        plt.title("Regression with Uncertainty")

    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def plot_posterior_samples(
    X, posterior_samples, X_train=None, y_train=None, title=None
):
    """Plot samples from the posterior distribution.

    Args:
        X: Input data for posterior samples (shape: n_samples, 1)
        posterior_samples: Samples from posterior
            (shape: n_posterior_samples, n_samples, 1)
        X_train: Training input data (shape: n_train_samples, 1)
        y_train: Training target data (shape: n_train_samples, 1)
        title: Plot title

    Returns:
        The figure.
    """
    plt.figure(figsize=(10, 6))

    # Sort X for proper line plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]

    # Plot posterior samples
    for i in range(min(10, posterior_samples.shape[0])):  # Plot up to 10 samples
        sample_sorted = posterior_samples[i][sort_idx]
        plt.plot(X_sorted, sample_sorted, color="red", alpha=0.3)

    # Plot training data if provided
    if X_train is not None and y_train is not None:
        plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")

    # Plot true function
    x_true = np.linspace(0, 8, 1000).reshape(-1, 1)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, color="black", linestyle="--", label="True function")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    else:
        plt.title("Posterior Samples")

    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def plot_predictive_distribution(
    X_train, y_train, X_test, y_test, X_pred, y_pred_mean, y_pred_std, title=None
):
    """Plot the predictive distribution for regression.

    Args:
        X_train: Training input data (shape: n_train_samples, 1)
        y_train: Training target data (shape: n_train_samples, 1)
        X_test: Test input data (shape: n_test_samples, 1)
        y_test: Test target data (shape: n_test_samples, 1)
        X_pred: Prediction input points (shape: n_pred_samples, 1)
        y_pred_mean: Predictive mean (shape: n_pred_samples, 1)
        y_pred_std: Predictive standard deviation (shape: n_pred_samples, 1)
        title: Plot title

    Returns:
        The figure.
    """
    # Convert to numpy arrays if they are JAX arrays
    if hasattr(X_train, "device_buffer"):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_pred = np.array(X_pred)
        y_pred_mean = np.array(y_pred_mean)
        y_pred_std = np.array(y_pred_std)

    # Sort X_pred for proper line plotting
    sort_idx = np.argsort(X_pred.flatten())
    X_pred_sorted = X_pred[sort_idx]
    y_pred_mean_sorted = y_pred_mean[sort_idx]
    y_pred_std_sorted = y_pred_std[sort_idx]

    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")

    # Plot test data
    plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test data")

    # Plot predictive mean
    plt.plot(X_pred_sorted, y_pred_mean_sorted, color="red", label="Predictive mean")

    # Plot uncertainty regions
    plt.fill_between(
        X_pred_sorted.flatten(),
        (y_pred_mean_sorted - 1 * y_pred_std_sorted).flatten(),
        (y_pred_mean_sorted + 1 * y_pred_std_sorted).flatten(),
        color="red",
        alpha=0.3,
        label="68% confidence interval",
    )

    plt.fill_between(
        X_pred_sorted.flatten(),
        (y_pred_mean_sorted - 2 * y_pred_std_sorted).flatten(),
        (y_pred_mean_sorted + 2 * y_pred_std_sorted).flatten(),
        color="red",
        alpha=0.1,
        label="95% confidence interval",
    )

    # Plot true function
    x_true = np.linspace(
        min(X_pred.min(), X_train.min()), max(X_pred.max(), X_train.max()), 1000
    ).reshape(-1, 1)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, color="black", linestyle="--", label="True function")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    else:
        plt.title("Predictive Distribution")

    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_reliability_diagram(
    bin_confidences: jax.Array,
    bin_accuracies: jax.Array,
    num_bins: int,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(visible=True, axis="y")

    bar_centers = jnp.linspace(0, 1, num_bins + 1)[:-1] + 1 / (2 * num_bins)
    bar_width = 1 / num_bins

    ax.bar(
        x=bar_centers,
        height=bin_accuracies,
        width=bar_width,
        label="Outputs",
        color="blue",
        edgecolor="black",
    )

    ax.bar(
        x=bar_centers,
        height=bin_confidences - bin_accuracies,
        width=bar_width / 2,
        bottom=bin_accuracies,
        label="Gap",
        color="red",
        edgecolor="red",
        alpha=0.4,
    )

    ax.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle="--", color="gray")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    fig.legend()

    ax.set_aspect("equal")

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()

    else:
        plt.show()

    return plt.gcf()


def create_proportion_diagram(
    bin_proportions: jax.Array,
    num_bins: int,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(visible=True, axis="y")

    bar_centers = jnp.linspace(0, 1, num_bins + 1)[:-1] + 1 / (2 * num_bins)
    bar_width = 1 / num_bins

    ax.bar(
        x=bar_centers,
        height=bin_proportions,
        width=bar_width,
        label="Proportions",
        color="green",
        edgecolor="black",
        alpha=0.4,
    )

    ax.axhline(y=1 / num_bins, color="gray", linestyle="--", label="Uniform")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Proportion")
    fig.legend()

    ax.set_aspect("equal")

    if save_path is not None:
        fig.savefig(save_path)
        fig.clear()

    else:
        plt.show()

    return plt.gcf()


def plot_sinusoid_task(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    X_pred=None,
    y_pred=None,
    y_std=None,
    title=None,
):
    """Plot the training and test data for the sinusoid task with optional predictions.

    Args:
        X_train: Training input data (shape: n_samples, 1)
        y_train: Training target data (shape: n_samples, 1)
        X_test: Test input data (shape: n_samples, 1), optional
        y_test: Test target data (shape: n_samples, 1), optional
        X_pred: Prediction input data (shape: n_samples, 1), optional
        y_pred: Prediction mean (shape: n_samples, 1), optional
        y_std: Prediction standard deviation (shape: n_samples, 1), optional
        title: Plot title, optional

    Returns:
        The matplotlib figure object
    """
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays if they are JAX arrays
    if hasattr(X_train, "device_buffer"):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        if X_pred is not None and y_pred is not None:
            X_pred = np.array(X_pred)
            y_pred = np.array(y_pred)
            if y_std is not None:
                y_std = np.array(y_std)

    # Plot training data
    plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")

    # Plot test data if provided
    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test data")

    # Plot prediction with uncertainty if provided
    if X_pred is not None and y_pred is not None:
        # Sort X for proper line plotting
        sort_idx = np.argsort(X_pred.flatten())
        X_pred_sorted = X_pred[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        plt.plot(X_pred_sorted, y_pred_sorted, color="red", label="Prediction")

        # Plot uncertainty if provided
        if y_std is not None:
            y_std_sorted = y_std[sort_idx]
            plt.fill_between(
                X_pred_sorted.flatten(),
                (y_pred_sorted - 2 * y_std_sorted).flatten(),
                (y_pred_sorted + 2 * y_std_sorted).flatten(),
                color="red",
                alpha=0.2,
                label="95% confidence interval",
            )

    # Plot true function
    x_true = np.linspace(0, 8, 1000).reshape(-1, 1)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, color="black", linestyle="--", label="True function")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    else:
        plt.title("Sinusoid Task Data")

    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def print_results(results_dict, title=None):
    """Print a dictionary of results in a nicely formatted way.

    Args:
        results_dict: Dictionary containing metric names and values
        title: Optional title to display before results
    """
    if title:
        print(f"\n{title}")  # noqa: T201
        print("-" * 40)  # noqa: T201

    # Find the longest key for alignment
    max_key_length = max(len(str(key)) for key in results_dict)

    # Print each key-value pair with aligned formatting
    for key, value in results_dict.items():
        if isinstance(value, float | np.floating | jnp.floating):
            print(f"{key!s:<{max_key_length}} : {value:.6f}")  # noqa: T201
        else:
            try:
                print(f"{key!s:<{max_key_length}} : {value.item():.6f}")  # noqa: T201
            except Exception as _:  # noqa: BLE001
                print(f"{key!s:<{max_key_length}} : {value}")  # noqa: T201


def plot_figure_1(params, curv, *, save_fig=True):
    """Plot the loss landscape and Laplace approximation ellipses.

    Args:
        params: Dictionary of model parameters
        curv: Scale matrix from the posterior (R_laplax)
        save_fig: Whether to save the figure (default: True)

    Returns:
        fig: The matplotlib figure object
        ax: The matplotlib axes object
    """
    # Select a style bundle
    style = bundles.icml2024(usetex=True)

    # Apply the style to matplotlib
    plt.rcParams.update(style)

    # Get the optimal parameters
    w1_opt_laplax = params["theta1"]
    w2_opt_laplax = params["theta2"]

    # Create parameter grid for visualization
    W1, W2 = jnp.meshgrid(jnp.linspace(-3, 3, 1000), jnp.linspace(-3, 3, 1000))

    # Compute loss landscape
    b = -1
    eps = 0.2
    x1, y1 = 1, 1
    x2, y2 = -1, -1

    f1 = jax.nn.relu(W1.ravel() * x1 + b) * W2.ravel()
    f2 = jax.nn.relu(W1.ravel() * x2 + b) * W2.ravel()

    loss = 0.5 * ((f1 - y1) ** 2 + (f2 - y2) ** 2).reshape(W1.shape) + 0.5 * eps * (
        W1**2 + W2**2
    )

    # Create figure
    fig, ax = plt.subplots()

    # Plot contours of the loss landscape
    levels = [0.95, 1.0, 1.1, 1.2, 1.5, 2, 3, 5, 7, 10, 20]
    CS = plt.contour(
        W1, W2, loss, levels=levels, colors="k", alpha=0.5
    )  # , fontsize=3)  # Added smaller fontsize

    # Add contour labels
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return f"{s}"

    ax.clabel(CS, CS.levels, fmt=fmt, fontsize=6)

    # Plot the optimal point
    ax.plot(w1_opt_laplax, w2_opt_laplax, "ko")  # , label="Optimal Parameters")

    # Plot the ellipse representing the curvature
    ellipse = jnp.linspace(-jnp.pi, jnp.pi, 100)
    x = jnp.cos(ellipse)
    y = jnp.sin(ellipse)
    xy = jnp.vstack((x, y))
    xy = curv @ xy

    # Plot 1-sigma and 2-sigma ellipses
    ax.plot(
        w1_opt_laplax + xy[0, :],
        w2_opt_laplax + xy[1, :],
        linestyle="solid",
        color="#00695b",
        lw=2,
    )  # , label="1-sigma (LapLaX)")
    ax.plot(
        w1_opt_laplax + 2 * xy[0, :],
        w2_opt_laplax + 2 * xy[1, :],
        linestyle="--",
        color="#00695b",
        lw=2,
    )  # , label="2-sigma (LapLaX)")

    # Set labels and limits
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()

    if save_fig:
        plt.savefig("laplax_figure_1.png", bbox_inches="tight", dpi=600)

    return fig, ax
