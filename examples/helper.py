# /examples/helper.py

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return self.X[batch_indices], self.y[batch_indices]


DEFAULT_INTERVALS = [
    (0, 2),
    (4, 5),
    (6, 8),
]


class DictLoader:
    """Wrap a (xb, yb) loader to yield {"input": xb, "target": yb} batches."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for xb, yb in self.loader:
            yield {"input": xb, "target": yb}


def kernel_from_gpjax(x: jnp.ndarray, y: jnp.ndarray, kernel) -> jnp.ndarray:
    """Evaluate a GPJax kernel and return a dense covariance matrix.

    Args:
        x: Inputs with shape (n, d) or (d,)
        y: Inputs with shape (m, d) or (d,)
        kernel: A GPJax kernel instance.

    Returns:
        Dense covariance matrix with shape (n, m).
    """
    cov = kernel.cross_covariance(jnp.atleast_2d(x), jnp.atleast_2d(y))
    return cov


def bce_logits_full_data_approx(
    yb: jnp.ndarray, logits: jnp.ndarray, *, dataset_size: int
) -> jnp.ndarray:
    """Binary cross entropy from logits, scaled to approximate full-data NLL.

    This matches the scaling convention used throughout the examples:
    sum over batch, then scale by N / batch_size.

    Args:
        yb: Binary labels with shape (batch, 1) or (batch,)
        logits: Logits with shape (batch, 1) or (batch,)
        dataset_size: Total number of training samples N.

    Returns:
        Scalar NLL approximation.
    """
    yb = jnp.asarray(yb)
    logits = jnp.asarray(logits)
    batch_size = yb.shape[0]
    per = jax.nn.softplus(logits) - yb * logits
    return (dataset_size / batch_size) * jnp.sum(per)


def sigmoid_gaussian_moment(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    """Approximate E[sigmoid(Z)] for Z ~ N(mean, var).

    Uses the common logistic-Gaussian moment approximation.

    Args:
        mean: Mean of logits.
        var: Variance of logits (must be non-negative).

    Returns:
        Approximate Bernoulli probability.
    """
    return jax.nn.sigmoid(mean / jnp.sqrt(1.0 + (math.pi / 8.0) * var))


def entropy_bernoulli(prob: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a Bernoulli distribution.

    Args:
        prob: Bernoulli probability.

    Returns:
        Entropy in nats.
    """
    prob = jnp.clip(prob, 1e-6, 1.0 - 1e-6)
    return -(prob * jnp.log(prob) + (1.0 - prob) * jnp.log(1.0 - prob))


# Function to create the sinusoid dataset
def get_sinusoid_example(
    num_train_data: int = 150,
    num_valid_data: int = 50,
    num_test_data: int = 100,
    sigma_noise: float = 0.3,
    sinus_factor: float = 1.0,
    intervals: list[tuple[float, float]] = DEFAULT_INTERVALS,
    test_interval: tuple[float, float] = (0.0, 8.0),
    rng_key=None,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Generate a sinusoid dataset.

    Args:
        num_train_data: Number of training data points.
        num_valid_data: Number of validation data points.
        num_test_data: Number of test data points.
        sigma_noise: Standard deviation of the noise.
        sinus_factor: Factor to multiply the sinus input with.
        intervals: List of (min, max) tuples defining intervals for train/valid data.
        test_interval: (min, max) tuple defining interval for test data.
        rng_key: Random number generator key.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        X_valid: Validation input data.
        y_valid: Validation target data.
        X_test: Test input data.
        y_test: Test target data.
    """
    if rng_key is None:
        rng_key = random.key(0)

    # Split RNG key for reproducibility
    (
        rng_key,
        rng_x_train,
        rng_x_valid,
        rng_noise_train,
        rng_noise_valid,
        rng_noise_test,
    ) = random.split(rng_key, 6)

    tuples_as_array = jnp.asarray(intervals)

    def f(key):
        key1, key2 = jax.random.split(key, 2)
        interval = jax.random.choice(key1, tuples_as_array, axis=0)
        x = jax.random.uniform(key2, minval=interval[0], maxval=interval[1])
        return x

    # Generate random training data
    X_train = (jax.vmap(f)(jax.random.split(rng_x_train, num_train_data))).reshape(
        -1, 1
    )
    noise = random.normal(rng_noise_train, X_train.shape) * sigma_noise
    y_train = jnp.sin(X_train * sinus_factor) + noise

    # Generate calibration data
    X_valid = (jax.vmap(f)(jax.random.split(rng_x_valid, num_valid_data))).reshape(
        -1, 1
    )
    noise = random.normal(rng_noise_valid, X_valid.shape) * sigma_noise
    y_valid = jnp.sin(X_valid * sinus_factor) + noise

    # Generate testing data
    X_test = jnp.linspace(test_interval[0], test_interval[1], num_test_data).reshape(
        -1, 1
    )
    noise = random.normal(rng_noise_test, X_test.shape) * sigma_noise
    y_test = jnp.sin(X_test * sinus_factor) + noise

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_two_moons_example(
    num_data: int = 300,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray, np.ndarray]:
    """Generate a sklearn-free two-moons dataset.

    Args:
        num_data: Number of samples.
        noise: Additive Gaussian noise level.
        seed: Random seed.

    Returns:
        Feature matrix as JAX array, labels as JAX array, and NumPy copies.
    """
    rng = np.random.default_rng(seed)
    n1 = num_data // 2
    n2 = num_data - n1
    t1 = rng.uniform(0.0, np.pi, size=n1)
    t2 = rng.uniform(0.0, np.pi, size=n2)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    x2 = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    X_np = np.concatenate([x1, x2], axis=0)
    y_np = np.concatenate([np.zeros(n1), np.ones(n2)], axis=0)
    X_np = X_np + noise * rng.normal(size=X_np.shape)
    perm = rng.permutation(num_data)
    X_np = X_np[perm]
    y_np = y_np[perm]
    X = jnp.asarray(X_np, dtype=jnp.float64)
    y = jnp.asarray(y_np, dtype=jnp.float64).reshape(-1, 1)
    return X, y, X_np, y_np
