from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
from jax import random
import jax
from typing import Iterator, List, Tuple


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


# Function to create the sinusoid dataset
def get_sinusoid_example(
    num_data: int = 150,
    sigma_noise: float = 0.3,
    batch_size: int = 150,
    rng_key=None,
) -> tuple[
    jnp.ndarray, jnp.ndarray, Iterator[tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray
]:
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    """Generate a sinusoid dataset.

    Args:
        num_data: Number of data points.
        sigma_noise: Standard deviation of the noise.
        batch_size: Batch size for the data loader.
        rng_key: Random number generator key.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        train_loader: Data loader for training data.
        X_test: Testing input data.
    """
    # Split RNG key for reproducibility
    rng_key, rng_noise = random.split(rng_key)

    # Generate random training data
    X_train = (
        random.uniform(rng_key, (num_data, 1)) * 8
    )  # X_train values between 0 and 8
    noise = random.normal(rng_noise, X_train.shape) * sigma_noise
    y_train = jnp.sin(X_train) + noise

    # Create a simple data loader function (generator)
    # def _data_loader(X, y, batch_size):
    #     dataset_size = X.shape[0]
    #     indices = np.arange(dataset_size)
    #     np.random.shuffle(indices)
    #     for start_idx in range(0, dataset_size, batch_size):
    #         batch_indices = indices[start_idx : start_idx + batch_size]
    #         yield X[batch_indices], y[batch_indices]

    # Generate testing data
    X_test = jnp.linspace(-5, 13, 500).reshape(-1, 1)

    # Create the training data loader
    train_loader = DataLoader(X_train, y_train, batch_size)

    return X_train, y_train, train_loader, X_test


def get_sinusoid_examples_spaced(
    num_data: int = 150,
    sigma_noise: float = 0.3,
    batch_size: int = 150,
    rng_key=None,
    intervals: List[Tuple[float, float]] = [(-5, -3), (0, 2), (4, 5), (7, 10), (12, 13)],
) -> Tuple[jnp.ndarray, jnp.ndarray, Iterator[Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
    """
    Generate a sinusoid dataset with data points sampled from specified intervals.

    Args:
        num_data: Number of data points.
        sigma_noise: Standard deviation of the noise.
        batch_size: Batch size for the data loader.
        rng_key: Random number generator key.
        intervals: List of (start, end) tuples specifying allowed data intervals.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        train_loader: Data loader for training data.
        X_test: Testing input data.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    # Split RNG key for reproducibility
    rng_key, rng_interval, rng_noise = random.split(rng_key, 3)


    # Generate random values within selected intervals

    tuples_as_array = jnp.asarray(intervals)

    def f(key):
        key1, key2 = jax.random.split(key, 2)
        interval = jax.random.choice(key1, tuples_as_array, axis=0)
        x = jax.random.uniform(key2, minval=interval[0], maxval=interval[1])
        return x


    X_train = (jax.vmap(f)(jax.random.split(rng_key, num_data))).reshape(-1, 1)

    """
    # Generate X_train by selecting a random interval for each point
    X_train_list = []
    for x in X_train:
        selected_interval_idx = int(np.random.choice(len(intervals)))
        selected_interval = intervals[selected_interval_idx]
        # Scale `x` to the selected interval
        x = x * (selected_interval[1] - selected_interval[0]) + selected_interval[0]
        X_train_list.append(x)

    X_train = jnp.array(X_train_list)
    """
    # Add noise to the output
    noise = random.normal(rng_noise, X_train.shape) * sigma_noise
    y_train = jnp.sin(X_train) + noise

    # Generate testing data across a larger range
    X_test = jnp.linspace(-5, 13, 500).reshape(-1, 1)

    # Create the training data loader
    train_loader = DataLoader(X_train, y_train, batch_size)

    return X_train, y_train, train_loader, X_test