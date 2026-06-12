from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import random
from loguru import logger


# Context manager to suppress info-level logging
@contextmanager
def suppress_info_logging(module: str):
    logger.disable(module)
    try:
        # Only disable INFO and below, allow WARNING and above
        logger.remove()
        logger.add(
            lambda _: None,  # Sink that does nothing
            level="INFO",  # Only suppress INFO and below
            filter=lambda record: record["name"] == module,
        )
        yield
    finally:
        logger.enable(module)
        logger.remove()


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

    def add(self, x, y):
        new_X = jnp.concatenate((self.X, jnp.atleast_2d(x)))
        new_Y = jnp.concatenate((self.y, y))
        return DataLoader(
            new_X, new_Y, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def n_elements(self):
        return self.dataset_size


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
        self.linear4 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        x = nnx.tanh(self.linear3(x))
        return self.linear4(x)


def train_model(model, optimizer, dataloader, train_step, n_epochs=1000):
    """Trains the given model on the data.

    Args:
        model: nnx.Module that represents the model, can be pretrained
        optimizer: nnx.Optimizer to use for training
        dataloader: Data on which to train
        train_step: Function that performs the train step for one batch
        n_epochs: Number of epochs to train for
        lr: learning rate for optimizer

    Returns:
        Trained model
    """
    for epoch in range(n_epochs):
        for x_batch, y_batch in dataloader:
            loss = train_step(model, optimizer, x_batch, y_batch)

        if epoch % 100 == 0 and epoch != 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")
    print(f"Final loss: {loss:.4f}")
    return model


def split(model):
    """Split an nnx module into parameters and parameter-agnostic function.

    Args:
        model: nnx.module to split.

    Returns:
        Tuple of callable function taking model input and parameters,
        and model parameters.
    """
    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    return model_fn, params


DEFAULT_INTERVALS = [
    (0, 2),
    (4, 5),
    (6, 8),
]


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
