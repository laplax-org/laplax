# /examples/ex_helper.py

import datetime
import itertools
import math
import pickle
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import torch
from flax import nnx
from loguru import logger
from orbax import checkpoint as ocp

from laplax.types import Callable, Float, Kwargs, PriorArguments

# ---------------------------------------------------------------------
# Checkpoint Helper
# ---------------------------------------------------------------------


def save_with_pickle(obj, path):
    """Save object to pickle file."""
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_with_pickle(path):
    """Load object from pickle file.

    Returns:
        The stored object.
    """
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def save_model_checkpoint(
    model,
    checkpoint_path: str | Path = "./tests/test-checkpoints",
):
    """Save model checkpoint using Orbax.

    Returns:
        The checkpoint directory.
    """
    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Split model into graph and params for checkpointing
    _, state = nnx.split(model)

    # Save the checkpoint
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir.resolve(), state, force=True)
    checkpointer.wait_until_finished()
    logger.info(f"Model checkpoint saved to {ckpt_dir}")
    return ckpt_dir


def load_model_checkpoint(
    model_class,
    model_kwargs,
    checkpoint_path,
):
    """Load model checkpoint using Orbax.

    Returns:
        A triple of the model, the graph def, and the restored state.
    """
    model = model_class(**model_kwargs, rngs=nnx.Rngs(0))
    graph_def, abstract_state = nnx.split(model)

    # Restore the checkpoint
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        abstract_state,
    )

    # Merge into model
    model = nnx.merge(graph_def, state_restored)

    logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    return model, graph_def, state_restored


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class CSVLogger:
    def __init__(
        self,
        output_dir="results",
        file_name="regression_experiments.csv",
        *,
        force: bool = True,
    ):
        """A CSV logger for experiments.

        Args:
            output_dir (str or Path): Directory in which to store the CSV.
            file_name (str): Name of the CSV file.
            force (bool): If True, delete any existing CSV at initialization.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / file_name

        if force and self.csv_path.exists():
            self.csv_path.unlink()
            logger.info(f"Existing log {self.csv_path} removed (force=True)")

        # Track whether to write headers on next write
        self._write_header = not self.csv_path.exists()

    def log(
        self, results: dict, experiment_name: str, *, log_args: dict | None = None
    ) -> Path:
        """Append a single experiment's results to the CSV.

        Args:
            results (dict): A dict that may contain an "evaluation" sub-dict and
                optional "nll" field.
            experiment_name (str): A unique name or identifier for this run.
            log_args (dict, optional): Any additional metadata to record.

        Returns:
            Path: The path to the CSV file.
        """
        log_args = {} if log_args is None else dict(log_args)

        # Build the row data
        row = {
            **results,
            **log_args,
            "experiment_name": experiment_name,
            "timestamp": datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        log_df = pd.DataFrame([row])
        log_df.to_csv(
            self.csv_path,
            mode="a",
            header=self._write_header,
            index=False,
        )

        if self._write_header:
            self._write_header = False

        logger.info(f"Logged experiment '{experiment_name}' to {self.csv_path}")
        return self.csv_path

    def log_samples(self, results: dict, experiment_name: str):
        """Storing samples."""
        sample_path = self.output_dir / ("samples_" + experiment_name)
        save_with_pickle(results, path=sample_path)


def generate_experiment_name(**kwargs: Kwargs):
    """Generate a descriptive name for the experiment.

    Returns:
        The experiment name.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    name_parts = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{timestamp}_{'_'.join(name_parts)}"


# ---------------------------------------------------------------------
# Fix randomness
# ---------------------------------------------------------------------


def fix_random_seed(seed: int):
    """Fix random seed in numpy, scipy and torch backend."""
    # Python built-in RNG
    random.seed(seed + 1)
    # NumPy RNG (also covers SciPy)
    np.random.seed(seed + 2)  # noqa: NPY002
    # PyTorch CPU RNG
    torch.manual_seed(seed + 3)
    # PyTorch GPU RNG (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 4)
    # Ensure deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# DataLoader Helper
# ---------------------------------------------------------------------


class SimpleLoader:
    def __init__(self, input, target, batch_size=50):
        self._input, self._target = input, target
        self._batch_size = batch_size
        self._rng = np.random.default_rng()

    def __iter__(self):
        # Always shuffle at the start of each epoch
        perm = self._rng.permutation(self._input.shape[0])
        idx_iter = iter(perm)

        for batch_idx in iter(
            lambda: list(itertools.islice(idx_iter, self._batch_size)), []
        ):
            input_batch = jnp.take(self._input, jnp.array(batch_idx), axis=0)
            target_batch = jnp.take(self._target, jnp.array(batch_idx), axis=0)

            yield {"input": input_batch, "target": target_batch}


class DataLoader:
    """Simple dataloader used in many examples."""

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


class Model(nnx.Module):
    """Two-layer tanh MLP for regression."""

    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.final_layer = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.final_layer(h)


class LimitedLoader:
    """DataLoader wrapper that limits the number of batches."""

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        batch_iter = iter(self.loader)
        for _ in range(self.max_batches):
            yield next(batch_iter)

    def __len__(self):
        return self.max_batches


def _ensure_data_loader(data):
    if isinstance(data, list | tuple):
        if len(data) == 2:
            return SimpleLoader(data[0], data[1])
        msg = f"Unknown length of data: {len(data)}"
        raise TypeError(msg)
    if isinstance(data, dict):
        return SimpleLoader(data["input"], data["target"])
    msg = f"Unknown data type: {type(data)}"
    raise TypeError(msg)


# ---------------------------------------------------------------------
# Data Generation Helper
# ---------------------------------------------------------------------

DEFAULT_INTERVALS = [
    (0, 2),
    (4, 5),
    (6, 8),
]


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


# ---------------------------------------------------------------------
# Kernel Helper
# ---------------------------------------------------------------------


def kernel_matern52_1d(x, y, params):
    """Matern 5/2 kernel in 1D.

    Args:
        x: Input data.
        y: Input data.
        params: Parameters of the kernel.

    Returns:
        The kernel matrix.
    """
    variance = params.get("variance", 1.0)
    lengthscale = params.get("lengthscale", 1.0)
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    r = jnp.abs(x[:, None, 0] - y[None, :, 0])
    sqrt5 = jnp.sqrt(5.0)
    val = sqrt5 * r / lengthscale
    return variance * (1.0 + val + (val**2) / 3.0) * jnp.exp(-val)


def kernel_periodic_1d(x, y, variance=1.0, lengthscale=1.0, period=1.0, params=None):
    """Periodic kernel in 1D.

    Args:
        x: Input data.
        y: Input data.
        variance: Variance of the kernel.
        lengthscale: Lengthscale of the kernel.
        period: Period of the kernel.
        params: Parameters of the kernel.

    Returns:
        The kernel matrix.
    """
    if params is not None:
        variance = params.get("variance", variance)
        lengthscale = params.get("lengthscale", lengthscale)
        period = params.get("period", period)

    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    d = jnp.abs(x[:, None, 0] - y[None, :, 0])
    s = jnp.sin(math.pi * d / period)
    return variance * jnp.exp(-(2.0 * s**2) / (lengthscale**2))


# ---------------------------------------------------------------------
# Last-layer-only helper
# ---------------------------------------------------------------------


def split_model(model, *, last_layer_only=False):
    """Split model into graph and params.

    Returns:
        A tuple of the model function and the parameters PyTree.
    """
    if last_layer_only:
        graph_def, relevant_params, remaining_params = nnx.split(
            model, lambda n, _: "final_layer" in n, ...
        )

        def model_fn(input, params):
            return nnx.call((graph_def, params, remaining_params))(input)[0]

        return model_fn, relevant_params

    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    return model_fn, params


# ---------------------------------------------------------------------
# Train Helper
# ---------------------------------------------------------------------


def train_map_model(
    model,
    train_loader,
    n_epochs,
    *,
    lr=1e-3,
    verbose=True,
    log_every_n_epochs=10,
    loss_type="mse",
    test_loader=None,
    warmup_steps=0,
    decay_steps=None,
    end_lr=None,
):
    """Train a model using MAP estimation.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        n_epochs: Number of epochs to train for
        lr: Initial learning rate
        verbose: Whether to print progress
        log_every_n_epochs: How often to log progress
        loss_type: Type of loss function ("mse" or "cross_entropy")
        test_loader: Optional DataLoader for test data
        warmup_steps: Number of warmup steps for learning rate schedule
        decay_steps: Number of decay steps for learning rate schedule
            (defaults to total steps)
        end_lr: Final learning rate after decay (defaults to 0.1 * initial lr)

    Returns:
        The trained model.

    Raises:
        ValueError: If an unknown loss type is provided.
    """
    # Calculate total steps for learning rate schedule
    total_steps = n_epochs * len(train_loader)
    decay_steps = decay_steps or total_steps
    end_lr = end_lr or lr * 0.0001

    # Create learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_lr,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule))
    loss = 0.0

    if loss_type == "mse":

        def loss_fn(y_pred, y):
            return jnp.sum((y_pred - y) ** 2)

    elif loss_type == "cross_entropy":

        def loss_fn(y_pred, y):
            return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()

    else:
        msg = f"Unknown loss type: {loss_type}"
        raise ValueError(msg)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def forward(model):
            y_pred = nnx.vmap(model)(x)
            return loss_fn(y_pred, y)

        loss, grads = nnx.value_and_grad(forward)(model)
        optimizer.update(grads)  # Inplace updates

        return loss

    @nnx.jit
    def eval_step(model, x, y):
        y_pred = nnx.vmap(model)(x)
        pred_class = jnp.argmax(y_pred, axis=1)
        return jnp.mean(pred_class == y)

    for epoch in range(1, n_epochs + 1):
        for xb, yb in train_loader:
            loss = train_step(model, optimizer, xb, yb)
        if verbose and epoch % log_every_n_epochs == 0:
            logger.info(f"Epoch {epoch}/{n_epochs}, loss={loss:.4f}")
    if verbose:
        logger.info(f"Final training loss: {loss:.4f}")

    if loss_type == "cross_entropy" and test_loader is not None:
        total_acc = 0.0
        n_batches = 0
        for xb, yb in test_loader:
            acc = eval_step(model, xb, yb)
            total_acc += acc
            n_batches += 1
        avg_acc = total_acc / n_batches
        logger.info(f"Test accuracy: {avg_acc:.4f}")

    return model


def train_model_with_objective(
    model,
    optimizer,
    objective_fn,
    data_loader,
    num_epochs,
    log_every=100,
):
    """Train model with a custom objective function.

    Args:
        model: nnx.Module
        optimizer: nnx.Optimizer
        objective_fn: function taking (model, batch) -> loss
        data_loader: iterable yielding batches
        num_epochs: int
        log_every: int

    Returns:
        The trained model
    """

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(objective_fn)(model, batch)
        optimizer.update(grads)
        return loss

    for epoch in range(num_epochs):
        for batch in data_loader:
            loss = train_step(model, optimizer, batch)

        if epoch % log_every == 0:
            logger.info(f"Epoch {epoch}: loss={loss:.4f}")

    return model


def optimize_prior_prec_gradient(
    objective: Callable[[PriorArguments], float],
    data,
    init_prior_prec: Float | None = None,
    init_sigma_noise: Float | None = None,
    *,
    num_epochs: int = 100,
    learning_rate: float = 1,
    **kwargs: Kwargs,
) -> Float:
    """Optimize prior precision using gradient descent.

    Args:
        objective: A callable objective function that takes `PriorArguments` as input
            and returns a float result.
        data: A batch of data.
        init_prior_prec: Initial prior precision value (default: None)
        init_sigma_noise: Initial noise standard deviation value (default: None)
        num_epochs: Number of optimization epochs (default: 100)
        learning_rate: Initial learning rate for the optimizer (default: 1)
        **kwargs: Additional arguments

    Returns:
        The optimized prior precision value.

    Raises:
        ValueError: When neither init_prior_prec nor init_sigma_noise is provided.
    """
    del kwargs

    # Validate inputs
    if init_prior_prec is None and init_sigma_noise is None:
        msg = "Provide at least one of init_prior_prec or init_sigma_noise."
        raise ValueError(msg)

    # Initialize log-parameters
    params = {}
    if init_prior_prec is not None:
        params["prior_prec"] = jnp.array(jnp.log(init_prior_prec))
    if init_sigma_noise is not None:
        params["sigma"] = jnp.array(jnp.log(init_sigma_noise))

    logger.info("Initial params: {}", params)

    # Initialize optimizer with learning rate schedule
    logger.info("Initializing optimizer with cosine learning rate schedule")
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create a simple data loader if not provided
    data = _ensure_data_loader(data)

    # Single optimization step
    @jax.jit
    def step(params, data, opt_state):
        # Compute loss and gradients w.r.t. log-params
        loss, grads = jax.value_and_grad(
            lambda p: objective(jax.tree.map(jnp.exp, p), data)
        )(params)

        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    # Optimization loop
    for epoch in range(1, num_epochs + 1):
        for dp in data:
            params, opt_state, loss = step(params, dp, opt_state)
        logger.debug(f"Epoch {epoch:02d}: loss={loss:.6f}, ")

    # Convert back from log-domain
    params = jax.tree.map(jnp.exp, params)

    return params
