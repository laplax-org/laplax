import datetime
import itertools
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


# # ---------------------------------------------------------------------
# DataLoader Helper
# ---------------------------------------------------------------------


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
    optimizer = nnx.Optimizer(model, optax.adamw(schedule))
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
