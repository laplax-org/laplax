"""Utilities for handling DataLoaders/Iterables instead of single batches."""

import operator
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm

from laplax.types import Any, Array, Callable, Data, Iterable, Kwargs, Layout, PyTree
from laplax.util.flatten import wrap_function
from laplax.util.mv import diagonal, to_dense
from laplax.util.tree import add
from laplax.util.utils import identity

# ------------------------------------------------------------------------
#  Data transformations
# ------------------------------------------------------------------------


def input_target_split(batch: tuple[Array, Array]) -> Data:
    """Split a batch into input and target components.

    Args:
        batch: A tuple where the first element is the input data and the second
            element is the target data.

    Returns:
        A dictionary containing:

            - "input": Input data from the batch.
            - "target": Target data from the batch.
    """
    return {"input": batch[0], "target": batch[1]}


# ------------------------------------------------------------------------
#  Reduction functions
# ------------------------------------------------------------------------


def reduce_sum(
    res_new: Any, state: Any | None = None, *, keepdims: bool = True, axis: int = 0
) -> tuple[Any, Any]:
    """Perform a reduction by summing results across a specified axis.

    Args:
        res_new: The new result to add to the current state.
        state: The current accumulated state (default: None).
        keepdims: Whether to keep reduced dimensions (default: True).
        axis: The axis along which to sum (default: 0).

    Returns:
        The updated state and the new accumulated sum.
    """
    summed = jax.tree.map(lambda x: jnp.sum(x, keepdims=keepdims, axis=axis), res_new)
    if state is None:
        return summed, summed
    new_state = add(state, summed)
    return new_state, new_state


def reduce_add(
    res_new: Any,
    state: Any | None = None,
) -> tuple[Any, Any]:
    """Perform a reduction by adding results.

    Args:
        res_new: The new result to add to the current state.
        state: The current accumulated state (default: None).

    Returns:
        The updated state and the new accumulated sum.
    """
    if state is None:
        return res_new, res_new
    new_state = add(res_new, state)
    return new_state, new_state


def concat(
    tree1: PyTree,
    tree2: PyTree,
    axis: int = 0,
) -> PyTree:
    """Concatenate two PyTrees along a specified axis.

    Args:
        tree1: The first PyTree to concatenate.
        tree2: The second PyTree to concatenate.
        axis: The axis along which to concatenate (default: 0).

    Returns:
        A PyTree resulting from concatenating `tree1` and `tree2`.
    """
    return jax.tree.map(
        lambda x, y: jax.numpy.concatenate([x, y], axis=axis), tree1, tree2
    )


def reduce_concat(
    res_new: Any,
    state: Any | None = None,
    *,
    axis: int = 0,
) -> tuple[Any, Any]:
    """Perform a reduction by concatenating results.

    Args:
        res_new: The new result to concatenate with the current state.
        state: The current accumulated state (default: None).
        axis: The axis along which to concatenate (default: 0).

    Returns:
        The updated state and the concatenated result.
    """
    if state is None:
        return res_new, res_new
    new_state = concat(state, res_new, axis=axis)
    return new_state, new_state


def reduce_online_mean(
    res_new: Any,
    state: tuple | None = None,
) -> tuple[Any, tuple]:
    """Compute the online mean of results, maintaining a running count and sum.

    Args:
        res_new: The new result to incorporate into the mean calculation.
        state: A tuple containing the current count and running sum (default: None).

    Returns:
        The updated mean and the new state (count, running sum).
    """
    batch_size = jax.tree.map(lambda x: x.shape[0] if x.ndim > 0 else 1, res_new)
    batch_sum = jax.tree.map(
        lambda x: jnp.sum(x, axis=0) if x.ndim > 0 else jnp.sum(x),
        res_new,
    )

    if state is None:
        return jax.tree.map(operator.truediv, batch_sum, batch_size), (
            batch_size,
            batch_sum,
        )

    old_count, old_sum = state
    total_count = jax.tree.map(operator.add, old_count, batch_size)
    new_sum = add(old_sum, batch_sum)

    current_mean = jax.tree.map(operator.truediv, new_sum, total_count)

    return current_mean, (total_count, new_sum)


# ------------------------------------------------------------------------
#  Core batch processing logic
# ------------------------------------------------------------------------


def process_batches(
    function: Callable,
    data_loader: Iterable,
    transform: Callable,
    reduce: Callable,
    *args: Any,
    verbose_logging: bool = False,
    **kwargs: Kwargs,
) -> Any:
    """Process batches of data using a function, transformation, and reduction.

    Args:
        function: A callable that processes a single batch of data.
        data_loader: An iterable yielding batches of data.
        transform: A callable that transforms each batch into the desired format.
        reduce: A callable that reduces results across batches.
        *args: Additional positional arguments for the processing function.
        verbose_logging: Whether to log progress using tqdm (default: False).
        **kwargs: Additional keyword arguments for the processing function.

    Returns:
        The final result after processing all batches.

    Raises:
        ValueError: If the data loader is empty.
    """
    state = None
    result = None
    for batch in tqdm(
        data_loader, desc="Processing batches", disable=not verbose_logging
    ):
        result = function(*args, data=transform(batch), **kwargs)
        result, state = reduce(result, state)
    if result is None:
        msg = "data loader was empty"
        raise ValueError(msg)
    return result


# ------------------------------------------------------------------------
#  Wrapper functions
# ------------------------------------------------------------------------


def execute_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
    *,
    jit: bool = False,
    **kwargs: Kwargs,
) -> Any:
    """Execute batch processing with a data loader.

    Args:
        function: A callable that processes a single batch of data.
        data_loader: An iterable yielding batches of data.
        transform: A callable to transform each batch into the desired format
            (default: `input_target_split`).
        reduce: A callable to reduce results across batches
            (default: `reduce_online_mean`).
        jit: Whether to JIT compile the processing function (default: False).
        **kwargs: Additional keyword arguments for the processing function.

    Returns:
        The final result after processing all batches.
    """
    fn = jax.jit(function) if jit else function
    return process_batches(fn, data_loader, transform, reduce, **kwargs)


def wrap_function_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
    *,
    jit: bool = False,
) -> Callable:
    """Wrap a function to process batches with a data loader.

    This wrapper generates a callable that processes all batches from the data loader
    using the specified function, transformation, and reduction.

    Args:
        function: A callable that processes a single batch of data.
        data_loader: An iterable yielding batches of data.
        transform: A callable to transform each batch into the desired format
            (default: `input_target_split`).
        reduce: A callable to reduce results across batches
            (default: `reduce_online_mean`).
        jit: Whether to JIT compile the processing function (default: False).

    Returns:
        A wrapped function for batch processing.
    """
    fn = jax.jit(function) if jit else function

    def wrapped(*args, **kwargs):
        return process_batches(fn, data_loader, transform, reduce, *args, **kwargs)

    return wrapped


# ------------------------------------------------------------------------
# DataLoader MV Wrapper
# ------------------------------------------------------------------------


class DataLoaderMV:
    def __init__(
        self,
        mv: Callable,
        loader: Iterable,
        transform: Callable = input_target_split,
        reduce: Callable = reduce_online_mean,
        *,
        verbose_logging: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        """Initialize the DataLoaderMV object.

        Args:
            mv: A callable that processes a single batch of data.
            loader: An iterable yielding batches of data.
            transform: A callable to transform each batch into the desired format
                (default: `input_target_split`).
            reduce: A callable to reduce results across batches
                (default: `reduce_online_mean`).
            verbose_logging: Whether to log progress using tqdm (default: False).
            **kwargs: Additional keyword arguments (currently unused).
        """
        del kwargs
        self.mv = mv
        self.loader = loader
        self.transform = transform
        self.reduce = reduce
        self.input_transform = identity
        self.output_transform = identity
        self.verbose_logging = verbose_logging

    def __call__(self, vec: Array) -> Array | PyTree:
        """Process the input vector using the data loader and return the result.

        Args:
            vec: The input vector to process.

        Returns:
            The processed result as an Array or PyTree.
        """
        return self.output_transform(
            process_batches(
                self.mv,
                self.loader,
                transform=self.transform,
                reduce=self.reduce,
                vec=self.input_transform(vec),
                verbose_logging=self.verbose_logging,
            )
        )

    def lower_func(self, func: Callable, **kwargs: Kwargs) -> Array:
        """Apply a function to the data loader and return the result.

        Args:
            func: A callable to apply to the data loader.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            The result of applying the function to the data loader.
        """

        def _body_fn(data):
            return func(
                wrap_function(
                    partial(self.mv, data=data),
                    input_fn=self.input_transform,
                    output_fn=self.output_transform,
                ),
                **kwargs,
            )

        return process_batches(
            _body_fn,
            self.loader,
            transform=self.transform,
            reduce=self.reduce,
            verbose_logging=self.verbose_logging,
        )


@to_dense.register
def _(mv: DataLoaderMV, layout: Layout, **kwargs: Kwargs) -> Array:
    """Apply to_dense to DataLoaderMV.

    Returns:
        The result of applying the function to the data loader.
    """
    return mv.lower_func(to_dense, layout=layout, **kwargs)


@diagonal.register
def _(mv: DataLoaderMV, layout: Layout | None = None) -> Array:
    """Apply diagonal to DataLoaderMV.

    Returns:
        The result of applying the function to the data loader.
    """
    return mv.lower_func(diagonal, layout=layout)


@wrap_function.register
def _(
    mv: DataLoaderMV,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
    argnums: int = 0,
) -> Callable:
    """Apply wrap_function to DataLoaderMV.

    Returns:
        A DataLoaderMV object representing the wrapped MV.
    """
    # Create new transforms without overwriting existing ones
    new_input_transform = wrap_function(
        mv.input_transform, input_fn=input_fn, argnums=argnums
    )
    new_output_transform = wrap_function(
        mv.output_transform,
        output_fn=output_fn,
    )

    new_mv = DataLoaderMV(
        mv.mv, mv.loader, mv.transform, mv.reduce, verbose_logging=mv.verbose_logging
    )
    new_mv.input_transform = new_input_transform
    new_mv.output_transform = new_output_transform

    return new_mv
