"""Operations for flattening PyTrees into arrays."""

import math
from collections.abc import Generator
from functools import singledispatch

import jax
import jax.numpy as jnp

from laplax.types import Any, Array, Callable, PyTree
from laplax.util.utils import identity

# ---------------------------------------------------------------
# Flattening utilities
# ---------------------------------------------------------------


def cumsum(
    seq: Generator,
) -> list[int]:
    """Compute the cumulative sum of a sequence.

    This function takes a sequence of integers and returns a list of cumulative
    sums.

    Args:
        seq: A generator or sequence of integers.

    Returns:
        A list where each element is the cumulative sum up to that point
            in the input sequence.
    """
    total = 0
    return [total := total + ele for ele in seq]


def full_flatten(
    tree: PyTree,
) -> Array:
    """Flatten a PyTree into a single 1D array.

    This function takes a PyTree and concatenates all its leaves into a single
    array.

    Returns:
        The flattened PyTree.
    """
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.flatten(tree)[0]])


def flatten_function(
    fn: Callable,
    layout: PyTree,
) -> Callable:
    """Wrap a function to flatten its input and output.

    This function takes a function and a layout, and returns a new function that
    accepts flattened input and returns also a flattened output.

    Args:
        fn: The function to wrap.
        layout: The layout of the PyTree.

    Returns:
        The wrapped function.
    """
    flatten, unflatten = create_pytree_flattener(layout)
    return wrap_function(fn, input_fn=unflatten, output_fn=flatten)


def create_pytree_flattener(
    tree: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """Create functions to flatten and unflatten a PyTree into and from a 1D array.

    The `flatten` function concatenates all leaves of the PyTree into a single
    vector. The `unflatten` function reconstructs the original PyTree from the
    flattened vector.

    Args:
        tree: A PyTree to derive the structure for flattening and unflattening.

    Returns:
        Tuple containing:

            - `flatten`: A function that flattens a PyTree into a 1D array.
            - `unflatten`: A function that reconstructs the PyTree from a 1D array.
    """
    # Get shapes and tree def for unflattening
    flat, tree_def = jax.tree.flatten(tree)
    all_shapes = [leaf.shape for leaf in flat]

    def _unflatten(arr: Array) -> PyTree:
        flat_vector_split = jnp.split(
            arr, cumsum(math.prod(sh) for sh in all_shapes)[:-1]
        )
        return jax.tree.unflatten(
            tree_def,
            [
                a.reshape(sh)
                for a, sh in zip(flat_vector_split, all_shapes, strict=True)
            ],
        )

    return full_flatten, _unflatten


def create_partial_pytree_flattener(
    tree: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """Create functions to flatten and unflatten partial PyTrees into and from arrays.

    This function assumes that each leaf in the PyTree is a multi-dimensional
    array, where the last dimension represents column indices. The `flatten`
    function combines all rows across leaves into a single 2D array. The
    `unflatten` function reconstructs the PyTree from this 2D array.

    Args:
        tree: A PyTree to derive the structure for flattening and unflattening.

    Returns:
        Tuple containing:

            - `flatten`: A function that flattens a PyTree into a 2D array.
            - `unflatten`: A function that reconstructs the PyTree from a 2D array.
    """

    def flatten(tree: PyTree) -> jax.Array:
        flat, _ = jax.tree_util.tree_flatten(tree)
        return jnp.concatenate(
            [leaf.reshape(-1, leaf.shape[-1]) for leaf in flat], axis=0
        )

    # Get shapes and tree def for unflattening
    flat, tree_def = jax.tree_util.tree_flatten(tree)
    all_shapes = [leaf.shape for leaf in flat]

    def unflatten(arr: jax.Array) -> PyTree:
        flat_vector_split = jnp.split(
            arr, cumsum(math.prod(sh[:-1]) for sh in all_shapes)[:-1], axis=0
        )  # Ignore column indices in shape.
        return jax.tree_util.tree_unflatten(
            tree_def,
            [
                flat_vector_split[i].reshape(all_shapes[i])
                for i in range(len(flat_vector_split))
            ],
        )

    return flatten, unflatten


def unravel_array_into_pytree(
    pytree: PyTree,
    axis: int,
    arr: Array,
) -> PyTree:
    """Unravel an array into a PyTree with a specified structure.

    This function splits and reshapes an array to match the structure of a given
    PyTree, with options to control the resulting shapes using the `axis` parameter.

    Args:
        pytree: The PyTree defining the desired structure.
        axis: The axis along which to split the array.
        arr: The array to be unraveled into the PyTree structure.

    Returns:
        A PyTree with the specified structure, populated with parts of the input array.

    This function follows the implementation in jax._src.api._unravel_array_into_pytree.
    """
    leaves, treedef = jax.tree.flatten(pytree)
    axis %= arr.ndim
    shapes = [arr.shape[:axis] + l.shape + arr.shape[axis + 1 :] for l in leaves]
    parts = jnp.split(arr, cumsum(math.prod(leaf.shape) for leaf in leaves[:-1]), axis)
    reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes, strict=True)]

    return jax.tree.unflatten(treedef, reshaped_parts)


@singledispatch
def wrap_function(
    fn: Callable,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
    argnums: int = 0,
) -> Callable:
    """Wrap a function with input and output transformations.

    This utility wraps a function `fn`, applying an optional transformation to its
    inputs before execution and another transformation to its outputs after
    execution.

    Args:
        fn: The function to be wrapped.
        input_fn: A callable to transform the input arguments (default: identity).
        output_fn: A callable to transform the output of the function
            (default: identity).
        argnums: The index of the argument to be transformed by `input_fn`.

    Returns:
        The wrapped function with input and output transformations applied.
    """

    def wrapper(*args, **kwargs) -> Any:
        # Use the identity function if input_fn or output_fn is None
        effective_input_fn = input_fn or identity
        effective_output_fn = output_fn or identity

        # Call the original function on transformed input
        transformed_args = (
            *args[:argnums],
            effective_input_fn(args[argnums]),
            *args[argnums + 1 :],
        )
        result = fn(*transformed_args, **kwargs)

        # Apply the output transformation function
        return effective_output_fn(result)

    return wrapper


def wrap_factory(
    factory: Callable,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
) -> Callable:
    """Wrap a factory function to apply input and output transformations.

    This function wraps a factory, ensuring that any callable it produces is
    transformed with `wrap_function` to apply input and output transformations.

    Args:
        factory: The factory function that returns a callable.
        input_fn: A callable to transform the input arguments (default: identity).
        output_fn: A callable to transform the output of the function
            (default: identity).

    Returns:
        The wrapped factory that produces transformed callables.
    """

    def wrapped_factory(*args, **kwargs) -> Callable:
        fn = factory(*args, **kwargs)
        return wrap_function(fn, input_fn, output_fn)

    return wrapped_factory
