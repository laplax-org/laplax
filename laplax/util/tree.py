# tree.py

"""Relevant tree operations."""

import math
import operator
from functools import partial
from itertools import starmap

import jax
import jax.numpy as jnp

from laplax.types import Any, Array, Callable, Float, KeyType, Kwargs, PyTree
from laplax.util.flatten import unravel_array_into_pytree

# ---------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------


def get_size(tree: PyTree) -> int:
    """Compute the total number of elements in a PyTree.

    Args:
        tree: A PyTree whose total size is to be calculated.

    Returns:
        The total number of elements across all leaves in the PyTree.
    """
    flat, _ = jax.tree_util.tree_flatten(tree)
    return sum(math.prod(arr.shape) for arr in flat)


def to_dtype(tree: PyTree, dtype: Any) -> PyTree:
    """Convert all elements of a PyTree to a specified data type.

    Args:
        tree: A PyTree whose elements are to be converted.
        dtype: The target data type.

    Returns:
        A PyTree with all elements converted to the specified data type.
    """
    return jax.tree.map(lambda x: x.astype(dtype), tree)


# ---------------------------------------------------------------
# Tree operations
# ---------------------------------------------------------------


def add(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Add corresponding elements of two PyTrees.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        A PyTree where each leaf is the element-wise sum of the leaves in
            `tree1` and `tree2`.
    """
    return jax.tree.map(jnp.add, tree1, tree2)


def neg(tree: PyTree) -> PyTree:
    """Negate all elements of a PyTree.

    Args:
        tree: A PyTree to negate.

    Returns:
        A PyTree with negated elements.
    """
    return jax.tree.map(jnp.negative, tree)


def sub(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Subtract corresponding elements of two PyTrees.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        A PyTree where each leaf is the element-wise difference of the leaves in
            `tree1` and `tree2`.
    """
    return add(tree1, neg(tree2))


def mul(scalar: Float, tree: PyTree) -> PyTree:
    """Multiply all elements of a PyTree by a scalar.

    Args:
        scalar: The scalar value to multiply by.
        tree: A PyTree to multiply.

    Returns:
        A PyTree where each leaf is the element-wise product of the leaves in
            `tree` and `scalar`.
    """
    return jax.tree.map(lambda x: scalar * x, tree)


def sqrt(tree: PyTree) -> PyTree:
    """Compute the square root of each element in a PyTree.

    Args:
        tree: A PyTree whose elements are to be square-rooted.

    Returns:
        A PyTree with square-rooted elements.
    """
    return jax.tree.map(jnp.sqrt, tree)


def invert(tree: PyTree) -> PyTree:
    """Invert all elements of a PyTree.

    Args:
        tree: A PyTree to invert.

    Returns:
        A PyTree with inverted elements.
    """
    return jax.tree.map(jnp.invert, tree)


def mean(tree: PyTree, **kwargs: Kwargs) -> PyTree:
    """Compute the mean of each element in a PyTree.

    Args:
        tree: A PyTree whose elements are to be averaged.
        **kwargs: Additional keyword arguments for `jnp.mean`.

    Returns:
        A PyTree with averaged elements.
    """
    return jax.tree.map(partial(jnp.mean, **kwargs), tree)


def std(tree: PyTree, **kwargs: Kwargs) -> PyTree:
    """Compute the standard deviation of each element in a PyTree.

    Args:
        tree: A PyTree whose elements are to be standard-deviated.
        **kwargs: Additional keyword arguments for `jnp.std`.

    Returns:
        A PyTree with standard-deviated elements.
    """
    return jax.tree.map(partial(jnp.std, **kwargs), tree)


def var(tree: PyTree, **kwargs: Kwargs) -> PyTree:
    """Compute the variance of each element in a PyTree.

    Args:
        tree: A PyTree whose elements are to be variance-ed.
        **kwargs: Additional keyword arguments for `jnp.var`.

    Returns:
        A PyTree with variance-ed elements.
    """
    return jax.tree.map(partial(jnp.var, **kwargs), tree)


def cov(tree: PyTree, **kwargs: Kwargs) -> PyTree:
    """Compute the covariance of each element in a PyTree.

    Args:
        tree: A PyTree whose elements are to be covariance-ed.
        **kwargs: Additional keyword arguments for `jnp.cov`.

    Returns:
        A PyTree with covariance-ed elements.
    """
    return jax.tree.map(partial(jnp.cov, **kwargs), tree)


def tree_matvec(tree: PyTree, vector: Array) -> PyTree:
    """Multiply a PyTree by a vector.

    Args:
        tree: A PyTree to multiply.
        vector: A vector to multiply by.

    Returns:
        A PyTree with multiplied elements.
    """
    # Flatten the vector
    vec_flatten, vec_def = jax.tree.flatten(vector)
    num_vec_flatten = len(vec_flatten)

    # Array flattening and reshaping
    arr_flatten, _ = jax.tree.flatten(tree)
    arr_flatten = [
        jnp.concatenate(
            [
                arr_flatten[i * num_vec_flatten + j].reshape(*vec_flatten[j].shape, -1)
                for i in range(num_vec_flatten)
            ],
            axis=-1,
        )
        for j in range(num_vec_flatten)
    ]

    # Array, vector to correct shape
    tree = jax.tree.unflatten(vec_def, arr_flatten)
    vec_flatten = jnp.concatenate([v.reshape(-1) for v in vec_flatten])

    # Apply matmul
    return jax.tree.map(lambda p: p @ vec_flatten, tree)


def tree_partialmatvec(tree: PyTree, vector: Array) -> PyTree:
    """Multiply a PyTree by a vector.

    Args:
        tree: A PyTree to multiply.
        vector: A vector to multiply by.

    Returns:
        A PyTree with multiplied elements.
    """
    return jax.tree.map(lambda arr: arr @ vector, tree)


# ---------------------------------------------------------------
# Create common arrays in given tree structure
# ---------------------------------------------------------------


def ones_like(tree: PyTree) -> PyTree:
    """Create a PyTree of ones with the same structure as the input tree.

    Args:
        tree: A PyTree whose structure and shape will be used.

    Returns:
        A PyTree of ones with the same structure and shape as `tree`.
    """
    return jax.tree.map(jnp.ones_like, tree)


def zeros_like(tree: PyTree) -> PyTree:
    """Create a PyTree of zeros with the same structure as the input tree.

    Args:
        tree: A PyTree whose structure and shape will be used.

    Returns:
        A PyTree of zeros with the same structure and shape as `tree`.
    """
    return jax.tree.map(jnp.zeros_like, tree)


def randn_tree_like(key: KeyType, tree: PyTree) -> PyTree:
    """Generate a PyTree of random normal values with the same structure as the input.

    Args:
        key: A JAX PRNG key.
        tree: A PyTree whose structure will be replicated.

    Returns:
        A PyTree of random normal values.
    """
    # Flatten the tree
    leaves, treedef = jax.tree.flatten(tree)

    # Split key
    keys = jax.random.split(key, len(leaves))

    # Generate random numbers
    random_leaves = [
        jax.random.normal(k, shape=leaf.shape)
        for k, leaf in zip(keys, leaves, strict=True)
    ]

    return jax.tree.unflatten(treedef, random_leaves)


def randn_like(key: KeyType, layout: PyTree | int) -> PyTree:
    """Generate random normal values matching the given layout.

    Args:
        key: A JAX PRNG key.
        layout: A PyTree whose structure will be replicated, or the rank of a
            1D random vector.

    Returns:
        A PyTree of random normal values.
    """
    if isinstance(layout, int):
        return jax.random.normal(key, shape=(layout,))

    return randn_tree_like(key, layout)


def normal_like(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable[[PyTree], PyTree],
    *,
    rank: int | None = None,
) -> PyTree:
    """Generate a PyTree of random normal values scaled and shifted by `mean`.

    Args:
        key: A JAX PRNG key.
        mean: A PyTree representing the mean of the distribution.
        scale_mv: A callable that scales a PyTree.
        rank: The rank of the PyTree. If `None`, the layout is inferred from `mean`
            assuming full rank.

    Returns:
        A PyTree of random normal values shifted by `mean`.
    """
    return add(mean, scale_mv(randn_like(key, layout=mean if rank is None else rank)))


def basis_vector_from_index(idx: int, tree: PyTree) -> PyTree:
    """Create a basis vector from an index in a PyTree.

    Args:
        idx: The index of the basis vector.
        tree: A PyTree whose structure will be used.

    Returns:
        A PyTree with a basis vector at the specified index.
    """
    # Create a tree of zeros with the same structure
    zeros = zeros_like(tree)

    # Flatten the tree to get a list of arrays and the tree definition
    flat, tree_def = jax.tree_util.tree_flatten(zeros)

    # Compute the cumulative sizes of each array in the flat structure
    sizes = jnp.array([math.prod(arr.shape) for arr in flat])

    # Find the index of the array containing the idx
    cum_sizes = jnp.cumsum(sizes)
    k = jnp.searchsorted(cum_sizes, idx, side="right")

    # Compute the adjusted index within the identified array
    idx_corr = idx - jnp.where(k > 0, cum_sizes[k - 1], 0)

    # Define a function to update the k-th array with the basis vector
    def update_array(i, arr):
        return jax.lax.cond(
            i == k,
            lambda: arr.flatten().at[idx_corr].set(1.0).reshape(arr.shape),
            lambda: arr,
        )

    # Map the update function across the flattened list of arrays
    updated_flat = list(starmap(update_array, enumerate(flat)))

    # Reconstruct the tree with the updated flat structure
    return jax.tree_util.tree_unflatten(tree_def, updated_flat)


def eye_like_with_basis_vector(tree: PyTree) -> PyTree:
    """Create a PyTree where each element is a basis vector.

    Args:
        tree: A PyTree defining the structure.

    Returns:
        A PyTree of basis vectors.
    """
    num_elems = get_size(tree)
    return jax.lax.map(
        partial(basis_vector_from_index, tree=tree), jnp.arange(num_elems)
    )


def eye_like(tree: PyTree) -> PyTree:
    """Create a PyTree equivalent of an identity matrix.

    Args:
        tree: A PyTree defining the structure.

    Returns:
        A PyTree equivalent to an identity matrix.
    """
    return unravel_array_into_pytree(tree, 1, jnp.eye(get_size(tree)))


def tree_slice(tree: PyTree, a: int, b: int) -> PyTree:
    """Slice each leaf of a PyTree along the first dimension.

    Args:
        tree: A PyTree to slice.
        a: The start index.
        b: The end index.

    Returns:
        A PyTree with sliced leaves.
    """
    return jax.tree.map(operator.itemgetter(slice(a, b)), tree)


def tree_vec_get(tree: PyTree, idx: int) -> Any:
    """Retrieve the element at the specified index from a flattened PyTree.

    Args:
        tree: A PyTree to retrieve the element from.
        idx: The index of the element.

    Returns:
        The element at the specified index.
    """
    if isinstance(tree, jnp.ndarray):
        return tree.reshape(-1)[idx]
    # Column flat and get index
    flat, _ = jax.tree_util.tree_flatten(tree)
    flat = jnp.concatenate([f.reshape(-1) for f in flat])
    return flat[idx]


# ---------------------------------------------------------------
# For testing
# ---------------------------------------------------------------


def allclose(tree1: PyTree, tree2: PyTree) -> bool:
    """Check whether all elements in two PyTrees are approximately equal.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        True if all elements are approximately equal, otherwise False.
    """
    return jax.tree.all(jax.tree.map(jnp.allclose, tree1, tree2))
