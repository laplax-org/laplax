"""Matrix-free array operations for matrix-vector products."""

from collections.abc import Callable
from functools import singledispatch

import jax
import jax.numpy as jnp

from laplax import util
from laplax.types import Array, Kwargs, Layout, PyTree
from laplax.util.tree import (
    basis_vector_from_index,
    eye_like,
    get_size,
)


@singledispatch
def diagonal(
    mv: Callable | jnp.ndarray,
    layout: Layout | None = None,
    *,
    mv_jittable: bool = True,
    **kwargs: Kwargs,
) -> Array:
    """Compute the diagonal of a matrix represented by a matrix-vector product function.

    This function extracts the diagonal of a matrix using basis vectors and a
    matrix-vector product (MVP) function. If the input is already a dense matrix, its
    diagonal is directly computed.

    Args:
        mv: Either:

            - A callable that implements the MVP, or
            - A dense matrix (jax.Array) for which the diagonal is directly extracted.
        layout: Specifies the structure of the matrix:

            - int: The size of the matrix (for flat MVP functions).
            - PyTree: A structure to generate basis vectors matching the matrix
                dimensions.
            - None: If `mv` is a dense matrix.
        mv_jittable: Whether to JIT compile the basis vector generator.
        **kwargs:
            diagonal_batch_size: Batch size for applying the MVP function.

    Returns:
        An array representing the diagonal of the matrix.

    Raises:
        TypeError: If `layout` is not provided when `mv` is a callable.
    """
    if isinstance(mv, Callable) and layout is None:
        msg = "either size or tree needs to be present"
        raise TypeError(msg)

    if isinstance(mv, jax.Array):
        return jnp.diag(mv)

    # Define basis vector generator based on layout type
    if isinstance(layout, int):  # Integer layout defines size
        size = layout

        @jax.jit
        def get_basis_vec(idx: int) -> jax.Array:
            zero_vec = jnp.zeros(size)
            return zero_vec.at[idx].set(1.0)

    else:  # PyTree layout
        size = get_size(layout)

        @jax.jit
        def get_basis_vec(idx: int) -> PyTree:
            return basis_vector_from_index(idx, layout)

    def diag_elem(i):
        return util.tree.tree_vec_get(mv(get_basis_vec(i)), i)

    if mv_jittable:
        diag_elem = jax.jit(diag_elem)

    return jax.lax.map(
        diag_elem, jnp.arange(size), batch_size=kwargs.get("diagonal_batch_size")
    )


@singledispatch
def to_dense(mv: Callable, layout: Layout, **kwargs: Kwargs) -> Array:
    """Generate a dense matrix representation from a matrix-vector product function.

    Converts a matrix-vector product function into its equivalent dense matrix form
    by applying the function to identity-like basis vectors.

    Args:
        mv: A callable implementing the matrix-vector product function.
        layout: Specifies the structure of the input:

            - int: The size of the input dimension (flat vectors).
            - PyTree: The structure for input to the MVP.
            - None: Defaults to an identity-like structure.
        **kwargs: Additional options:

            - `to_dense_batch_size`: Batch size for applying the MVP function.

    Returns:
        A dense matrix representation of the MVP function.

    Raises:
        TypeError: If `layout` is neither an integer nor a PyTree structure.
    """
    # Create the identity-like basis based on `layout`
    if isinstance(layout, int):
        identity = jnp.eye(layout)
    elif isinstance(layout, PyTree):
        identity = eye_like(layout)
    else:
        msg = "`layout` must be an integer or a PyTree structure."
        raise TypeError(msg)

    return jax.tree.map(
        jnp.transpose,
        jax.lax.map(mv, identity, batch_size=kwargs.get("to_dense_batch_size")),
    )  # jax.lax.map shares along the first axis (rows instead of columns).
