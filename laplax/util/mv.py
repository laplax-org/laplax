# mv.py

"""Matrix-free array operations for matrix-vector products."""

from collections.abc import Callable
from functools import singledispatch

import jax
from jax import numpy as jnp

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
    low_rank: bool = False,
    **kwargs,
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
        low_rank: Assumes VV^T, where mv = V is a low-rank matrix.
        **kwargs:
            diagonal_batch_size: Batch size for applying the MVP function.

    Returns:
        jax.Array: An array representing the diagonal of the matrix.

    Raises:
        TypeError: If `layout` is not provided when `mv` is a callable.
    """
    if isinstance(mv, Callable) and layout is None:
        msg = "either size or tree needs to be present"
        raise TypeError(msg)

    if low_rank:
        if isinstance(mv, jax.Array):
            """diag(V V^T) = sum_j V[:, j]^2"""
            return jnp.sum(jnp.square(mv), axis=1)
        if isinstance(mv, Callable):
            dense = util.mv.to_dense(mv, layout=layout, **kwargs)
            return jnp.sum(jnp.square(dense), axis=1)

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


@singledispatch
def kronecker(
    mv_a: Callable,
    mv_b: Callable,
    layout_a: Layout,
    layout_b: Layout,
    *,
    mode: str = "vmap",
    batch_size: int | None = None,
) -> Callable:
    """Create a Kronecker product MVP with selectable mapping mode.
    Uses (A ⊗ B) vec(X) = vec(B X A^T).
    - mode="vmap": vectorized mapping (fast, higher memory).
    - mode="map": sequential mapping via lax.map (lower memory).
    """
    if isinstance(layout_a, int) and isinstance(layout_b, int):
        size_a = layout_a
        size_b = layout_b
    else:
        size_a = get_size(layout_a)
        size_b = get_size(layout_b)

    if mode not in {"vmap", "map"}:
        msg = "mode must be 'vmap' or 'map'"
        raise ValueError(msg)

    if mode == "vmap":

        def kronecker_mv(v):
            X = v.reshape(size_a, size_b).T
            mv_b_vmap = jax.vmap(mv_b, in_axes=1, out_axes=1)
            Y = mv_b_vmap(X)
            mv_a_vmap = jax.vmap(mv_a, in_axes=0, out_axes=0)
            Z = mv_a_vmap(Y)
            return Z.T.reshape(-1)

    else:  # mode == "map"

        def kronecker_mv(v):
            X = v.reshape(size_a, size_b).T

            def apply_b_to_column(col):
                return mv_b(col)

            Y = jax.lax.map(apply_b_to_column, X.T, batch_size=batch_size).T

            def apply_a_to_row(row):
                return mv_a(row)

            Z = jax.lax.map(apply_a_to_row, Y, batch_size=batch_size)
            return Z.T.reshape(-1)

    return kronecker_mv


def kronecker_product_factors(factors_mv: list[Callable], factors_layout: list[Layout]) -> Callable:
    """Create an efficient Kronecker product from multiple factors.

    Computes (A_1 ⊗ A_2 ⊗ ... ⊗ A_n) @ v efficiently without forming the full product.

    Args:
        factors_mv: List of matrix-vector product functions, ordered from left to right
        factors_layout: List of layout specifications for each factor

    Returns:
        A callable that computes the full Kronecker product matrix-vector product

    Example:
        >>> # For (A ⊗ B ⊗ C) @ v
        >>> mv_kron = kronecker_product_factors([mv_a, mv_b, mv_c], [n, m, k])
        >>> result = mv_kron(v)
    """
    if len(factors_mv) == 1:
        return factors_mv[0]

    if len(factors_mv) == 2:
        return kronecker(factors_mv[0], factors_mv[1], factors_layout[0], factors_layout[1])

    # For multiple factors, compose pairwise from right to left
    # (A ⊗ B ⊗ C) = ((A ⊗ B) ⊗ C)
    result_mv = factors_mv[-1]
    result_layout = factors_layout[-1]

    for i in range(len(factors_mv) - 2, -1, -1):
        result_mv = kronecker(factors_mv[i], result_mv, factors_layout[i], result_layout)
        # Update layout to reflect the Kronecker product size
        result_layout = get_size(factors_layout[i]) * get_size(result_layout)

    return result_mv
