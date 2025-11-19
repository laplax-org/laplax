import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import diagonal, kronecker, to_dense
from laplax.util.tree import get_size


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_dense(n):
    key = jax.random.PRNGKey(123)
    A = jax.random.normal(key, (n, n))

    # Calling diagonal on a dense matrix should return jnp.diag(A)
    diag_computed = diagonal(A)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_and_to_dense_flat_mvp(n):
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # 1) Compare diagonal(mv, layout=n) to jnp.diag(A)
    diag_computed = diagonal(mv, layout=n)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=n) to A
    dense_computed = to_dense(mv, layout=n)
    np.testing.assert_allclose(dense_computed, A, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("n", [2, 5, 20, 100])
def test_diagonal_low_rank(n):
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Rank-1: V is n x 1, VV^T is rank-1 positive semidefinite
    u = jax.random.normal(key1, (n,))
    rank1_factor = u.reshape(-1, 1)  # n x 1 matrix
    rank1_matrix = rank1_factor @ rank1_factor.T  # uu^T, not uv^T

    key3 = jax.random.split(key2)[0]
    V_rank2 = jax.random.normal(key3, (n, 2))
    rank2_matrix = V_rank2 @ V_rank2.T

    key4 = jax.random.split(key3)[0]
    V_rank5 = jax.random.normal(key4, (n, 5))
    rank5_matrix = V_rank5 @ V_rank5.T

    def compare_diagonals(full_matrix, factor_matrix):
        assert jnp.allclose(
            jnp.diag(full_matrix), diagonal(factor_matrix, layout=n, low_rank=True)
        ), "Diagonal computation mismatch for low-rank matrix"

    compare_diagonals(rank1_matrix, rank1_factor)
    compare_diagonals(rank2_matrix, V_rank2)
    compare_diagonals(rank5_matrix, V_rank5)


@pytest.mark.parametrize(("n1", "n2"), [(2, 3), (3, 4)])
def test_diagonal_and_to_dense_pytree_mvp(n1, n2):
    key = jax.random.PRNGKey(999)
    layout = {
        "x": jnp.zeros(n1),
        "y": jnp.zeros(n2),
    }
    example_flat, tree_def = jax.tree.flatten(layout)
    sizes = [leaf.size for leaf in example_flat]
    total_dim = sum(sizes)

    # Create a random (total_dim x total_dim) matrix A
    A = jax.random.normal(key, (total_dim, total_dim))

    def mv(pytree_vec):
        leaves, _ = jax.tree.flatten(pytree_vec)
        x_flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
        y_flat = A @ x_flat

        # Split back into the shapes
        split_indices = np.cumsum(sizes[:-1])
        y_splits = jnp.split(y_flat, split_indices)

        # Reshape each split to match original leaf shape
        y_leaves = []
        for s, leaf_shape in zip(
            y_splits, [leaf.shape for leaf in example_flat], strict=True
        ):
            y_leaves.append(s.reshape(leaf_shape))

        # Unflatten back
        return jax.tree.unflatten(tree_def, y_leaves)

    # 1) Compare diagonal(mv, layout=layout) to jnp.diag(A)
    diag_computed = diagonal(mv, layout)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=layout) to A
    dense_computed = to_dense(mv, layout)

    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)

    mv_wrapped = wrap_function(mv, input_fn=unflatten, output_fn=flatten)
    dense_computed = to_dense(mv_wrapped, get_size(layout))
    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)


@pytest.mark.parametrize(("na", "nb"), [(2, 3), (3, 2), (1, 4)])
@pytest.mark.parametrize("mode", ["vmap", "map"])
def test_kronecker_dense_equivalence(na, nb, mode):
    key = jax.random.PRNGKey(2025)
    kA, kB = jax.random.split(key)

    A = jax.random.normal(kA, (na, na))
    B = jax.random.normal(kB, (nb, nb))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    mv_kron = kronecker(mv_a, mv_b, layout_a=na, layout_b=nb, mode=mode)

    dense_kron_mv = to_dense(mv_kron, layout=na * nb)
    dense_kron_ref = jnp.kron(A, B)

    np.testing.assert_allclose(dense_kron_mv, dense_kron_ref, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(("na", "nb"), [(2, 2), (2, 3)])
@pytest.mark.parametrize("mode", ["vmap", "map"])
def test_kronecker_apply_vector(na, nb, mode):
    key = jax.random.PRNGKey(7)
    kA, kB, kv = jax.random.split(key, 3)

    A = jax.random.normal(kA, (na, na))
    B = jax.random.normal(kB, (nb, nb))
    v = jax.random.normal(kv, (na * nb,))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    mv_kron = kronecker(mv_a, mv_b, layout_a=na, layout_b=nb, mode=mode)

    out_mv = mv_kron(v)
    out_ref = jnp.kron(A, B) @ v

    np.testing.assert_allclose(out_mv, out_ref, atol=1e-7, rtol=1e-7)
