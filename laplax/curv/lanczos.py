import jax
import jax.numpy as jnp

from laplax.curv.utils import LowRankTerms, get_matvec
from laplax.types import Array, Callable, DType, Float, KeyType, Layout
from laplax.util.flatten import wrap_function


def lanczos_iterations(
    matvec: Callable[[Array], Array],
    b: jax.Array,
    *,
    maxiter: int = 20,
    tol: Float = 1e-6,
    full_reorthogonalize: bool = True,
    dtype: DType = jnp.float64,
) -> tuple[Array, Array, Array]:
    """Runs Lanczos iterations starting from vector b.

    Args:
        matvec: A callable that computes A @ x.
        b: Starting vector.
        maxiter: Number of iterations.
        tol: Tolerance to detect convergence.
        full_reorthogonalize: If True, reorthogonalize at every step.
        dtype: Data type for the Lanczos scalars/vectors.

    Returns:
        alpha: 1D array of Lanczos scalars (diagonal of T).
        beta: 1D array of off-diagonals (with beta[-1] not used).
        V: 2D array (maxiter+1 x input_dim) of Lanczos vectors.
    """
    b = jnp.asarray(b, dtype=dtype)
    b_norm = jnp.linalg.norm(b, 2)
    v0 = b / b_norm

    alpha = jnp.zeros(maxiter, dtype=dtype)
    beta = jnp.zeros(maxiter, dtype=dtype)
    V = jnp.zeros((maxiter + 1, b.shape[0]), dtype=dtype)
    V = V.at[0].set(v0)

    def reorthogonalize(w: Array, V: Array, i: int) -> Array:
        def body_fn(j: int, w_acc: Array) -> Array:
            coeff = jnp.dot(V[j], w_acc)
            return w_acc - coeff * V[j]

        return jax.lax.fori_loop(0, i, body_fn, w)

    def _body_fn(carry, i):
        v, alpha, beta, V = carry
        w = matvec(v)
        a = jnp.dot(v, w)
        w = w - a * v
        if full_reorthogonalize:
            w = reorthogonalize(w, V, i)
            w = reorthogonalize(w, V, i)

        b_val = jnp.linalg.norm(w, 2)
        b_val = jnp.where(b_val < tol, 0.0, b_val)
        v_next = jax.lax.cond(
            b_val > 0,  # type: ignore[operator]
            lambda _: w / b_val,
            lambda _: v,  # In degenerate cases, no progress is made.
            operand=None,
        )
        alpha = alpha.at[i].set(a)
        beta = beta.at[i].set(b_val)
        V = V.at[i + 1].set(v_next)
        return (v_next, alpha, beta, V), None

    init_carry = (v0, alpha, beta, V)
    indices = jnp.arange(maxiter)
    (_, alpha, beta, V), _ = jax.lax.scan(_body_fn, init_carry, indices)
    return alpha, beta, V


def construct_tridiagonal(alpha: Array, beta: Array) -> Array:
    """Constructs the symmetric tridiagonal matrix from Lanczos scalars.

    Args:
        alpha: Diagonal elements.
        beta: Off-diagonal elements (only beta[:k-1] are used).

    Returns:
        A k x k symmetric tridiagonal matrix T.
    """
    k = alpha.shape[0]
    T = jnp.zeros((k, k), dtype=alpha.dtype)
    T = T.at[jnp.arange(k), jnp.arange(k)].set(alpha)
    # Only the first k-1 values of beta are used.
    T = T.at[jnp.arange(k - 1), jnp.arange(1, k)].set(beta[: k - 1])
    T = T.at[jnp.arange(1, k), jnp.arange(k - 1)].set(beta[: k - 1])
    return T


def compute_eigendecomposition(
    alpha: Array, beta: Array, V: Array, *, compute_vectors: bool = False
) -> Array | tuple[Array, Array]:
    """Computes the eigendecomposition of the tridiagonal matrix generated by Lanczos.

    Args:
        alpha: Diagonal elements.
        beta: Off-diagonal elements.
        V: Lanczos vectors.
        compute_vectors: If True, compute Ritz vectors in the original space.

    Returns:
        If compute_vectors is True: (eigvals, ritz_vectors),
        else: eigvals.
    """
    T = construct_tridiagonal(alpha, beta)
    if compute_vectors:
        eigvals, eigvecs = jnp.linalg.eigh(T)
        # Use the first maxiter Lanczos vectors (exclude the extra one).
        V_matrix = V[:-1].T  # shape: (input_dim, maxiter)
        ritz_vectors = jnp.dot(V_matrix, eigvecs)
        return eigvals, ritz_vectors
    eigvals = jnp.linalg.eigvalsh(T)
    return eigvals


def lanczos_lowrank(
    A: Callable[[Array], Array] | Array,
    *,
    key: KeyType,
    layout: Layout | None = None,
    rank: int = 20,
    tol: float = 1e-6,
    mv_dtype: DType | None = None,
    calc_dtype: DType = jnp.float64,
    return_dtype: DType | None = None,
    mv_jittable: bool = True,
    full_reorthogonalize: bool = True,
    **kwargs,
) -> LowRankTerms:
    """Compute a low-rank approximation using the Lanczos algorithm.

    Args:
        A: Matrix or callable representing the matrix-vector product.
        key: PRNG key for random initialization.
        layout: Dimension of input vector (required if A is callable).
        rank: Number of leading eigenpairs to compute.
        tol: Convergence tolerance for the algorithm.
        mv_dtype: Data type for matrix-vector products.
        calc_dtype: Data type for internal calculations.
        return_dtype: Data type for returned results.
        mv_jittable: If True, enables JIT compilation of matrix-vector products.
        full_reorthogonalize: Whether to perform full reorthogonalization.
        **kwargs: Additional arguments (ignored).

    Returns:
        LowRankTerms: A dataclass containing:
            - U: Eigenvectors as a matrix of shape (size, rank)
            - S: Eigenvalues as an array of length rank
            - scalar: Scalar factor, initialized to 0.0
    """
    del kwargs

    # Initialize handling mixed precision.
    original_float64_enabled = jax.config.read("jax_enable_x64")

    if mv_dtype is None:
        mv_dtype = jnp.float64 if original_float64_enabled else jnp.float32

    if return_dtype is None:
        return_dtype = jnp.float64 if original_float64_enabled else jnp.float32

    jax.config.update("jax_enable_x64", calc_dtype == jnp.float64)

    # Obtain a uniform matrix-vector multiplication function.
    matvec, size = get_matvec(A, layout=layout, jit=mv_jittable)

    # Wrap to_dtype around mv if necessary.
    if mv_dtype != calc_dtype:
        matvec = wrap_function(
            matvec,
            input_fn=lambda x: jnp.asarray(x, dtype=mv_dtype),
            output_fn=lambda x: jnp.asarray(x, dtype=calc_dtype),
        )

    # Initialize random starting vector.
    b = jax.random.normal(key, (size,), dtype=calc_dtype)

    # Run Lanczos iterations.
    alpha, beta, V = lanczos_iterations(
        matvec, b, maxiter=rank, tol=tol, full_reorthogonalize=full_reorthogonalize
    )
    eigvals, eigvecs = compute_eigendecomposition(alpha, beta, V, compute_vectors=True)

    # Prepare and convert the results
    low_rank_result = LowRankTerms(
        U=jnp.asarray(eigvecs, dtype=return_dtype),
        S=jnp.asarray(eigvals, dtype=return_dtype),
        scalar=jnp.asarray(0.0, dtype=return_dtype),
    )

    # Restore the original configuration dtype
    jax.config.update("jax_enable_x64", original_float64_enabled)
    return low_rank_result
