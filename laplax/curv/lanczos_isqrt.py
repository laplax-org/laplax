import jax
import jax.numpy as jnp


def lanczos_isqrt(
    A,
    b,
    *,
    tol=1e-5,
    min_eta=1e-20,
    max_iter=10,
    overwrite_b=False,
):
    """Conjugate gradient method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - x0 (array): initial guess.
    - atol (float): absolut tolerance.
    - tol (float): relative tolerance.
    - max_iter (int): maximum number of iterations.
    """

    @jax.jit
    def _step(values):
        ds, rs, rs_norm_sq, p, eta, k = values
        # Compute search direction
        true_fn = lambda _p: rs[:, k] + rs_norm_sq[k] / rs_norm_sq[k - 1] * _p
        false_fn = lambda _p: _p
        p = jax.lax.cond(k > 0, true_fn, false_fn, p)

        # Compute modified Lanzcos vector
        w = A(p)
        eta = p @ w
        ds = ds.at[:, k].set(p / jnp.sqrt(eta))

        # Update residual
        mu = rs_norm_sq[k] / eta
        rs_prev_k = rs  # rs[:, :k]
        rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)

        # Full reorthogonalization of residual (double Gram-Schmidt)
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )

        rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1].T @ rs[:, k + 1])
        jax.debug.print("eta: {a} - sq_norm = {r}", a=eta, r=rs_norm_sq[k])

        return ds, rs, rs_norm_sq, p, eta, k + 1

    def _cond_fun(values):
        rs_norm_sq = values[2]
        k = values[-1]
        return (rs_norm_sq[k] > sqtol) & (k < max_iter)  # & (eta > min_eta)

    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter))  # only implemented for order='K'
    rs = jnp.zeros((b.size, max_iter + 1))  #   only implemented for order='K'
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)

    # Initialize loop variables
    sqtol = (
        tol**2
    )  # max(tol ** 2, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    # min_eta = min_eta  # max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf  # to make sure the first iteration is done
    rs = rs.at[:, 0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k = jax.lax.while_loop(
        _cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0)
    )

    return ds[:, :k]


import numpy as np


def test_lanczos_compute_efficient():
    # Create a simple positive definite matrix
    n = 100
    np.random.seed(42)
    A_np = np.random.randn(n, n)
    A_np = A_np @ A_np.T + n * np.eye(n)  # Make it positive definite
    A_jax = jnp.array(A_np)

    # Define the linear operator
    def linear_op(x):
        return A_jax @ x

    # Create a random vector b
    b = jnp.ones(n) / np.sqrt(n)  # Normalized vector

    # Run the Lanczos algorithm
    ds = lanczos_compute_efficient(
        linear_op,
        b,
        tol=1e-8,
        max_iter=100
    )
    
    print(f"Lanczos vectors shape: {ds.shape}")
    
    # Verify orthogonality of Lanczos vectors
    D = ds.T @ ds
    print("Orthogonality check (should be close to identity):")
    print(np.round(D, 5))
    
    # Verify the tridiagonal matrix T = D.T @ A @ D
    T = ds.T @ (A_jax @ ds)
    print("Tridiagonal matrix T:")
    print(np.round(T, 5))
    
    # Optional: Compare with a direct eigendecomposition
    eigvals_lanczos = np.linalg.eigvalsh(T)
    eigvals_direct = np.linalg.eigvalsh(A_np)[:ds.shape[1]]
    print("Eigenvalues from L anczos:")
    print(np.sort(eigvals_lanczos))
    print("Smallest eigenvalues from direct computation:")
    print(np.sort(eigvals_direct)[:ds.shape[1]])

if __name__ == "__main__":
    test_lanczos_compute_efficient()