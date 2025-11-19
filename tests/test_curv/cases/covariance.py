import jax
import jax.numpy as jnp
import pytest_cases


def draw_random_onb(key, shape: tuple[int]) -> tuple[jax.Array, jax.Array]:
    """Draw a random orthogonal matrix and a random vector of singular values.

    Note: Shape (n, m) with n being the full rank and m being the low rank.

    Returns:
        A tuple of singular vectors and singular values.
    """
    keys = jax.random.split(key, 2)
    A = jax.random.uniform(keys[0], 2 * shape[:1])
    A = 0.5 * (A + A.T)
    A += jnp.eye(shape[0])
    U, _ = jnp.linalg.qr(A)
    S = jax.random.uniform(keys[1], (shape[1],))
    return U[: shape[1], :].T, S


class CurvatureTask:
    method = None
    atol = 1e-3
    rtol = 1e-3

    def __init__(self, seed, size=100, rank=10):
        # Set seed and keys
        self.seed = seed
        key = jax.random.key(seed)
        keys = jax.random.split(key, 3)
        self.key_arr = keys[0]
        self.key_curv_estimate = keys[1]

        # Initialize task
        self.size = size
        self.rank = rank
        self._initialize()

    def _initialize(self):
        # Get low rank array (repr. GGN and Hessian)
        u, s = draw_random_onb(self.key_arr, (self.size, self.rank))
        self.arr = u @ jnp.diag(s) @ u.T
        self.u = u
        self.s = s

    def adjust_curv_estimate(self, *args, **kwargs):
        raise NotImplementedError

    def adjust_prec(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def true_curv(self):
        raise NotImplementedError

    @property
    def arr_mv(self):
        def arr_mv(vec):
            return self.arr @ vec

        return arr_mv

    @property
    def layout(self):
        return jax.numpy.ones((100,))


class LANCZOSLowRankCurvatureTask(CurvatureTask):
    method = "lanczos"
    atol = 1e-1
    rtol = 5e-2

    @staticmethod
    def adjust_curv_estimate(low_rank_terms: dict):
        U = low_rank_terms.U
        S = low_rank_terms.S
        return U @ jnp.diag(S) @ U.T

    def adjust_prec(self, low_rank_terms: dict):
        U = low_rank_terms.U
        S = low_rank_terms.S
        scalar = low_rank_terms.scalar
        return U @ jnp.diag(S) @ U.T + scalar * jnp.eye(self.size)

    @property
    def true_curv(self):
        return self.arr

    @property
    def layout(self):
        return jax.numpy.ones((100,))


class LOBPCGLowRankCurvatureTask(LANCZOSLowRankCurvatureTask):
    method = "lobpcg"
    atol = 1e-2
    rtol = 1e-2


class DiagonalCurvatureTask(CurvatureTask):
    method = "diagonal"

    @staticmethod
    def adjust_curv_estimate(curv_estimate):
        return jnp.diag(curv_estimate)

    @staticmethod
    def adjust_prec(prec):
        return jnp.diag(prec)

    @property
    def true_curv(self):
        return jnp.diag(jnp.diag(self.arr))


class FullCurvatureTask(CurvatureTask):
    method = "full"

    @staticmethod
    def adjust_curv_estimate(curv_estimate):
        return curv_estimate

    @staticmethod
    def adjust_prec(prec):
        return prec

    @property
    def true_curv(self):
        return self.arr


@pytest_cases.parametrize(
    "task_class",
    [
        LANCZOSLowRankCurvatureTask,
        LOBPCGLowRankCurvatureTask,
        DiagonalCurvatureTask,
        FullCurvatureTask,
    ],
)
@pytest_cases.parametrize("seed", [0, 42])
def case_posterior_covariance(task_class, seed):
    return task_class(seed=seed)
