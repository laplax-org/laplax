import jax
import jax.numpy as jnp

from laplax.types import Data, ModelFn, Params, PredArray
from laplax.util.flatten import create_pytree_flattener


def _compute_curvature_fn(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    ggn: PredArray,
    prior_var: PredArray,
    u: PredArray,
):
    """Compute the curvature function of FSP Laplace.

    Can be used to test the compute_curvature_fn in fsp.

    Args:
        model_fn: Model function mapping inputs and parameters to outputs.
        params: Parameters of the model.
        data: Dataset dictionary containing at least ``\"test_inputs\"``.
        ggn: Generalized Gauss-Newton matrix.
        prior_var: Prior variance vector.
        u: Direction vectors used for the curvature computation.

    Returns:
        jax.Array: Curvature factors stacked along the last axis.
    """  # noqa: D205
    eigvals_, eigvecs_ = jnp.linalg.eigh(ggn)
    eps = jnp.finfo(ggn.dtype).eps
    tol = eps * (eigvals_.max() ** 0.5) * eigvals_.shape[0]
    eigvals = eigvals_[eigvals_ > tol]
    eigvecs = eigvecs_[:, eigvals_ > tol]
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    def normalize_eigvecs(evals, u, evecs):
        return u @ (evecs * (1 / jnp.sqrt(evals)))

    def jvp(x, v):
        return jax.jvp(lambda p: model_fn(x, params=p), (params,), (v,))[1]

    i = 0
    post_var = jnp.zeros((prior_var.shape[0],))
    jnp.sum(prior_var)
    cov_sqrt = []
    _, unflatten = create_pytree_flattener(params)
    while jnp.all(post_var < prior_var) and i < eigvals.shape[0]:
        cov_sqrt += [jax.jit(normalize_eigvecs)(eigvals[i], u, eigvecs[:, i])]
        lr_fac_i = jax.jit(unflatten)(cov_sqrt[-1])
        post_var += jnp.concatenate(
            [jax.jit(jvp)(x_c, lr_fac_i) ** 2 for x_c in data["test_inputs"]], axis=0
        )
        i += 1

    truncation_idx = i if i == eigvals.shape[0] else i - 1
    return jnp.stack(cov_sqrt[:truncation_idx], axis=-1)
