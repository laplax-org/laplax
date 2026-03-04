# /laplax/curv/fsp.py

"""Function Space Prior (FSP) Laplace Approximation."""

from functools import partial

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.cov import Posterior
from laplax.curv.ggn import _create_loss_fn, create_ggn_fsp_operator_without_data
from laplax.curv.lanczos import lanczos_inverse_sqrt_factor
from laplax.curv.utils import LowRankTerms
from laplax.enums import LossFn, LowRankMethod
from laplax.types import (
    Array,
    Callable,
    Float,
    InputArray,
    Iterable,
    KeyType,
    Kwargs,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
    PriorArguments,
)
from laplax.util.context_points import select_context_points
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
    wrap_factory,
)
from laplax.util.tree import ones_like, to_dtype

# -----------------------------------------------------------------------------
# FSP Curvature creation
# -----------------------------------------------------------------------------


def _fsp_state_to_scale(state: PosteriorState) -> Callable[[Array], Array]:
    """Convert posterior state to scale matrix-vector product.

    Returns:
        Function that computes scale matrix-vector product.
    """
    s = state["scale_sqrt"]

    def mv(v: Array) -> Array:
        return s @ v

    return mv


def _fsp_state_to_cov(state: PosteriorState) -> Callable[[Array], Array]:
    """Convert posterior state to covariance matrix-vector product.

    Returns:
        Function that computes covariance matrix-vector product.
    """
    s = state["scale_sqrt"]

    def mv(v: Array) -> Array:
        return s @ (s.T @ v)

    return mv


def create_fsp_curvature(
    *,
    model_fn: ModelFn,
    params: Params,
    data: Iterable,
    loss_fn: LossFn | str | Callable,
    kernel_fn: Callable[[Array, Array], Array],
    low_rank_method: LowRankMethod,
    jitter: float,
    ggn_factor: float,
    n_context_points: int,
    context_selection: str,
    n_chunks: int,
    max_rank: int | None,
    **kwargs: Kwargs,
) -> LowRankTerms:
    """Compute FSP curvature (posterior covariance factor).

    Returns:
        Low-rank terms representing the posterior covariance factor.

    Raises:
        ValueError: If low_rank_method is not LANCZOS.
    """
    if low_rank_method is not LowRankMethod.LANCZOS:
        msg = f"Unsupported FSP algorithm: {low_rank_method}"
        raise ValueError(msg)

    params = to_dtype(params, dtype=jnp.float64)

    loss_fn_callable = _create_loss_fn(loss_fn)

    # 1. Select context points using utility
    x_context, _ = select_context_points(
        data=data,
        method=context_selection,
        n_context_points=n_context_points,
        **kwargs,
    )

    # 2. Output shape detection
    y0 = jax.vmap(lambda x: model_fn(input=x, params=params))(x_context[:2])
    output_shape = (x_context.shape[0], *y0.shape[1:])

    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))

    # 3. Compute Prior Inverse Sqrt (Lanczos)
    k_inv_sqrt_dense, _rank = _compute_full_prior_inverse_sqrt(
        kernel_fn=kernel_fn,
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        output_shape=output_shape,
        n_chunks_eff=n_chunks_eff,
        jitter=jitter,
    )

    # 4. Compute M matrix (Jacobian * D)
    M_flat = _compute_M_memory_efficient(
        model_fn, params, x_context, k_inv_sqrt_dense, n_chunks_eff
    )
    # k_inv_sqrt_dense consumed

    # 5. SVD of M
    u_svd, s_svd, _ = jnp.linalg.svd(M_flat, full_matrices=False)
    tol = jnp.finfo(M_flat.dtype).eps ** 0.5
    keep = s_svd > tol
    s = s_svd[keep]
    u = u_svd[:, keep]  # (P, R)

    # 6. GGN Projection: U^T G G N U
    _, project = create_ggn_fsp_operator_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn_callable,
        factor=ggn_factor,
        vmap_over_data=True,
    )

    uTggnu = jnp.zeros((u.shape[1], u.shape[1]))

    for batch in data:
        if isinstance(batch, (tuple, list)):
            batch_data = {"input": batch[0], "target": batch[1]}
        elif isinstance(batch, dict):
            batch_data = batch
        else:
            continue

        term = project(u, batch_data)
        uTggnu = uTggnu + term

    # 7. Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + uTggnu)

    keep_eigs = eigvals > 0
    eigvals = eigvals[keep_eigs]
    eigvecs = eigvecs[:, keep_eigs]

    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # 8. Trace-based truncation
    K_full = kernel_fn(x_context, x_context)
    prior_var = jnp.diag(K_full) + jitter
    # Reshape to (N, Out)
    prior_var = jax.vmap(lambda v: jnp.full(output_shape[1:], v))(prior_var)
    prior_var = prior_var.reshape(n_functions, -1)

    _, unravel_fn = create_pytree_flattener(params)

    max_rank = len(eigvals) if max_rank is None else min(max_rank, len(eigvals))

    f1 = jax.jit(lambda e, u_mat, v: u_mat @ (v * (1.0 / e**0.5)))

    def f2_fn(x, v):
        v_pytree = unravel_fn(v)
        _, out_tangent = jax.jvp(
            lambda p: model_fn(input=x, params=p), (params,), (v_pytree,)
        )
        return out_tangent.reshape(-1)

    f2 = jax.jit(f2_fn)

    i = 0
    post_var = jnp.zeros_like(prior_var)
    cov_sqrt_cols = []
    dim = u.shape[0]

    while i < max_rank:
        col = f1(eigvals[i], u, eigvecs[:, i])

        # Check trace condition (expensive but required by algo)
        # Use partial to properly bind col for vmap
        f2_with_col = partial(f2, v=col)
        delta_vecs = jax.vmap(f2_with_col)(x_context)
        new_post_var = post_var + delta_vecs**2

        if jnp.any(new_post_var > prior_var):
            break

        post_var = new_post_var
        cov_sqrt_cols.append(col)
        i += 1

    truncation_idx = i

    if truncation_idx == 0:
        cov_sqrt = jnp.zeros((dim, 1))
    else:
        cov_sqrt = jnp.stack(cov_sqrt_cols, axis=-1)

    # Final decomposition for LowRankTerms
    U_final, S_final, _ = jnp.linalg.svd(cov_sqrt, full_matrices=False)

    return LowRankTerms(U=U_final, S=S_final**2, scalar=0.0)


def create_fsp_posterior(
    *,
    model_fn: ModelFn,
    params: Params,
    data: Iterable,
    loss_fn: LossFn | str | Callable,
    key: KeyType,
    kernel_fn: Callable[[Array, Array], Array],
    low_rank_method: LowRankMethod = LowRankMethod.LANCZOS,
    jitter: float = 1e-4,
    ggn_factor: float = 1.0,
    n_context_points: int = 100,
    context_selection: str = "sobol",
    n_chunks: int = 4,
    max_rank: int | None = None,
) -> Callable[[PriorArguments, Float], Posterior]:
    """Factory function to create FSP posterior.

    This replaces the generic create_posterior_fn for FSP since FSP is not a CurvApprox.

    Returns:
        Function that creates a Posterior from prior arguments and loss scaling factor.
    """
    # 1. Estimate curvature (LowRankTerms)
    curv_estimate = create_fsp_curvature(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=loss_fn,
        key=key,
        kernel_fn=kernel_fn,
        low_rank_method=low_rank_method,
        jitter=jitter,
        ggn_factor=ggn_factor,
        n_context_points=n_context_points,
        context_selection=context_selection,
        n_chunks=n_chunks,
        max_rank=max_rank,
    )

    # 2. Setup flatten/unflatten
    flatten, unflatten = create_pytree_flattener(params)

    def posterior_fn(
        prior_arguments: PriorArguments,
        loss_scaling_factor: Float = 1.0,
    ) -> Posterior:
        del loss_scaling_factor

        # Apply prior precision scaling
        prior_prec = prior_arguments.get("prior_prec", 1.0)

        # Scale eigenvalues: S_new = S / prior_prec
        # Scale factor sqrt: 1 / sqrt(prior_prec)
        scale_sqrt = curv_estimate.U @ jnp.diag(jnp.sqrt(curv_estimate.S / prior_prec))

        state = {
            "scale_sqrt": scale_sqrt,
            "rank": int(scale_sqrt.shape[1]),
            # Optional: store full low_rank_terms if needed
        }

        return Posterior(
            state=state,
            cov_mv=wrap_factory(_fsp_state_to_cov, flatten, unflatten),
            scale_mv=wrap_factory(_fsp_state_to_scale, flatten, unflatten),
            rank=state["rank"],
            low_rank_terms=curv_estimate,
        )

    return posterior_fn


# -----------------------------------------------------------------------------
# Helper Functions (Internal)
# -----------------------------------------------------------------------------


def _create_concatenated_model_jvp(
    model_fn: ModelFn, params: Params, xs: InputArray, num_chunks: int
) -> Array:
    """Create concatenated model JVP output for initialization.

    Returns:
        Flattened concatenated JVP outputs.
    """
    ones_pytree = ones_like(params)

    def model_jvp(xs_batch: InputArray) -> PredArray:
        return jax.lax.map(
            lambda x: jax.jvp(
                lambda w: model_fn(input=x, params=w),
                (params,),
                (ones_pytree,),
            )[1],
            xs_batch,
            batch_size=1,
        )

    model_jvp = jax.jit(model_jvp)
    # Using array_split to handle uneven chunks
    xs_chunks = jnp.array_split(xs, num_chunks, axis=0)
    b_chunks = [model_jvp(xs_chunk) for xs_chunk in xs_chunks]

    res = jnp.concatenate(b_chunks, axis=0)
    return res.reshape(-1)


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def _lanczos_init_full(
    model_fn: ModelFn, params: Params, xs: InputArray, num_chunks: int
) -> Array:
    """Initialize Lanczos vector using parameters ones-vector JVP.

    Returns:
        Normalized Lanczos initialization vector.
    """
    b = _create_concatenated_model_jvp(model_fn, params, xs, num_chunks)
    b = b / jnp.linalg.norm(b)
    return b


@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def _model_vjp(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    vs: PredArray,
    batch_size: int = 1,
) -> Params:
    """Compute vector-Jacobian products.

    Returns:
        VJP results as parameter pytree.
    """

    def single_vjp(args: tuple[InputArray, PredArray]) -> Params:
        x, v = args
        _, vjp_fn = jax.vjp(lambda w: model_fn(input=x, params=w), params)
        return vjp_fn(v)[0]

    return jax.lax.map(single_vjp, (xs, vs), batch_size=batch_size)


@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def _M_batch(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    L: PredArray,
    batch_size: int = 8,
) -> Params:
    """Compute M = J^T L for a batch.

    Returns:
        M matrix as parameter pytree.
    """
    L_transposed = jnp.moveaxis(L, -1, 0)

    def process_rank(L_slice: PredArray) -> Params:
        vjp_result = _model_vjp(model_fn, params, xs, L_slice, batch_size=batch_size)
        return jax.tree.map(lambda param: jnp.sum(param, axis=0), vjp_result)

    result = jax.lax.map(process_rank, L_transposed, batch_size=batch_size)
    return jax.tree.map(lambda x: jnp.moveaxis(x, 0, -1), result)


def _compute_full_prior_inverse_sqrt(
    kernel_fn: Callable[[Array, Array], Array],
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    output_shape: tuple,
    n_chunks_eff: int,
    jitter: float = 1e-4,
) -> tuple[Array, int]:
    """Compute K^{-1/2} using Lanczos/CG.

    Returns:
        Tuple of (inverse sqrt covariance, rank).
    """
    init_vector = _lanczos_init_full(
        model_fn, params, x_context, num_chunks=n_chunks_eff
    )

    n_outputs = int(jnp.prod(jnp.array(output_shape[1:])))

    def matvec(v):
        # v: (N * n_outputs,)
        v_mat = v.reshape(x_context.shape[0], n_outputs)

        # Compute K @ v_mat using memory-efficient specialized operator
        res_rows = util.mv.kernel_mv(
            kernel_fn, x_context, x_context, v_mat, batch_size=64
        )

        # Add jitter
        res_rows = res_rows + jitter * v_mat

        return res_rows.reshape(-1)

    # Note: K is assumed symmetric, so we skip 0.5*(K+K.T) logic which
    # would require 2x compute.
    # If explicit symmetrization is needed, we would need a second map.

    b = init_vector
    inverse_sqrt_cov = lanczos_inverse_sqrt_factor(matvec, b, max_iter=200)
    rank = inverse_sqrt_cov.shape[-1]

    inverse_sqrt_cov = inverse_sqrt_cov.reshape(
        x_context.shape[0], *output_shape[1:], rank
    )

    return inverse_sqrt_cov, rank


def _compute_M_memory_efficient(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    k_inv_sqrt_dense: Array,
    n_chunks_eff: int,
) -> Array:
    """Compute M = J^T D.

    Returns:
        Flattened M matrix.
    """
    flatten, _ = create_partial_pytree_flattener(jax.tree.map(jnp.zeros_like, params))

    x_chunks = jnp.array_split(x_context, n_chunks_eff, axis=0)
    k_chunks = jnp.array_split(k_inv_sqrt_dense, n_chunks_eff, axis=0)

    first_chunk = _M_batch(model_fn, params, x_chunks[0], k_chunks[0])
    M_flat = flatten(first_chunk)

    remaining_chunks = list(zip(x_chunks[1:], k_chunks[1:], strict=False))

    for x_c, k_c in remaining_chunks:
        chunk = _M_batch(model_fn, params, x_c, k_c)
        chunk_flat = flatten(chunk)

        M_flat = M_flat + chunk_flat

    return M_flat
