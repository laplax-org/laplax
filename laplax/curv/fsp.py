# fsp.py

"""FSP (Function-Space Prior) inference module with support for different kernel structures."""

from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger

import laplax
from laplax.curv.cov import Posterior, PosteriorState
from laplax.curv.ggn import compute_ggn_quadratic_form
from laplax.curv.lanczos import lanczos_invert_sqrt
from laplax.curv.utils import (
    LowRankTerms,
    compute_posterior_truncation_index,
    create_model_jvp,
)
from laplax.enums import CovarianceStructure
from laplax.types import (
    Callable,
    InputArray,
    Int,
    Kwargs,
    ModelFn,
    Params,
    PredArray,
)
from laplax.util import mv as util_mv
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
)

KernelStructure = CovarianceStructure


# ==============================================================================
# Helper functions
# ==============================================================================


def _truncated_left_svd(M_flat: jnp.ndarray):
    """Compute truncated left SVD (U, s) of a matrix efficiently.

    For speed, uses an eigen decomposition of the smaller Gram matrix
    and reconstructs the left singular vectors when advantageous.

    Returns
    -------
    tuple (U, s)
        U: left singular vectors with only columns above tolerance
        s: corresponding singular values (descending)
    """
    d, r = M_flat.shape
    tol = jnp.finfo(M_flat.dtype).eps ** 2

    if d > r:
        gram = M_flat.T @ M_flat  # (r, r)
        eigvals, V = jnp.linalg.eigh(gram)
        order = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V = V[:, order]
        s_all = jnp.sqrt(jnp.clip(eigvals, 0.0))
        mask = s_all > tol
        s = s_all[mask]
        if s.size == 0:
            return jnp.zeros((d, 0), dtype=M_flat.dtype), s
        V = V[:, : s.size]
        U = M_flat @ V
        U = U / s  # column-wise divide via broadcasting
        return U, s
    else:
        gram = M_flat @ M_flat.T  # (d, d)
        eigvals, U = jnp.linalg.eigh(gram)
        order = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        U = U[:, order]
        s_all = jnp.sqrt(jnp.clip(eigvals, 0.0))
        mask = s_all > tol
        s = s_all[mask]
        if s.size == 0:
            return jnp.zeros((d, 0), dtype=M_flat.dtype), s
        U = U[:, : s.size]
        return U, s


def _accumulate_M_over_chunks(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    k_inv_sqrt_dense: jnp.ndarray,
    n_chunks: int,
    *,
    mode: str = "map",
):
    """Accumulate _M_batch over context chunks using a chosen reduction mode.

    Modes:
    - "vmap": vectorized map over stacked chunks, then sum the results
    - "map": lax.map over chunk pairs, then sum the results
    - "scan": lax.scan accumulating the sum (default, typically fastest/memory‑lean)
    """
    x_chunks = jnp.split(x_context, n_chunks, axis=0)
    k_chunks = jnp.split(k_inv_sqrt_dense, n_chunks, axis=0)

    if mode == "vmap":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        M_chunks = jax.vmap(partial(_M_batch, model_fn, params), in_axes=(0, 0))(
            x_stacked, k_stacked
        )
        return jax.tree.map(lambda x: x.sum(axis=0), M_chunks)

    if mode == "map":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        pairs = (x_stacked, k_stacked)
        M_chunks = jax.lax.map(
            lambda pair: _M_batch(model_fn, params, pair[0], pair[1]), pairs
        )
        return jax.tree.map(lambda x: x.sum(axis=0), M_chunks)

    if mode == "scan":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        init_M = _M_batch(model_fn, params, x_stacked[0], k_stacked[0])

        def scan_fn(carry, pair):
            x_c, k_c = pair
            M_chunk = _M_batch(model_fn, params, x_c, k_c)
            return laplax.util.tree.add(carry, M_chunk), None

        M, _ = jax.lax.scan(scan_fn, init_M, (x_stacked[1:], k_stacked[1:]))
        return M

    msg = f"Unknown chunk accumulation mode: {mode}"
    raise ValueError(msg)


@partial(jax.jit, static_argnames=("model_fn",))
def _model_jvp(
    model_fn: ModelFn, params: Params, xs: InputArray, vs: Params
) -> PredArray:
    """Compute multiple Jacobian-vector products of the model.

    res[b] == jvp(f(xs[b], :), vs)

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : jnp.array with shape `(B,) + input_shape`
        Primals
    vs : Params
        Tangent vectors (pytree matching params structure)

    Returns
    -------
    jnp.array with shape `(B,) + output_shape`
        Batch of Jacobian-vector products
    """
    return jax.vmap(
        lambda x: jax.jvp(lambda w: model_fn(x, w), (params,), (vs,))[1],
        in_axes=0,
        out_axes=0,
    )(xs)


def _model_vjp(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    vs: PredArray,
    *,
    batch_axis: int = 0,
    output_batch_axis: int = 0,
) -> jax.Array:
    return jax.vmap(
        lambda x, v: jax.vjp(lambda w: model_fn(x, w), params)[1](v)[0],
        in_axes=(batch_axis, output_batch_axis),
        out_axes=output_batch_axis,
    )(xs, vs)


@partial(jax.jit, static_argnames=("model_fn",))
def _M_batch(model_fn: ModelFn, params: Params, xs: InputArray, L: PredArray):
    """Compute batched matrix-Jacobian product.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : InputArray
        Input data
    L : PredArray
        Matrix to multiply with Jacobian

    Returns
    -------
    Pytree
        Batched matrix-Jacobian product
    """

    def process_single_vs(vs):
        vjp_result = _model_vjp(model_fn, params, xs, vs)
        return jax.tree.map(lambda param: jnp.sum(param, axis=0), vjp_result)

    L_transposed = jnp.moveaxis(L, -1, 0)
    result = jax.lax.map(process_single_vs, L_transposed)

    return jax.tree.map(lambda x: jnp.moveaxis(x, 0, -1), result)


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def _lanczos_init(model_fn: ModelFn, params: Params, xs: InputArray, num_chunks: Int):
    """Initialize Lanczos vectors using HOSVD.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : Array
        Input data with shape (B, S1, S2, ..., C)
    num_chunks : int
        Number of chunks for processing

    Returns
    -------
    tuple
        (initial_vectors_function, initial_vectors_spatial)
    """  # noqa: DOC501
    if xs.ndim < 4:
        msg = f"Input must have shape (B, S1, S2, ..., C), but got {xs.shape}"
        raise ValueError(msg)

    ones_pytree = jax.tree.map(jnp.ones_like, params)

    model_jvp = jax.vmap(
        lambda x: jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1],
        in_axes=0,
        out_axes=0,
    )

    b = jnp.concatenate(
        [model_jvp(xs_batch) for xs_batch in jnp.split(xs, num_chunks, axis=0)],
        axis=0,
    )

    spatial_dims = tuple(s for s in xs.shape[1:-2] if s > 1)
    n_function = xs.shape[0]
    b = b.reshape((n_function,) + spatial_dims)

    initial_vectors = []

    for mode in range(len(b.shape)):
        n_mode = b.shape[mode]
        b_unfolded = jnp.moveaxis(b, mode, 0).reshape(n_mode, -1)

        u, _s, _v = jnp.linalg.svd(b_unfolded, full_matrices=False)
        vec = u[:, 0] / jnp.linalg.norm(u[:, 0])
        initial_vectors.append(vec)

    initial_vectors_function = [initial_vectors[0]]
    initial_vectors_spatial = initial_vectors[1:]

    return initial_vectors_function, initial_vectors_spatial


# ==============================================================================
# Kernel structure specific implementations
# ==============================================================================


def _lanczos_kronecker_structure(
    kernels_list: list[Callable],
    initial_vectors: list[jnp.ndarray],
    max_iters: list[int] | None = None,
):
    """Compute Lanczos inverse sqrt factor for Kronecker structured kernel.

    Parameters
    ----------
    kernels_list : list[Callable]
        List of kernel functions for each dimension
    initial_vectors : list[jnp.ndarray]
        Initial vectors for Lanczos for each dimension
    max_iters : list[int] | None
        Maximum iterations for each dimension

    Returns
    -------
    list
        List of Lanczos inverse sqrt factors for each dimension
    """
    if max_iters is None:
        max_iters = [None] * len(kernels_list)

    lanczos_results = []
    for kernel, init_vec, max_iter in zip(
        kernels_list, initial_vectors, max_iters, strict=False
    ):
        kwargs = {}
        if max_iter is not None:
            kwargs["max_iter"] = max_iter

        result = lanczos_invert_sqrt(kernel, init_vec, **kwargs)
        lanczos_results.append(result)

    return lanczos_results


def _lanczos_none_structure(
    kernel: Callable,
    initial_vector: jnp.ndarray,
    max_iter: int | None = None,
):
    """Compute Lanczos inverse sqrt factor for unstructured kernel.

    Parameters
    ----------
    kernel : Callable
        Kernel function or matrix
    initial_vector : jnp.ndarray
        Initial vector for Lanczos
    max_iter : int | None
        Maximum iterations

    Returns
    -------
    Array
        Lanczos inverse sqrt factor
    """
    kwargs = {}
    if max_iter is not None:
        kwargs["max_iter"] = max_iter

    return lanczos_invert_sqrt(kernel, initial_vector, **kwargs)


# ==============================================================================
# Main inference functions
# ==============================================================================


def create_fsp_posterior_kronecker(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    spatial_kernels: list[Callable],
    function_kernels: list[Callable],
    prior_variance: jnp.ndarray,
    n_chunks: int,
    *,
    spatial_max_iters: list[int] | None = [8, 3],
    is_classification: bool = False,
    chunk_mode: str = "scan",
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with Kronecker structured prior.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    spatial_kernels : list[Callable]
        List of spatial kernel functions
    function_kernels : list[Callable]
        List of function kernel functions
    prior_variance : jnp.ndarray
        Prior variance
    n_chunks : int
        Number of chunks for processing
    spatial_max_iters : list[int] | None
        Maximum Lanczos iterations for each spatial dimension

    Returns
    -------
    Posterior
        FSP posterior approximation
    """
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count to evenly divide number of functions
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))

    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    dim = sum(x.size for x in jax.tree_util.tree_leaves(params))

    initial_vectors_function, initial_vectors_spatial = _lanczos_init(
        model_fn, params, x_context, num_chunks=n_chunks_eff
    )

    spatial_lanczos_results = _lanczos_kronecker_structure(
        spatial_kernels, initial_vectors_spatial, spatial_max_iters
    )

    function_lanczos_results = _lanczos_kronecker_structure(
        function_kernels, initial_vectors_function, max_iters=None
    )

    def make_mv(matrix):
        return lambda v: matrix @ v

    # Combine all Kronecker factors (spatial + function)
    all_factors = spatial_lanczos_results + function_lanczos_results
    all_mvs = [make_mv(factor) for factor in all_factors]
    # Use column dimension (shape[1]) since MVP input size = number of columns
    all_layouts = [factor.shape[1] for factor in all_factors]

    # Create efficient Kronecker MVP (lazy, no intermediate densification)
    k_inv_sqrt_mv = util_mv.kronecker_product_factors(all_mvs, all_layouts)
    rank = int(jnp.prod(jnp.array(all_layouts)))

    def get_column(idx):
        """Extract column idx from Kronecker MVP without densifying full matrix."""
        return util_mv.column(k_inv_sqrt_mv, layout=rank, idx=idx, mv_jittable=False)

    # Use lax.map (sequential) instead of vmap to avoid constant capture
    k_inv_sqrt_cols = jax.lax.map(get_column, jnp.arange(rank))  # (rank, N_out)
    k_inv_sqrt_dense = k_inv_sqrt_cols.T.reshape(n_functions, *output_shape[1:], rank)

    # Use original _accumulate_M_over_chunks (processes all columns at once)
    M = _accumulate_M_over_chunks(
        model_fn,
        params,
        x_context,
        k_inv_sqrt_dense,
        n_chunks_eff,
        mode="map",
    )

    # Flatten M
    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Truncated left SVD
    _u, s = _truncated_left_svd(M_flat)

    # Unflatten U to pytree (with trailing rank dim k)
    u = unflatten(_u)
    uTggnu = compute_ggn_quadratic_form(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        U=u,
        is_classification=is_classification,
        regression_noise_scale=regression_noise_scale,
        col_chunk_size=ggn_col_chunk_size,
    )

    # Compute U_A, D_A
    A_eigh = jnp.diag(s**2) + uTggnu
    eigvals, eigvecs = jnp.linalg.eigh(A_eigh)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Compute S: $S = U_M U_A D_A^\dagger$
    cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    truncation_idx = compute_posterior_truncation_index(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        cov_sqrt=cov_sqrt,
        prior_variance=prior_variance,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    parameter_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"FSP posterior using {truncation_idx} / {parameter_count} components.")
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    # Create flatten/unflatten for posterior
    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=posterior_state["scale_sqrt"].shape[-1],
        low_rank_terms=low_rank_terms,
    )

    logger.info(
        f"Created FSP posterior with rank {posterior.rank} and shape {M_flat.shape}"
    )

    return posterior


def create_fsp_posterior_none(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel: Callable,
    prior_variance: jnp.ndarray,
    n_chunks: int,
    *,
    max_iter: int | None = None,
    is_classification: bool = False,
    independent_outputs: bool = False,
    kernels_per_output: list[Callable] | None = None,
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with unstructured prior (full covariance Lanczos).

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    kernel : Callable
        Full kernel function
    prior_variance : jnp.ndarray
        Prior variance
    n_chunks : int
        Number of chunks for processing
    max_iter : int | None
        Maximum Lanczos iterations

    Returns
    -------
    Posterior
        FSP posterior approximation
    """
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    dim = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # Initialize with ones (simple initialization for unstructured case)
    ones_pytree = jax.tree.map(lambda x: jnp.ones_like(x), params)
    model_jvp = create_model_jvp(params, ones_pytree, model_fn, in_axes=0, out_axes=0)
    b = model_jvp(x_context)

    # Compute Lanczos inverse sqrt for the prior kernel
    if independent_outputs or kernels_per_output is not None:
        # Build block-diagonal K^{-1/2} across output channels using per-output kernels
        output_dim = 1 if b.ndim == 1 else int(b.shape[-1])
        per_output_cols = []
        for k in range(output_dim):
            b_k = b if output_dim == 1 else b[:, k]
            b_k = b_k.reshape(-1)
            init_k = b_k / (jnp.linalg.norm(b_k) + 1e-12)
            kernel_k = (
                kernels_per_output[k]
                if kernels_per_output is not None and k < len(kernels_per_output)
                else kernel
            )
            k_inv_sqrt_k = _lanczos_none_structure(kernel_k, init_k, max_iter)
            per_output_cols.append(k_inv_sqrt_k)

        ranks = [c.shape[-1] for c in per_output_cols]
        total_rank = int(sum(ranks))
        B = int(x_context.shape[0])
        out_dim = 1 if b.ndim == 1 else int(b.shape[-1])
        k_inv_sqrt_dense = jnp.zeros((B, out_dim, total_rank), dtype=b.dtype)
        offset = 0
        for k, cols in enumerate(per_output_cols):
            r = int(cols.shape[-1])
            k_inv_sqrt_dense = k_inv_sqrt_dense.at[:, k, offset : offset + r].set(
                cols.reshape(B, r)
            )
            offset += r
    else:
        initial_vector = b.flatten() / (jnp.linalg.norm(b.flatten()) + 1e-12)
        k_inv_sqrt = _lanczos_none_structure(kernel, initial_vector, max_iter)

        rank = k_inv_sqrt.shape[-1]
        k_inv_sqrt_dense = k_inv_sqrt.reshape(*output_shape, rank)

    M = _accumulate_M_over_chunks(
        model_fn,
        params,
        x_context,
        k_inv_sqrt_dense,
        n_chunks_eff,
        mode="map",  # Use map for memory efficiency (lax.map over chunks)
    )

    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Truncated left SVD
    _u, s = _truncated_left_svd(M_flat)

    u = unflatten(_u)

    uTggnu = compute_ggn_quadratic_form(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        U=u,  # Pass U here!
        is_classification=is_classification,
        regression_noise_scale=regression_noise_scale,
        col_chunk_size=ggn_col_chunk_size,
    )

    # Compute U_A, D_A
    A_eigh = jnp.diag(s**2) + uTggnu
    eigvals, eigvecs = jnp.linalg.eigh(A_eigh)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Compute S
    cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    truncation_idx = compute_posterior_truncation_index(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        cov_sqrt=cov_sqrt,
        prior_variance=prior_variance,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    parameter_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"FSP posterior using {truncation_idx} / {parameter_count} components.")
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            (state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x)))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=posterior_state["scale_sqrt"].shape[-1],
        low_rank_terms=low_rank_terms,
    )

    logger.info(
        f"Created FSP posterior with rank {posterior.rank} and shape {M_flat.shape}"
    )

    return posterior


def create_fsp_posterior(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel_structure: CovarianceStructure | str,
    n_chunks: int,
    *,
    kernel: Callable | None = None,
    spatial_kernels: list[Callable] | None = None,
    function_kernels: list[Callable] | None = None,
    prior_variance: jnp.ndarray | None = None,
    spatial_max_iters: list[int] | None = None,
    max_iter: int | None = None,
    is_classification: bool = False,
    independent_outputs: bool = False,
    kernels_per_output: list[Callable] | None = None,
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with specified kernel structure.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    kernel_structure : KernelStructure
        Type of kernel structure ('kronecker' or 'none')
    n_chunks : int
        Number of chunks for processing
    kernel : Callable | None
        Full kernel for 'none' structure
    spatial_kernels : list[Callable] | None
        Spatial kernels for 'kronecker' structure
    function_kernels : list[Callable] | None
        Function kernels for 'kronecker' structure
    prior_variance : jnp.ndarray | None
        Prior variance (computed if not provided)
    spatial_max_iters : list[int] | None
        Max iterations for spatial Lanczos
    max_iter : int | None
        Max iterations for full Lanczos

    Returns
    -------
    Posterior
        FSP posterior approximation

    Raises
    ------
    ValueError
        If required arguments for the specified kernel structure are missing
    """  # noqa: DOC501
    if (
        kernel_structure == CovarianceStructure.KRONECKER
        or str(kernel_structure) == "kronecker"
    ):
        if spatial_kernels is None or function_kernels is None:
            msg = (
                "spatial_kernels and function_kernels must be provided "
                "for Kronecker structure"
            )
            raise ValueError(msg)

        return create_fsp_posterior_kronecker(
            model_fn=model_fn,
            params=params,
            x_context=x_context,
            spatial_kernels=spatial_kernels,
            function_kernels=function_kernels,
            prior_variance=prior_variance,
            n_chunks=n_chunks,
            spatial_max_iters=spatial_max_iters,
            is_classification=is_classification,
            regression_noise_scale=regression_noise_scale,
            ggn_col_chunk_size=ggn_col_chunk_size,
            **kwargs,
        )

    elif (
        kernel_structure == CovarianceStructure.NONE or str(kernel_structure) == "none"
    ):
        if kernel is None:
            msg = "kernel must be provided for None structure"
            raise ValueError(msg)

        return create_fsp_posterior_none(
            model_fn=model_fn,
            params=params,
            x_context=x_context,
            kernel=kernel,
            prior_variance=prior_variance,
            n_chunks=n_chunks,
            max_iter=max_iter,
            is_classification=is_classification,
            independent_outputs=independent_outputs,
            kernels_per_output=kernels_per_output,
            regression_noise_scale=regression_noise_scale,
            ggn_col_chunk_size=ggn_col_chunk_size,
            **kwargs,
        )

    else:
        msg = f"Unknown kernel structure: {kernel_structure}"
        raise ValueError(msg)


# Public mapping similar to CURVATURE_PRECISION_METHODS
COVARIANCE_STRUCTURE_METHODS: dict[CovarianceStructure | str, Callable] = {
    CovarianceStructure.KRONECKER: create_fsp_posterior_kronecker,
    CovarianceStructure.NONE: create_fsp_posterior_none,
}
