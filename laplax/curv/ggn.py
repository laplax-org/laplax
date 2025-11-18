# ggn.py

"""Generalized Gauss-Newton matrix-vector product and loss hessian."""

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from laplax.curv.hessian import hvp
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    Float,
    InputArray,
    Int,
    Kwargs,
    ModelFn,
    Num,
    Params,
    PredArray,
    TargetArray,
)
from laplax.util.tree import mul


def _binary_cross_entropy_hessian_mv(
    jv: PredArray,
    pred: PredArray,
    **kwargs: Kwargs,
) -> Num[Array, "..."]:
    r"""Compute the Hessian-vector product for the binary cross-entropy loss.

    This calculation uses the predicted sigmoid probabilities to compute the
    1x1 Hessian. The result is the product of the predicted probabilities for the
    positive and the negative class.

    Mathematically, the Hessian-vector product is computed as:

    $$
    H \cdot jv = p(1-p) \cdot jv,
    $$

    where $p = \text{sigmoid}(\text{pred})$.

    Args:
        jv: Vector to multiply with the Hessian.
        pred: Model predictions (logits).
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for cross-entropy loss.
    """
    del kwargs
    prob = jax.nn.sigmoid(pred)
    return prob * (1 - prob) * jv


def _cross_entropy_hessian_mv(
    jv: PredArray,
    pred: PredArray,
    **kwargs: Kwargs,
) -> Num[Array, "..."]:
    r"""Compute the Hessian-vector product for the cross-entropy loss.

    This calculation uses the predicted softmax probabilities to compute the
    diagonal and off-diagonal components of the Hessian. The result is the difference
    between the diagonal contribution and the off-diagonal contribution of the Hessian.

    Mathematically, the Hessian-vector product is computed as:

    $$
    H \cdot jv = \text{diag}(p) \cdot jv - p \cdot (p^\top \cdot jv),
    $$

    where $p = \text{softmax}(\text{pred})$.

    Args:
        jv: Vector to multiply with the Hessian.
        pred: Model predictions (logits).
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for cross-entropy loss.
    """
    del kwargs
    prob = jax.nn.softmax(pred)
    off_diag_jv = prob * (prob.reshape(1, -1) @ jv)
    diag_jv = prob * jv
    return diag_jv - off_diag_jv


def _mse_hessian_mv(
    jv: PredArray,
    **kwargs: Kwargs,
) -> PredArray:
    r"""Compute the Hessian-vector product for mean squared error loss.

    The Hessian of the mean squared error loss is a constant diagonal matrix with
    2 along the diagonal. Thus, the Hessian-vector product is simply 2 times the
    input vector.

    Mathematically, the Hessian-vector product is computed as:

    $$
    H \cdot jv = 2 \cdot jv,
    $$

    Args:
        jv: Vector to multiply with the Hessian.
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for MSE loss.
    """
    del kwargs
    return 2 * jv


def create_loss_hessian_mv(
    loss_fn: LossFn
    | str
    | Callable[[PredArray, TargetArray], Num[Array, "..."]]
    | None,
    **kwargs: Kwargs,
) -> Callable:
    r"""Create a function to compute the Hessian-vector product for a specified loss fn.

    For predefined loss functions like cross-entropy and mean squared error, the
    function computes their corresponding Hessian-vector products using efficient
    formulations. For custom loss functions, the Hessian-vector product is computed via
    automatic differentiation.

    Args:
        loss_fn: Loss function to compute the Hessian-vector product for. Supported
            options are:

            - `LossFn.BINARY_CROSS_ENTROPY` for binary cross-entropy loss.
            - `LossFn.CROSS_ENTROPY` for cross-entropy loss.
            - `LossFn.MSE` for mean squared error loss.
            - `LossFn.NONE` for no loss.
            - A custom callable loss function that takes predictions and targets.

        **kwargs: Unused keyword arguments.

    Returns:
        A function that computes the Hessian-vector product for the given loss function.

    Raises:
        ValueError: When `loss_fn` is `None`.
        ValueError: When an unsupported loss function (not of type: `Callable`)is
            provided.
    """
    del kwargs

    if loss_fn is None:
        msg = "loss_fn cannot be None"
        raise ValueError(msg)

    if loss_fn == LossFn.BINARY_CROSS_ENTROPY:
        return _binary_cross_entropy_hessian_mv

    if loss_fn == LossFn.CROSS_ENTROPY:
        return _cross_entropy_hessian_mv

    if loss_fn == LossFn.MSE:
        return _mse_hessian_mv

    if loss_fn == LossFn.NONE:

        def _identity(
            jv: PredArray,
            pred: PredArray,
            target: TargetArray,
            **kwargs,
        ) -> Num[Array, "..."]:
            del pred, target, kwargs
            return jv

        return _identity

    if isinstance(loss_fn, Callable):

        def custom_hessian_mv(
            jv: PredArray,
            pred: PredArray,
            target: TargetArray,
            **kwargs,
        ) -> Num[Array, "..."]:
            del kwargs

            def loss_fn_local(p):
                return loss_fn(p, target)

            return hvp(loss_fn_local, pred, jv)

        return custom_hessian_mv

    msg = "unsupported loss function provided"
    raise ValueError(msg)


# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("model_fn",))
def _jmp(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    u: Params,
) -> PredArray:
    u_rank_major = jax.tree.map(lambda leaf: jnp.moveaxis(leaf, -1, 0), u)

    def jmp_single_x(x_c: InputArray) -> PredArray:
        """J(x_c) U_M for one context point.

        Returns:
            Jacobian-matrix product for a single context point.
        """

        def jvp_single_rank(u_single: Params) -> PredArray:
            _, tang = jax.jvp(
                lambda p: model_fn(x_c, p),
                (params,),
                (u_single,),
            )
            return tang

        res_rank_major = jax.lax.map(jvp_single_rank, u_rank_major)
        return jnp.moveaxis(res_rank_major, 0, -1)

    return jax.lax.map(jmp_single_x, x_context)


@partial(jax.jit, static_argnames=("model_fn",))
def _jmp_fast(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    u: Params,
) -> PredArray:
    u_rank_major = jax.tree.map(lambda leaf: jnp.moveaxis(leaf, -1, 0), u)

    def jmp_single_x(x_c: InputArray) -> PredArray:
        return jax.vmap(
            lambda u_single: jax.jvp(
                lambda p: model_fn(x_c, p),
                (params,),
                (u_single,),
            )[1],
            in_axes=0,
            out_axes=-1,
        )(u_rank_major)

    return jax.lax.map(jmp_single_x, x_context)


def create_jmp(
    model_fn: ModelFn,
    *,
    vmap_over_data: bool,
) -> Callable[[Params, InputArray, Params], PredArray]:
    """Create a JVP-based Jacobian-matrix product specialized to `model_fn`.

    Returns:
        Function that computes Jacobian-matrix products for the given model.
    """
    jmp_impl = _jmp_fast if vmap_over_data else _jmp
    return partial(jmp_impl, model_fn=model_fn)


def create_ggn_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    loss_hessian_mv: Callable | None = None,
    fsp: bool = False,
) -> Callable[[Params, Data], Params]:
    r"""Create Generalized Gauss-Newton (GGN) matrix-vector productwithout fixed data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. The resulting product is given by:

    $$
    \text{factor} \cdot \sum_i J_i^\top \nabla^2_{f(x_i, \theta), f(x_i, \theta)}
    \mathcal{L}(f(x_i, \theta), y_i) J_i \cdot v
    $$

    where $J_i$ is the Jacobian of the model at data point $i$, $H_{L, i}$ is the
    Hessian of the loss, and $v$ is the vector. The `factor` is a scaling factor that
    is used to scale the GGN matrix.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_hessian_mv: The loss Hessian matrix-vector product.
        fsp: Whether to use FSP (Function-Space Posterior) mode. Defaults to False.

    Returns:
        A function that takes a vector and a batch of data, and computes the GGN
        matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.

    """
    # Create loss Hessian-vector product
    loss_hessian_mv = loss_hessian_mv or create_loss_hessian_mv(loss_fn)

    if vmap_over_data:
        loss_hessian_mv = jax.vmap(loss_hessian_mv)

    def ggn_mv(vec, data):
        # Step 1: Single jvp for entire batch, if vmap_over_data is True
        def fwd(p):
            if vmap_over_data:
                return jax.vmap(lambda x: model_fn(x, p))(data["input"])
            return model_fn(data["input"], p)

        # Step 2: Linearize the forward pass
        z, jvp = jax.linearize(fwd, params)

        # Step 3: Compute J^T H J v
        HJv = loss_hessian_mv(jvp(vec), pred=z, target=data["target"])

        # Step 4: Compute the GGN vector
        arr = jax.linear_transpose(jvp, vec)(HJv)[0]

        return mul(factor, arr)

    jmp = create_jmp(model_fn=model_fn, vmap_over_data=vmap_over_data)

    def ggn_fsp_mv(vec, data):
        x_context = data["context"]
        ju = jmp(
            params=params,
            x_context=x_context,
            u=vec,
        )
        batch_size = ju.shape[0]
        rank = ju.shape[-1]

        preds = jax.vmap(lambda x: model_fn(x, params))(x_context)

        ju_rank_major = jnp.moveaxis(ju, -1, 0)

        def apply_loss_hessian(jv_single):
            return loss_hessian_mv(jv_single, pred=preds, target=data["target"])

        hju_rank_major = jax.vmap(apply_loss_hessian)(ju_rank_major)
        hju = jnp.moveaxis(hju_rank_major, 0, -1)

        ju_flat = ju.reshape(batch_size, -1, rank)
        hju_flat = hju.reshape(batch_size, -1, rank)
        gram = jnp.einsum("bji,bjk->ik", ju_flat, hju_flat)

        return mul(factor, gram)

    return ggn_mv if not fsp else ggn_fsp_mv


def create_ggn_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable | None = None,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    loss_hessian_mv: Callable | None = None,
    fsp: bool = False,
) -> Callable[[Params], Params]:
    r"""Computes the Generalized Gauss-Newton (GGN) matrix-vector product with data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. For a given dataset, the GGN matrix-vector product is computed as:

    $$
    G(\theta) = \text{factor} \sum_{i=1}^N J_i^\top \nabla^2_{f(x_i, \theta), f(x_i,
    \theta)} \mathcal{L}_i(f(x_i, \theta), y_i) J_i \cdot v
    $$

    where $J_i$ is the Jacobian of the model for the $i$-th data point, $\nabla^2_{
    f(x, \theta), f(x, \theta)}\mathcal{L}_i(f(x_i, \theta), y_i)$ is the Hessian of
    the loss for the $i$-th data point, and $N$ is the number of data points. The
    `factor` is a scaling factor that is used to scale the GGN matrix.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset
    remains fixed.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data, for fsp "context".
        loss_fn: Loss function to use for the GGN computation.
        num_curv_samples: Number of samples used to calculate the GGN. Defaults to None,
            in which case it is inferred from `data` as its batch size. Note that for
            losses that contain sums even for a single input (e.g., pixel-wise semantic
            segmentation losses, this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_ggn_samples`'s description. Defaults to None, in which case
            it is set to equal `num_ggn_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_hessian_mv: The loss Hessian matrix-vector product. If not provided, it is
            computed using the `loss_fn`.
        fsp: Whether to use FSP (Function-Space Posterior) mode. Defaults to False.

    Returns:
        A function that takes a vector and computes the GGN matrix-vector product for
            the given data.

    Raises:
        ValueError: If both `loss_fn` and `loss_hessian_mv` are provided.
        ValueError: If neither `loss_fn` nor `loss_hessian_mv` are provided.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    # Enforce either loss_fn or loss_hessian_mv must be provided:
    if loss_fn is None and loss_hessian_mv is None:
        msg = "Either loss_fn or loss_hessian_mv must be provided."
        raise ValueError(msg)

    if loss_fn is not None and loss_hessian_mv is not None:
        msg = "Only one of loss_fn or loss_hessian_mv must be provided."
        raise ValueError(msg)

    if num_curv_samples is None:
        num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        vmap_over_data=vmap_over_data,
        loss_hessian_mv=loss_hessian_mv,
        fsp=fsp,
    )

    def wrapped_ggn_mv(vec: Params) -> Params:
        return ggn_mv(vec, data)

    return wrapped_ggn_mv
