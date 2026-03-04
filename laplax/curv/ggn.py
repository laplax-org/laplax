# /laplax/curv/ggn.py

from collections.abc import Callable

import jax
import jax.numpy as jnp

from laplax.curv.loss import fetch_loss_hessian_mv
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    Float,
    InputArray,
    Int,
    ModelFn,
    Num,
    Params,
    PredArray,
    TargetArray,
)
from laplax.util.flatten import create_pytree_flattener
from laplax.util.tree import mul

# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    loss_hessian_mv: Callable | None = None,
) -> Callable[[Params, Data], Params]:
    r"""Create Generalized Gauss-Newton (GGN) matrix-vector product without fixed data.

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

    Returns:
        A function that takes a vector and a batch of data, and computes the GGN
        matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.

    """
    # Create loss Hessian-vector product
    loss_hessian_mv = fetch_loss_hessian_mv(
        loss_fn, loss_hessian_mv, vmap_over_data=vmap_over_data
    )

    def ggn_mv(vec, data):
        # Step 1: Single jvp for entire batch, if vmap_over_data is True
        def fwd(p):
            if vmap_over_data:
                return jax.vmap(lambda x: model_fn(input=x, params=p))(data["input"])
            return model_fn(input=data["input"], params=p)

        # Step 2: Linearize the forward pass
        z, jvp = jax.linearize(fwd, params)

        # Step 3: Compute J^T H J v
        HJv = loss_hessian_mv(jvp(vec), pred=z, target=data["target"])

        # Step 4: Compute the GGN vector
        arr = jax.linear_transpose(jvp, vec)(HJv)[0]

        return mul(factor, arr)

    return ggn_mv


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
        data: A batch of input and target data.
        loss_fn: Loss function to use for the GGN computation.
        num_curv_samples: Number of samples used to calculate the GGN. Defaults to None,
            in which case it is inferred from `data` as its batch size. Note that for
            losses that contain sums even for a single input (e.g., pixel-wise semantic
            segmentation losses, this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_curv_samples`'s description. Defaults to None, in which case
            it is set to equal `num_curv_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_hessian_mv: The loss Hessian matrix-vector product. If not provided, it is
            computed using the `loss_fn`.

    Returns:
        A function that takes a vector and computes the GGN matrix-vector product for
            the given data.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
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
    )

    def wrapped_ggn_mv(vec: Params) -> Params:
        return ggn_mv(vec, data)

    return wrapped_ggn_mv


# -----------------------------------------------------------------------------------
# Loss function factories
# -----------------------------------------------------------------------------------


def _create_loss_fn(
    loss_fn: LossFn | Callable | None,
) -> Callable[[PredArray, TargetArray], Num[Array, "..."]]:
    """Create a loss function from various input types.

    Args:
        loss_fn: Loss function specification.

    Returns:
        A callable loss function that takes (pred, target) and returns loss.

    Raises:
        ValueError: If loss function is not supported for FSP.
    """
    if loss_fn in {LossFn.NONE, LossFn.MSE}:

        def mse_loss(pred: PredArray, target: TargetArray) -> Num[Array, "..."]:
            return jnp.mean((pred - target.reshape(*pred.shape)) ** 2)

        return mse_loss

    if loss_fn == LossFn.BINARY_CROSS_ENTROPY:

        def bce_loss(pred: PredArray, target: TargetArray) -> Num[Array, "..."]:
            return jnp.mean(
                -target * jax.nn.log_sigmoid(pred)
                - (1 - target) * jax.nn.log_sigmoid(-pred)
            )

        return bce_loss

    if loss_fn == LossFn.CROSS_ENTROPY:

        def ce_loss(pred: PredArray, target: TargetArray) -> Num[Array, "..."]:
            if target.ndim == pred.ndim - 1:
                # Integer labels
                return jnp.mean(
                    -jnp.sum(
                        jax.nn.one_hot(target, num_classes=pred.shape[-1])
                        * jax.nn.log_softmax(pred, axis=-1),
                        axis=-1,
                    )
                )
            # One-hot encoded targets
            return jnp.mean(
                -jnp.sum(target * jax.nn.log_softmax(pred, axis=-1), axis=-1)
            )

        return ce_loss

    if isinstance(loss_fn, Callable):
        return loss_fn

    msg = f"Unsupported loss function for FSP: {loss_fn}"
    raise ValueError(msg)


def _ggn_vector_product_batch_fsp(
    model_fn: ModelFn,
    params: Params,
    x_b: InputArray,
    y_b: TargetArray,
    u: Params,
    loss_fn: Callable[[PredArray, TargetArray], Num[Array, "..."]],
    *,
    vmap_over_data: bool = True,
) -> Params:
    """Compute GGN vector product for a single batch using FSP approach.

    Args:
        model_fn: Model forward pass function.
        params: Model parameters.
        x_b: Input batch.
        y_b: Target batch.
        u: Vector to multiply with GGN.
        loss_fn: Loss function.
        vmap_over_data: Whether model expects batch dimension.

    Returns:
        GGN matrix-vector product result.
    """

    def f(p: Params) -> PredArray:
        if vmap_over_data:
            return jax.lax.map(lambda x: model_fn(input=x, params=p), x_b, batch_size=1)
        return model_fn(input=x_b, params=p)

    def loss_fn_wrapped(f_pred: PredArray) -> Num[Array, "..."]:
        return loss_fn(f_pred, y_b)

    f_b, jx = jax.jvp(f, (params,), (u,))
    hjx = jax.jvp(jax.grad(loss_fn_wrapped), (f_b,), (jx,))[1]
    _, jhjx_fn = jax.vjp(f, params)
    return jhjx_fn(hjx)[0]


# -----------------------------------------------------------------------------------
# FSP-based GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_fsp_operator_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
) -> tuple[Callable[[Params, Data], Params], Callable[[Array, Data], Array]]:
    """Create (mv, project) for FSP-style GGN.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.

    Returns:
        A tuple containing `mv` and `project` functions.
    """
    loss_fn_callable = _create_loss_fn(loss_fn)
    _, unflatten = create_pytree_flattener(params)

    def mv(vec: Params, data: Data) -> Params:
        return mul(
            factor,
            _ggn_vector_product_batch_fsp(
                model_fn=model_fn,
                params=params,
                x_b=data["input"],
                y_b=data["target"],
                u=vec,
                loss_fn=loss_fn_callable,
                vmap_over_data=vmap_over_data,
            ),
        )

    def project(basis: Array, data: Data) -> Array:
        """Compute projection of GGN onto a basis U: U^T @ G @ U.

        Returns:
            Projected GGN matrix.
        """
        x_b, y_b = data["input"], data["target"]

        # Helper to compute J_i * u_r for a single input x and single basis vector u_r
        def model_jvp_single(x, u_flat):
            u_pytree = unflatten(u_flat)
            # jvp of model_fn(x, p) wrt p at params with tangent u_pytree
            _, out_tangent = jax.jvp(
                lambda p: model_fn(input=x, params=p),
                (params,),
                (u_pytree,),
            )
            # Flatten the output tangent
            out_flat, _ = jax.flatten_util.ravel_pytree(out_tangent)
            return out_flat

        # Helper to compute J_U for a single input x
        def compute_jvp_basis(x):
            return jax.vmap(lambda u_col: model_jvp_single(x, u_col))(basis.T)

        # Compute J_U for whole batch: (B, R, output_dim)
        if vmap_over_data:
            J_U = jax.vmap(compute_jvp_basis)(x_b)
            # Compute predictions for Hessian
            preds = jax.vmap(lambda x: model_fn(input=x, params=params))(x_b)
        else:
            J_U = jax.lax.map(compute_jvp_basis, x_b)
            preds = model_fn(input=x_b, params=params)

        # Helper for HVP of loss w.r.t pred
        _, unravel_pred = jax.flatten_util.ravel_pytree(
            preds[0]
        )  # Use first pred to get structure

        def loss_flat_single(pred_flat, target):
            pred_pytree = unravel_pred(pred_flat)
            return loss_fn_callable(pred_pytree, target)

        def batch_hvp_flat(pred_flat, target, vectors_flat_R):
            # vectors_flat_R: (R, output_dim)
            def dot_hvp(v):
                return jax.jvp(
                    jax.grad(lambda p: loss_flat_single(p, target)), (pred_flat,), (v,)
                )[1]

            return jax.vmap(dot_hvp)(vectors_flat_R)

        # Flatten predictions
        preds_flat = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(preds)

        # HB_U: (B, R, output_dim)
        HB_U = jax.vmap(batch_hvp_flat)(preds_flat, y_b, J_U)

        # Compute term: sum_{b} (J_U[b].T @ HB_U[b])
        batch_terms = jnp.einsum("bro,bko->rk", J_U, HB_U)

        return mul(factor, batch_terms)

    return mv, project


def create_ggn_mv_fsp_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
) -> Callable[[Params, Data], Params]:
    """Backwards-compatible: return only the mv callable (used by api.py).

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.

    Returns:
        A function that takes a vector and a batch of data, and computes the GGN
        matrix-vector product.
    """
    mv, _ = create_ggn_fsp_operator_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=factor,
        vmap_over_data=vmap_over_data,
    )
    return mv
