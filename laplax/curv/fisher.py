"""Fisher Matrix Vector Product."""

from collections.abc import Callable

import jax

from laplax.curv.loss import fetch_loss_gradient_fn
from laplax.enums import LossFn
from laplax.types import (
    Data,
    Float,
    Int,
    ModelFn,
    Params,
)
from laplax.util.tree import mul


def create_empirical_fisher_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    loss_grad_fn: Callable | None = None,
) -> Callable[[Params, Data], Params]:
    r"""Create empirical Fisher matrix-vector product without fixed data.

    #$$
    #\text{factor} \cdot \sum_i J_i^\top \nabla^2_{f(x_i, \theta), f(x_i, \theta)}
    #\mathcal{L}(f(x_i, \theta), y_i) J_i \cdot v
    #$$

    #where $J_i$ is the Jacobian of the model at data point $i$, $H_{L, i}$ is the
    #Hessian of the loss, and $v$ is the vector. The `factor` is a scaling factor that
    #is used to scale the GGN matrix.

    This function computes the above expression efficiently without hardcoding the
        dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the Fisher computation.
        factor: Scaling factor for the Fisher computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_grad_fn: The loss gradient function.

    Returns:
        A function that takes a vector and a batch of data,
        and computes the empirical Fisher matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    # Create loss gradient product
    loss_grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn, vmap_over_data)

    def empirical_fisher_mv(vec, data):
        def fwd(p):
            # Step 1: Single jvp for entire batch, if vmap_over_data is True
            if vmap_over_data:
                return jax.vmap(lambda x: model_fn(input=x, params=p))(data["input"])
            return model_fn(input=data["input"], params=p)

        # Step 2: Linearize the forward pass
        _, jvp = jax.linearize(fwd, params)

        grad = loss_grad_fn(fwd(params), data["target"])
        vjp = jax.linear_transpose(jvp, vec)

        Jv = jvp(vec)
        GtJv = grad.T @ Jv
        GGtJv = grad @ GtJv
        JtGGtJv = vjp(GGtJv)[0]
        return mul(factor, JtGGtJv)

    return empirical_fisher_mv


def create_empirical_fisher_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable | None = None,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    loss_grad_fn: Callable | None = None,
) -> Callable[[Params], Params]:
    r"""Creates the empirical Fisher matrix-vector product with data.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset
    remains fixed.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the Fisher computation.
        num_curv_samples: Number of samples used to calculate the Fisher.
            Defaults to None, in which case it is inferred from `data`
            as its batch size.
            Note that for losses that contain sums even for a single input
            (e.g., pixel-wise semantic segmentation losses),
            this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_curv_samples`'s description. Defaults to None, in which case
            it is set to equal `num_curv_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_grad_fn: The loss gradient function.
            If not provided, it is computed using the 'loss_fn'.

    Returns:
        A function that takes a vector and computes
        the empirical Fisher matrix-vector product for the given data.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    if num_curv_samples is None:
        num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    fisher_mv = create_empirical_fisher_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        vmap_over_data=vmap_over_data,
        loss_grad_fn=loss_grad_fn,
    )

    def wrapped_fisher_mv(vec: Params) -> Params:
        return fisher_mv(vec, data)

    return wrapped_fisher_mv
