"""Generalized Gauss-Newton matrix-vector product."""

from collections.abc import Callable

import jax

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    Float,
    Int,
    ModelFn,
    Params,
)
from laplax.util.tree import mul

from laplax.curv.loss import fetch_loss_hessian_mv



# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable | None,
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
    loss_hessian_mv = fetch_loss_hessian_mv(loss_fn, loss_hessian_mv, vmap_over_data)

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
            remark in `num_ggn_samples`'s description. Defaults to None, in which case
            it is set to equal `num_ggn_samples`.
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
