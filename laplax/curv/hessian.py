"""Hessian vector product for curvature estimation."""

import jax

from laplax.curv.utils import concatenate_model_and_loss_fn
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Callable,
    Data,
    Float,
    InputArray,
    Int,
    Kwargs,
    ModelFn,
    Num,
    Params,
    PyTree,
    TargetArray,
)
from laplax.util.tree import mul


def hvp(
    func: Callable,
    primals: PyTree,
    tangents: PyTree,
) -> PyTree:
    r"""Compute the Hessian-vector product (HVP) for a given function.

    The Hessian-vector product is computed by differentiating the gradient of the
    function. This avoids explicitly constructing the Hessian matrix, making the
    computation efficient.

    Args:
        func: The scalar function for which the HVP is computed.
        primals: The point at which the gradient and Hessian are evaluated.
        tangents: The vector to multiply with the Hessian.

    Returns:
        The Hessian-vector product.
    """
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def create_hessian_mv_without_data(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    loss_fn: LossFn | str | Callable,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    **kwargs: Kwargs,
) -> Callable[[Params, Data], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss function.

    This function computes the HVP by combining the model and loss functions into a
    single callable. It evaluates the Hessian at the provided model parameters, with
    respect to the model and loss function.

    Mathematically:

    $$
    H \cdot v = \nabla^2 \mathcal{L}(f(x, \theta), y) \cdot v,
    $$

    where $\mathcal{L}$ is the combined loss function, $f$ is the model function, $x$ is
    the input, $y$ is the target, $\theta$ are the parameters, and $v$ is the input
    input vector.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        loss_fn: The loss function to apply. Supported options are:

            - `LossFn.BINARY_CROSS_ENTROPY` for binary cross-entropy loss.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - `LossFn.MSE` for mean squared error.
            - `LossFn.NONE` for no loss.
            - A custom callable loss function.

        factor: Scaling factor for the Hessian computation.
        vmap_over_data: Whether the model function should be vectorized over the data.
        **kwargs: Additional arguments (ignored).

    Returns:
        A function that computes the HVP for a given vector and batch of data.
    """
    del kwargs

    new_model_fn: Callable[[InputArray, TargetArray, Params], Num[Array, "..."]] = (  # noqa: UP037
        concatenate_model_and_loss_fn(model_fn, loss_fn, vmap_over_data=vmap_over_data)
    )

    def _hessian_mv(vec: Params, data: Data) -> Params:
        return mul(
            factor,
            hvp(
                lambda p: new_model_fn(data["input"], data["target"], p),
                params,
                vec,
            ),
        )

    return _hessian_mv


def create_hessian_mv(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    **kwargs: Kwargs,
) -> Callable[[Params], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss fn. with data.

    This function wraps :func: `create_hessian_mv_without_data`, fixing the dataset to
    produce a function that computes the HVP for the specified data.

    Mathematically:

    $$
    H \cdot v = \nabla^2 \mathcal{L}(f(x, \theta), y) \cdot v,
    $$

    where $\mathcal{L}$ is the combined loss function, $f$ is the model function, $x$ is
    the input, $y$ is the target, $\theta$ are the parameters, and $v$ is the input
    vector of the HVP.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        data: A batch of input and target data.
        loss_fn: The loss function to apply. Supported options are:


            - `LossFn.MSE` for mean squared error.
            - `LossFn.BINARY_CROSS_ENTROPY` for binary cross-entropy loss.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - `LossFn.NONE` for no loss.
            - A custom callable loss function.

        num_curv_samples: Number of samples used to calculate the Hessian. Defaults to
            None, in which case it is inferred from `data` as its batch size. Note that
            for losses that contain sums even for a single input (e.g., pixel-wise
            semantic segmentation losses), this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_ggn_samples`'s description. Defaults to None, in which case
            it is set to equal `num_ggn_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        **kwargs: Additional arguments.

    Returns:
        A function that computes the HVP for a given vector and the fixed dataset.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    if num_curv_samples is None:
        num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    hessian_mv = create_hessian_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        vmap_over_data=vmap_over_data,
        **kwargs,
    )

    def wrapped_hessian_mv(vec: Params) -> Params:
        return hessian_mv(vec, data)

    return wrapped_hessian_mv
