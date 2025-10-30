"""Loss Hessians."""

from collections.abc import Callable

import jax

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Kwargs,
    Num,
    PredArray,
    TargetArray,
)

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

