"""Utility functions for curvature estimation."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Float,
    InputArray,
    Layout,
    ModelFn,
    Num,
    Params,
    TargetArray,
)
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.tree import get_size

# -----------------------------------------------------------------------------
# Low-rank terms
# -----------------------------------------------------------------------------


@dataclass
class LowRankTerms:
    """Components of the low-rank curvature approximation.

    This dataclass encapsulates the results of the low-rank approximation, including
    the eigenvectors, eigenvalues, and a scalar factor which can be used for the prior.

    Attributes:
        U: Matrix of eigenvectors, where each column corresponds to an eigenvector.
        S: Array of eigenvalues associated with the eigenvectors.
        scalar: Scalar factor added to the matrix during the approximation.
    """

    U: Num[Array, "P R"]
    S: Num[Array, " R"]
    scalar: Float[Array, ""]


jax.tree_util.register_pytree_node(
    LowRankTerms,
    lambda node: ((node.U, node.S, node.scalar), None),
    lambda _, children: LowRankTerms(U=children[0], S=children[1], scalar=children[2]),
)


def get_matvec(
    A: Callable | Array,
    *,
    layout: Layout | None = None,
    jit: bool = True,
) -> tuple[Callable[[Array], Array], int]:
    """Returns a function that computes the matrix-vector product.

    Args:
        A: Either a jnp.ndarray or a callable performing the operation.
        layout: Required if `A` is callable; ignored if `A` is an array.
        jit: Whether to jit-compile the operator.

    Returns:
        A tuple (matvec, input_dim) where matvec is the callable operator.

    Raises:
        TypeError: When `A` is a callable but `layout` is not provided.
    """
    if isinstance(A, jnp.ndarray):
        size = A.shape[0]

        def matvec(x):
            return A @ x

    else:
        matvec = A

        if layout is None:
            msg = "For a callable A, please provide `layout` in PyTree or int format."
            raise TypeError(msg)
        if isinstance(layout, int):
            size = layout
        else:
            try:
                flatten, unflatten = create_pytree_flattener(layout)
                matvec = wrap_function(matvec, input_fn=unflatten, output_fn=flatten)
                size = get_size(layout)
            except (ValueError, TypeError) as exc:
                msg = (
                    "For a callable A, please provide `layout` in PyTree or int format."
                )
                raise TypeError(msg) from exc

    if jit:
        matvec = jax.jit(matvec)

    return matvec, size


def log_sigmoid_cross_entropy(
    logits: Num[Array, "..."], targets: Num[Array, "..."]
) -> Num[Array, "..."]:
    r"""Computes log sigmoid cross entropy given logits and targets.

    This function computes the cross entropy loss between the sigmoid of the logits
    and the target values. The formula implemented is:

    $$
    \mathcal{L}(f(x, \theta), y) = -y \cdot \log \sigma(f(x, \theta)) -
    (1 - y) \cdot \log \sigma(-f(x, \theta))
    $$

    Args:
        logits: The predicted logits before sigmoid activation
        targets: The target values (0 or 1)

    Returns:
        The computed loss value
    """
    return -targets * jax.nn.log_sigmoid(logits) - (1 - targets) * jax.nn.log_sigmoid(
        -logits
    )


def concatenate_model_and_loss_fn(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    loss_fn: LossFn | str | Callable,
    *,
    vmap_over_data: bool = False,
) -> Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]:
    r"""Combine a model function and a loss function into a single callable.

    This creates a new function that evaluates the model and applies the specified
    loss function. If `vmap_over_data` is `True`, the model function is vectorized over
    the batch dimension using `jax.vmap`.

    Mathematically, the combined function computes:

    $$
    \mathcal{L}(x, y, \theta) = \text{loss}(f(x, \theta), y),
    $$

    where $f$ is the model function, $\theta$ are the model parameters, $x$ is the
    input, $y$ is the target, and $\mathcal{L}$ is the loss function.

    Args:
        model_fn: The model function to evaluate.
        loss_fn: The loss function to apply. Supported options are:

            - `LossFn.MSE` for mean squared error.
            - `LossFn.BINARY_CROSS_ENTROPY` for binary cross-entropy loss.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - `LossFn.NONE` for no loss.
            - A custom callable loss function.

        vmap_over_data: Whether the model function should be vectorized over the data.

    Returns:
        A combined function that computes the loss for given inputs, targets, and
            parameters.

    Raises:
        ValueError: When the loss function is unknown.
    """
    if vmap_over_data:
        model_fn = jax.vmap(model_fn, in_axes=(0, None))

    if loss_fn == LossFn.MSE:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return jnp.sum((model_fn(input, params) - target) ** 2)

        return loss_wrapper

    if loss_fn == LossFn.CROSS_ENTROPY:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return log_sigmoid_cross_entropy(model_fn(input, params), target)

        return loss_wrapper

    if callable(loss_fn):

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return loss_fn(model_fn(input, params), target)

        return loss_wrapper

    msg = f"unknown loss function: {loss_fn}"
    raise ValueError(msg)
