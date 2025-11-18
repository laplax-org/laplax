# utils.py

"""Utility functions for curvature estimation."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import flatten_util

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Float,
    InputArray,
    Layout,
    ModelFn,
    Num,
    Params,
    PredArray,
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


def create_model_jvp(
    params: Params, v: Params, model_fn: ModelFn, in_axes=0, out_axes=0
) -> Callable[[InputArray], PredArray]:
    """Compute the Jacobian-vector product of the model function.

    Args:
        params: Model parameters.
        v: Vector to multiply with the Jacobian.
        model_fn: The model function.

    Returns:
        The Jacobian-vector product.
    """
    return jax.vmap(
        lambda x: jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (v,),
        )[1],
        in_axes=in_axes,
        out_axes=out_axes,
    )


def create_model_vjp(
    params: Params, v: Num[Array, "..."], model_fn: ModelFn, in_axes=0, out_axes=0
) -> Callable[[InputArray], Params]:
    """Compute the vector-Jacobian product of the model function.

    Args:
        params: Model parameters.
        v: Vector to multiply with the Jacobian.
        model_fn: The model function.

    Returns:
        The vector-Jacobian product.
    """
    return jax.vmap(
        lambda x: jax.vjp(
            lambda w: model_fn(x, w),
            params,
        )[1](v)[0],
        in_axes=in_axes,
        out_axes=out_axes,
    )


def compute_posterior_truncation_index(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    cov_sqrt: Num[Array, "P R"],
    prior_variance: Num[Array, "..."],
):
    """Compute truncation index for posterior components.

    This matches the FSP truncation logic used in the Luno experiments:

    - For each low-rank component (column of ``cov_sqrt``), compute the
      contribution to the **total** posterior variance via squared JVPs over
      all context points.
    - Accumulate these contributions until the running sum reaches or exceeds
      the sum of the prior variance.
    - The truncation index is the smallest number of components needed to
      match the total prior variance.

    Args:
        model_fn: Model function f(x, params).
        params: Model parameters (pytree).
        x_context: Context inputs over which to evaluate JVPs.
        cov_sqrt: Matrix whose columns are low-rank factors in parameter space
            (shape (P, R), where P is number of parameters and R is rank).
        prior_variance: Prior variance over the context/output space. Only the
            total sum is used for the truncation criterion.

    Returns:
        truncation_idx: Number of components to keep (JAX scalar integer).
    """
    # Unravel columns of cov_sqrt back into the params pytree shape
    _, unravel_fn = flatten_util.ravel_pytree(params)

    # Sum of prior variance provides the global truncation target.
    prior_var_sum = jnp.sum(prior_variance)

    def jvp_fn(x, v):
        return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

    def scan_fn(carry, i):
        running_sum, truncation_idx = carry
        lr_fac = unravel_fn(cov_sqrt[:, i])

        # Compute squared JVP over all context points and sum contributions
        sqrt_jvp = jax.vmap(lambda xc: jvp_fn(xc, lr_fac) ** 2)(x_context)
        pv = jnp.sum(sqrt_jvp)
        new_running_sum = running_sum + pv

        new_truncation_idx = jax.lax.cond(
            (new_running_sum >= prior_var_sum) & (truncation_idx == -1),
            lambda _: (i + 1).astype(truncation_idx.dtype),
            lambda _: truncation_idx,
            operand=None,
        )

        return (new_running_sum, new_truncation_idx), jnp.array(0.0, pv.dtype)

    init_carry = (jnp.array(0.0, prior_var_sum.dtype), jnp.array(-1, jnp.int64))
    indices = jnp.arange(cov_sqrt.shape[1])
    (running_sum, truncation_idx), _ = jax.lax.scan(scan_fn, init_carry, indices)

    truncation_idx = jax.lax.cond(
        truncation_idx == -1,
        lambda _: jnp.array(cov_sqrt.shape[1], dtype=truncation_idx.dtype),
        lambda _: truncation_idx,
        operand=None,
    )

    # Ensure at least one component
    return jnp.maximum(truncation_idx, jnp.array(1, dtype=truncation_idx.dtype))
