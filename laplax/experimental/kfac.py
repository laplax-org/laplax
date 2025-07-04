from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr

from laplax.enums import LossFn
from laplax.types import Array, Data, ModelFn, Params
from laplax.util.intergrad import intergrad


def kfac_blocks(
    params: Params,
    model_fn: ModelFn,
    data: Data,
    fisher_type: Literal["type1", "empirical"] = "empirical",
    fisher_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable | LossFn = LossFn.CROSS_ENTROPY,
) -> tuple[list[Array], list[Array]]:
    """Computes the KFAC blocks.

    Args:
        params: The parameters of the model.
        model_fn: The model function that takes parameters and inputs.
        data: A dictionary containing input data under the key "input" and
            target labels under "target".
        fisher_type: The type of Fisher information to compute, either
            'type1' or 'empirical'.
        fisher_kwargs: Additional keyword arguments for the Fisher computation.
        loss_fn: The loss function to use, either a callable or a string identifier.

    Returns:
        A tuple containing two lists:
            - A list of activation matrices (A) for each layer.
            - A list of gradient matrices (B) for each layer.
    """
    if (
        loss_fn.value != LossFn.CROSS_ENTROPY.value
    ):  # python 3.11 doesn't like enum equalities
        msg = "Only cross_entropy loss is supported for now."
        raise NotImplementedError(msg)

    fisher_kwargs = fisher_kwargs or {}  # check for None

    def celoss(params, x, y):
        return -(y * jax.nn.log_softmax(model_fn(params, x))).mean()

    x, y = data["input"], data["target"]

    acts_and_grads = {"empirical": emp_fisher_grads, "type1": type_1_fisher_grads}[
        fisher_type
    ]

    activations, grads = acts_and_grads(
        params=params, model_fn=model_fn, data=data, loss_fn=celoss, **fisher_kwargs
    )
    activations, grads = (
        jax.tree.map(jnp.atleast_2d, activations),
        jax.tree.map(jnp.atleast_2d, grads),
    )
    As, Bs = [], []
    R, N = 1 / (x.shape[0] * y.shape[-1]), x.shape[0]  # reduction factors
    for A, G in zip(activations, grads, strict=False):
        # TODO(@author): Devil is in the detail: The order by which we concatenate
        # the ones to the A_in matters. The pytree definition of
        # the network defines the bias BEFORE the weights. Hence,
        # we have to pre-pend the ones to be compatible.
        A_aug = jnp.concatenate([jnp.ones((A.shape[0], 1)), A], axis=1)
        As.append(jax.lax.stop_gradient(A_aug.T @ A_aug * R))
        Bs.append(jax.lax.stop_gradient((G.T @ G) / N))
    return As, Bs


def emp_fisher_grads(
    params: Params, data: Data, loss_fn: Callable, **kwargs
) -> tuple[list[Array], list[Array]]:
    """Computes the empirical Fisher information matrix.

    Args:
        params: The parameters of the model.
        data: A dictionary containing input data under the key "input" and
            target labels under "target".
        loss_fn: The loss function to use.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        A tuple containing:
            - A list of activations for each layer.
            - A list of gradients for each layer.

    """
    del kwargs  # unused
    xs, ys = data["input"], data["target"]
    intergrad_jit = jax.jit(
        jax.vmap(intergrad(loss_fn, tagging_rule=None), in_axes=(None, 0, 0))
    )
    activations, grads = intergrad_jit(params, xs, ys)
    activations = [xs, *activations]
    return activations, grads


def type_1_fisher_grads(
    params: Params,
    model_fn: ModelFn,
    data: Data,
    loss_fn: Callable,
    num_samples: int = 128,
    **kwargs,
) -> tuple[list[Array], list[Array]]:
    r"""Computes the Type-1 Fisher information matrix.

    Args:
        params: The parameters of the model.
        model_fn: The model function that takes parameters and inputs.
        data: A dictionary containing input data under the key "input" and
            target labels under "target".
        loss_fn: The loss function to use.
        num_samples: The number of samples to draw from the output distribution.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        A tuple containing:
            - A list of activations for each layer.
            - A list of gradients for each layer.

    For each sample x, we compute the forward pass, then draw `num_samples`
    \hat y samples from the output distribution. We continue, by collecting
    `num_samples` gradients from intergrad(model, x, \hat y).
    """
    error_msg = "Type-1 Fisher gradients aren't sufficiently tested yet."
    raise NotImplementedError(error_msg)

    del kwargs  # unused
    xs = data["input"]
    intergrad_jit = jax.jit(
        jax.vmap(intergrad(loss_fn, tagging_rule=None), in_axes=(None, 0, 0))
    )
    key = jr.PRNGKey(0)
    activations_buf, grads_buf = [], []

    for x in xs:
        # draw samples
        key, subkey = jr.split(key)
        logits = model_fn(params, x)
        sampled_labels = jax.random.categorical(
            subkey, logits, axis=-1, shape=(num_samples,)
        )
        sampled_labels = jax.nn.one_hot(sampled_labels, num_classes=logits.shape[-1])

        x_repeat = jnp.repeat(x[jnp.newaxis, :], num_samples, axis=0)

        # compute activations and gradients from sampled labels
        # activations is [(num_samples, a1), (num_samples, a2), ...]
        # grads is [(num_samples, g1), (num_samples, g2), ...]
        activations, grads = intergrad_jit(params, x_repeat, sampled_labels)

        grads_buf.append(grads)

        # activations are duplicates, so we take the first one for
        # each leaf in the tree.
        acts = jax.tree.map(lambda x: x[0][jnp.newaxis, :], activations)
        activations_buf.append(acts)

    # grads is list of N  * [(g1,), (g2,), (g3,)], we need to turn this
    # into [(N, g1), (N, g2), (N, g3)].
    stacked_grads = [jnp.concat(gs) for gs in zip(*grads_buf, strict=False)]
    stacked_activations = [
        jnp.concat(acts) for acts in zip(*activations_buf, strict=False)
    ]
    stacked_activations = [xs, *stacked_activations]  # prepend the input x

    return stacked_activations, stacked_grads
