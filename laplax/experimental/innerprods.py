import warnings
import jax
import jax.numpy as jnp

from flax import nnx
from functools import partial
from laplax.experimental.kfac import kfac_blocks

from laplax.curv import create_ggn_mv
from laplax.util.tree import sub, dot, mul, add
from typing import *
from jaxtyping import Array, PyTree

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    ModelFn,
    Params
)

def kfac_inner_fn(params : PyTree, model_fn : ModelFn, data : Data, 
                  loss_fn : LossFn = LossFn.CROSS_ENTROPY,
                  *args, **kwargs) -> Callable[[Array], Array]:
    """
        Returns a function that computes the KFAC inner product function for a model.

        Args:
            params: Model parameters.
            model_fn: Function that computes the model output given parameters and input.
            trainloader: DataLoader providing training data.
            maxsamples: Number of samples to use from the loader.
            loss_fn: Loss function to use for the KFAC computation (default: 'cross_entropy

        Returns:
            inner: A function that computes the inner product with a vector v scaled by
            the block diagonal KFAC matrix.
    """
    warnings.warn(
            "Current assumption is that the pytree structure puts bias \
                before weights AND your model only consists of Linear Layers.")
    
    if loss_fn.value != LossFn.CROSS_ENTROPY.value: # python 3.11 doesn't like enum equalities
        raise NotImplementedError("Only cross_entropy loss is supported for now.")
        
    As, Bs = kfac_blocks(params=params, model_fn=model_fn, data=data, loss_fn=loss_fn)

    def kfac_vtmv(v, As=As, Bs=Bs):
        leaves = jax.tree_util.tree_leaves(v)
        vtFv = 0.0
        ws, bs = leaves[1::2], leaves[0::2]
        
        for A_fac, B_fac, w, b in zip(As, Bs, ws, bs):

            # fuse together according to structure of nnx pytree -> first bias, then weights
            W_ext = jnp.concatenate([b[jnp.newaxis, :], w], axis=0) 
            vtFv += jnp.sum(W_ext * (A_fac @ W_ext @ B_fac))
        return vtFv
    return kfac_vtmv

def emp_fisher_inner(params : Params, model_fn : ModelFn, data : Data,
                     loss_fn : LossFn = LossFn.CROSS_ENTROPY,
                     *args, **kwargs):
    """
    Computes the empirical Fisher information inner product function for a model.

    Args:
        params: Model parameters.
        model_fn: Function that computes the model output given parameters and input.
        trainloader: DataLoader providing training data.
        maxsamples: Number of samples to use from the loader.
        *args, **kwargs: Additional arguments (unused).

    Returns:
        inner: A function that computes the empirical Fisher inner product with a vector v.
    """
    
    if loss_fn.value != LossFn.CROSS_ENTROPY.value: # python 3.11 doesn't like enum equalities
        raise NotImplementedError("Only cross_entropy loss is supported for now.")
        
    x, y = data['input'], data['target']

    def cross_entropy(params, x, y):
        log_y_pred = jax.nn.log_softmax(model_fn(params, x))
        return -(log_y_pred * y).mean()

    grads = nnx.grad(cross_entropy)(params, x, y)
    sqgrads = jax.tree.map(lambda x: jnp.mean(x**2, axis=0), grads)  # square and mean

    def inner(v, sqgrads=sqgrads):
        return dot(v, jax.tree.map(lambda x, y: x * y, v, sqgrads))

    return inner

def zero(*args, **kwargs):
    return lambda x: 0

def unscaled_dot_product(*args, **kwargs):
    """
    Returns a function that computes the unscaled dot product of a vector with itself.

    Args:
        *args, **kwargs: Ignored.

    Returns:
        inner: A function that computes dot(v, v).
    """
    def inner(v):
        return dot(v, v)
    return inner

def ggn_inner(params : Params, model_fn : ModelFn, data : Data, 
            numsamples_train : int,
            loss_fn : LossFn = LossFn.CROSS_ENTROPY,
            *args, **kwargs):
    """
    Returns a function that computes the Generalized Gauss-Newton (GGN) inner product for a model.

    Args:
        params: Model parameters.
        model_fn: Function that computes the model output given parameters and input.
        trainloader: DataLoader providing training data.
        maxsamples: Number of samples to use from the loader.
        numsamples_train: Total number of training samples (for scaling).
        *args, **kwargs: Additional arguments (unused).

    Returns:
        inner: A function that computes the GGN inner product with a vector v.
    """
    partial_hvp = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=loss_fn,
        num_total_samples=numsamples_train
    )
    def inner(v, partial_hvp=partial_hvp):
        return dot(v, partial_hvp(v))

    return inner

def type1_fisher_inner(params : Params, 
                        model_fn : ModelFn,
                        data : Data, 
                        loss_fn : LossFn = LossFn.CROSS_ENTROPY,M=30, *args, **kwargs):
    """
    Computes the Type-1 Fisher information inner product function using Monte Carlo label sampling.

    Args:
        model: The neural network model.
        trainloader: DataLoader providing training data.
        maxsamples: Number of samples to use from the loader.
        M: Number of Monte Carlo samples (default: 30).
        *args, **kwargs: Additional arguments (unused).

    Returns:
        inner: A function that computes the Type-1 Fisher inner product with a vector v.
    """
    if loss_fn.value != LossFn.CROSS_ENTROPY.value: # python 3.11 doesn't like enum equalities
        raise NotImplementedError("Only cross_entropy loss is supported for now.")
        

    def sample_cross_entropy(params, x, y, *, key):
        logits = model_fn(params, x)
        probs = jax.nn.softmax(logits)
        # Sample one label per example in the batch
        sampled_labels = jax.random.categorical(key, logits, axis=-1)
        ysample = jax.nn.one_hot(sampled_labels, num_classes=probs.shape[-1])
        log_probs = jax.nn.log_softmax(logits)
        return -(log_probs * ysample).mean()

    x, y = data['input'], data['target']
    running_mean = None
    key = jax.random.PRNGKey(0)

    # running mean accumulation of gradients
    for i in range(M):
        key, curkey = jax.random.split(key)
        grads = nnx.vmap(
            nnx.grad(
                partial(sample_cross_entropy, key=curkey)
            ), in_axes=(None, 0, 0)
        )(params, x, y)  # shape (batch, *param_sizes)

        grads = jax.tree.map(lambda x: (x**2).mean(0), grads)

        if running_mean is None:
            running_mean = grads
        else:
            running_mean = add(running_mean, mul(1/(i + 1), sub(grads, running_mean)))

    def inner(v, running_mean=running_mean):
        return dot(v, jax.tree.map(lambda x, y: x * y, v, running_mean))

    return inner