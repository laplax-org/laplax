import jax
import jax.numpy as jnp; import jax.random as jr;

from typing import Callable, Union
from laplax.util.intergrad import intergrad
from typing import Tuple
from jaxtyping import Array, PyTree
from typing import Literal, Dict, Any, List

def kfac_blocks(params, model_fn, x, y, 
                fisher_type : Literal['type1', 'empirical'] = 'empirical', 
                fisher_kwargs : Dict[str, Any] = {},
                loss_fn : Union[Callable, Literal['cross_entropy', 'squared_error']] = 'cross_entropy') \
      -> Tuple[List[Array], List[Array]]:
    """
        Computes the KFAC blocks.

        Args:
            params: The parameters of the model.
            model_fn: The model function that takes parameters and inputs.
            x: The input data.
            y: The target labels.
            fisher_type: The type of Fisher information to compute, either 'type1' or 'empirical'.
            fisher_kwargs: Additional keyword arguments for the Fisher computation.
            loss_fn: The loss function to use, either a callable or a string identifier.

        Returns:
            A tuple containing two lists:
                - A list of activation matrices (A) for each layer.
                - A list of gradient matrices (B) for each layer.
    """
    if loss_fn != 'cross_entropy':
        raise NotImplementedError("Only cross_entropy loss is supported for now.")
    
    def celoss(params, x, y): 
        return -(y * jax.nn.log_softmax(model_fn(params, x))).mean()
    
    acts_and_grads = {
        'empirical': emp_fisher_grads,
        'type1': type_1_fisher_grads
    }[fisher_type]

    activations, grads = acts_and_grads(params=params, model_fn=model_fn, xs=x, ys=y, loss_fn=celoss, **fisher_kwargs)
    activations, grads =    jax.tree.map(lambda x: jnp.atleast_2d(x), activations), \
                            jax.tree.map(lambda x: jnp.atleast_2d(x), grads)
    As, Bs = [], []
    R, N = 1 / (x.shape[0] * y.shape[-1]), x.shape[0] # reduction factors
    for A, G in zip(activations, grads):

        # TODO: Devil is in the detail: The order by which we concatenate
        # the ones to the A_in matters. The pytree definition of
        # the network defines the bias BEFORE the weights. Hence,
        # we have to pre-pend the ones to be compatible.
        A_aug = jnp.concatenate([jnp.ones((A.shape[0], 1)), A], axis=1)
        As.append(jax.lax.stop_gradient((A_aug.T @ A_aug * R)))
        Bs.append(jax.lax.stop_gradient((G.T @ G) / N))
    return As, Bs

def emp_fisher_grads(params : PyTree, model_fn : Callable, xs, ys, loss_fn, **kwargs) \
    -> Tuple[List[Array], List[Array]]:
    """
    Computes the empirical Fisher information matrix by collecting
    activations and gradients for each input sample.

    Args:
        params: The parameters of the model.
        model_fn: The model function that takes parameters and inputs.
        xs: Input data.
        ys: Target labels.
        loss_fn: The loss function to use.
    
    Returns:
        A tuple containing:
            - A list of activations for each layer.
            - A list of gradients for each layer.

    """
    intergrad_jit = jax.jit(jax.vmap(intergrad(loss_fn, tagging_rule=None), 
                                  in_axes=(None, 0, 0)))
    activations, grads = intergrad_jit(params, xs, ys)
    activations = [xs] + activations
    return activations, grads

def type_1_fisher_grads(params : PyTree, model_fn : Callable, xs, ys, loss_fn, num_samples=128, **kwargs) \
    -> Tuple[List[Array], List[Array]]:
    r"""
        Computes the Type-1 Fisher information matrix by sampling from the output distribution.

        Args:
            params: The parameters of the model.
            model_fn: The model function that takes parameters and inputs.
            xs: Input data.
            ys: Target labels.
            loss_fn: The loss function to use.
            num_samples: The number of samples to draw from the output distribution.

        Returns:
            A tuple containing:
                - A list of activations for each layer.
                - A list of gradients for each layer.

        For each sample x, we compute the forward pass, then draw `num_samples`
        \hat y samples from the output distribution. We continue, by collecting
        `num_samples` gradients from intergrad(model, x, \hat y).
    """
    raise NotImplementedError("Type-1 Fisher gradients aren't sufficiently tested yet.")

    intergrad_jit = jax.jit(jax.vmap(intergrad(loss_fn, tagging_rule=None), 
                                  in_axes=(None, 0, 0)))
    key = jr.PRNGKey(0)
    activations_buf, grads_buf = [], []

    for x in xs:
        # draw samples
        key, subkey = jr.split(key)
        logits = model_fn(params, x)
        sampled_labels = jax.random.categorical(subkey, logits, axis=-1, shape=(num_samples,))
        sampled_labels = jax.nn.one_hot(sampled_labels, num_classes=logits.shape[-1])
        
        x_repeat = jnp.repeat(x[jnp.newaxis, :], num_samples, axis=0)

        # compute activations and gradients from sampled labels
        # activations is [(num_samples, a1), (num_samples, a2), ...]
        # grads is [(num_samples, g1), (num_samples, g2), ...]
        activations, grads = intergrad_jit(params, x_repeat, sampled_labels)

        grads_buf.append(grads)

        # activations are duplicates, so we take the first one for each leaf in the tree.
        acts = jax.tree.map(lambda x: x[0][jnp.newaxis, :], activations)
        activations_buf.append(acts)

    # grads is list of N  * [(g1,), (g2,), (g3,)], we need to turn this
    # into [(N, g1), (N, g2), (N, g3)]. 
    stacked_grads = [jnp.concat(gs) for gs in zip(*grads_buf)]
    stacked_activations = [jnp.concat(acts) for acts in zip(*activations_buf)]
    stacked_activations = [xs] + stacked_activations  # prepend the input x

    return stacked_activations, stacked_grads
    