import jax
import jax.numpy as jnp; import jax.random as jr;
from flax import nnx
from temp.intergrad import intergrad
from typing import Tuple
from jaxtyping import Array
from typing import Literal, Dict, Any

def kfac_blocks(model, x, y, fisher_type : Literal['type1', 'empirical'] = 'empirical', fisher_kwargs : Dict[str, Any] = {}) -> Tuple[Array, Array]:
    """
        Computes the KFAC blocks. This is the KFAC expand
        empirical implementation in which the Empirical Fisher blocks do not 
        equal to the computed KFAC blocks.
    """

    graph, params = nnx.split(model)
    def model_fn(p, x):
        return nnx.merge(graph, p)(x)
    
    def celoss(params, x, y): 
        return -(y * jax.nn.log_softmax(model_fn(params, x))).mean()
    
    acts_and_grads = {
        'empirical': emp_fisher_grads,
        'type1': type_1_fisher_grads
    }[fisher_type]

    activations, grads = acts_and_grads(model=model, xs=x, ys=y, loss_fn=celoss, **fisher_kwargs)
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

def emp_fisher_grads(model, xs, ys, loss_fn, **kwargs):
    intergrad_jit = jax.jit(jax.vmap(intergrad(loss_fn, tagging_rule=None), 
                                  in_axes=(None, 0, 0)))
    graph, params = nnx.split(model)
    activations, grads = intergrad_jit(params, xs, ys)
    activations = [xs] + activations
    return activations, grads

def type_1_fisher_grads(model, xs, ys, loss_fn, num_samples=128, **kwargs):
    r"""
        For each sample x, we compute the forward pass, then draw `num_samples`
        \hat y samples from the output distribution. We continue, by collecting
        `num_samples` gradients from intergrad(model, x, \hat y).
    """
    intergrad_jit = jax.jit(jax.vmap(intergrad(loss_fn, tagging_rule=None), 
                                  in_axes=(None, 0, 0)))
    key = jr.PRNGKey(0)
    activations_buf, grads_buf = [], []
    graph, params = nnx.split(model)

    for x in xs:
        # draw samples
        key, subkey = jr.split(key)
        logits = model(x)
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
    