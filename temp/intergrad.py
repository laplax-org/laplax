import time

import jax
import jax.numpy as jnp
from flax import nnx
import equinox as eqx

from itertools import chain
from functools import partial
from jax import jit, grad, vmap
from jax import random

from functools import wraps

from jax import lax
from jax.extend import core
from jax._src.util import safe_map
from jax.extend.core import Jaxpr, JaxprEqn, Var
from jax._src.core import Atom, jaxpr_as_fun
from typing import Sequence

"""
    Following: https://github.com/jax-ml/jax/discussions/5336

    1) go through the function and inject "perturbation" additions where needed.
    2) use regular "grad" to collect gradients
"""

def f_pert(x, ps):
    a = x**2
    b = a + 1 + ps[0]
    c = jax.nn.relu(b)
    d = c + 1 + ps[1]
    e = jax.nn.relu(d)
    f = e ** 2
    return f

def f(x):
    a = x**2
    b = a + 1
    c = jax.nn.relu(b)
    d = c + 1
    e = jax.nn.relu(d)
    f = e ** 2
    return f

"""
    We need to 
    
    1) construct the jaxpr of the regular function.
    2) given a special rule, we inject weight perturbations

        a = mul ia ib
        b = relu a
        c = add a b <<< collect gradients of L w.r.t. c

        will be transformed to

        a = mul ia ib
        b = relu a
        c = add a b
        d = c + cperturbation
"""

def flat_eval_jaxpr(jaxpr, consts, *args):
  """
    Evaluates a JaxPr.
  """
  env = {}

  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  for i, eqn in enumerate(jaxpr.eqns):
    invals = safe_map(read, eqn.invars)
    
    if "call_jaxpr" in eqn.params: # handle custom definitions (i.e. jax.nn.relu)
        subjaxpr = eqn.params["call_jaxpr"]
        sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
        
        if type(subjaxpr) is core.ClosedJaxpr:
            subjaxpr = subjaxpr.jaxpr
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()

        outvals = flat_eval_jaxpr(subjaxpr, sub_consts, *invals)
    else:
        outvals = eqn.primitive.bind(*invals, **eqn.params)

    if not eqn.primitive.multiple_results:
        outvals = [outvals]

    safe_map(write, eqn.outvars, outvals)
  # Read the final result of the Jaxpr from the environment
  return safe_map(read, jaxpr.outvars)

def perturb(jaxpr, consts, perturbations, *args):
    """
        To be differentiated w.r.t. perturbations
    """

    env = {}
    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]
    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for i, eqn in enumerate(jaxpr.eqns):
        invals = safe_map(read, eqn.invars)
        if "call_jaxpr" in eqn.params:              # if its a relu, record the activation
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            outvals = flat_eval_jaxpr(subjaxpr, sub_consts, *invals) # assuming that we're just encountering relus
        else:                                       
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        if eqn.primitive.name == 'add':
            pert = perturbations.pop(0)
            outvals[0] = outvals[0] + pert    
        safe_map(write, eqn.outvars, outvals)
    return safe_map(read, jaxpr.outvars)[0]


def flat_inject(jaxpr, consts, tagging_rule, *args):
    """
        We step through the jaxpr and collect post-relu activations.
        Additionally, we construct another jaxpr in which we perturb the weights.
        This jaxpr is interpreted as a function depending on the perturbation inputs and
        can be called with jax.grad to obtain the intermediate gradients.
    """
    
    env = {}
    perturbations, activations = [], []

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if "call_jaxpr" in eqn.params:              # if its a relu, record the activation
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            outvals = flat_eval_jaxpr(subjaxpr, sub_consts, *invals) # assuming that we're just encountering relus
            activations.append(outvals[0])
        else:                                       
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        if eqn.primitive.name == 'add':
            perturbations.append(jnp.zeros_like(outvals[0]))

        safe_map(write, eqn.outvars, outvals)

    perturbed_fn = lambda perts, *args : perturb(jaxpr, consts, perts, *args)
    grads = jax.grad(perturbed_fn)(perturbations, *args)

    return activations, grads

def intergrad(fun, tagging_rule):
    """
        Flattens the input values, then delegates.
    """
    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        flatargs = list(
            chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args])
        )
        out = flat_inject(closed_jaxpr.jaxpr, closed_jaxpr.literals, tagging_rule, *flatargs)
        return out
    return wrapped

def test_intergrad_equinox():
    model = eqxMLP(5, 10, 10, jax.random.PRNGKey(0))
    x, y = jnp.ones((10, 5)), jax.nn.one_hot(jnp.arange(10), num_classes=10)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def model_fn(params, x):
        model = eqx.combine(params, static)
        return model(x)
    
    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * jax.nn.log_softmax(logits)).mean()
        return loss
    
    jintergrad = jax.jit(intergrad(celoss, tagging_rule=None)) # can jit and vmap
    grads, activations = jax.vmap(jintergrad, in_axes=(None, 0, 0))(params, x, y)
    return grads, activations

def test_intergrad_nnx():
    model = nnx_mlp(5, 10, 10, nnx.Rngs(0))
    pmodel = p_nnx_mlp(5, 10, 10, nnx.Rngs(0))
    x, y = jnp.ones(5), jax.nn.one_hot(1, num_classes=10)

    # make sample run
    _ = pmodel(x)
    _ = model(x)

    graph, params = nnx.split(model)
    def model_fn(p, x):
        model = nnx.merge(graph, p)
        return model(x)
    
    @nnx.grad(argnums=nnx.DiffState(argnum=0, filter=nnx.Any(nnx.Param, nnx.Perturbation)))
    def p_celoss(model, x, y):
        preds = jax.nn.log_softmax(model(x))
        return -(preds * y).mean()

    def value_p_celoss(model, x, y):
        preds = jax.nn.log_softmax(model(x))
        return -(preds * y).mean()
    
    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * jax.nn.log_softmax(logits)).mean()
        return loss
    
    allgrads = p_celoss(pmodel, x, y)
    intm_grads = [allgrads.xgrad1.value, allgrads.xgrad2.value, allgrads.xgrad3.value]
    activations, grads = intergrad(celoss, tagging_rule=None)(params, x, y)

    print(all([jnp.allclose(a, b).all() for a, b in zip(intm_grads, grads)]))
    return activations, grads

class eqxMLP(eqx.Module):
    
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self, in_dim, mid_dim, out_dim, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_dim, mid_dim, key=key1)
        self.linear2 = eqx.nn.Linear(mid_dim, mid_dim, key=key2)
        self.linear3 = eqx.nn.Linear(mid_dim, out_dim, key=key3)

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)

        return x

class nnx_mlp(nnx.Module):

  def __init__(self, in_dim, mid_dim, out_dim, key):
    self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
    self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
    self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

  def __call__(self, x):

    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    x = jax.nn.relu(x)
    x = self.linear3(x)

    return x

class p_nnx_mlp(nnx.Module):

  def __init__(self, in_dim, mid_dim, out_dim, key):
    self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
    self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
    self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

  def __call__(self, x):

    x = self.linear1(x)
    x = self.perturb('xgrad1', x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    x = self.perturb('xgrad2', x)
    x = jax.nn.relu(x)
    x = self.linear3(x)
    x = self.perturb('xgrad3', x)

    return x



if __name__ == '__main__':
    # ff = intergrad(f, None)
    # print(ff(1.0))
    # print(jax.grad(f_pert, argnums=1)(1.0, [0.0, 0.0]))

    # g, a = test_intergrad_equinox()
    # print([gg.shape for gg in g])
    # print([aa.shape for aa in a])

    test_intergrad_nnx()

