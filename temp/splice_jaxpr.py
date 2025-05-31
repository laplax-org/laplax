import time

import jax
import jax.numpy as jnp
import equinox as eqx

from itertools import chain
from functools import partial
from jax import jit, grad, vmap
from jax import random

# Importing Jax functions useful for tracing/interpreting.
from functools import wraps

from jax import lax
from jax.extend import core
from jax._src.util import safe_map
from jax.extend.core import Jaxpr, JaxprEqn, Var
from jax._src.core import Atom, jaxpr_as_fun
from typing import Sequence

def flat_eval_jaxpr(jaxpr, consts, *args):
  # Mapping from variable -> value
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

def flat_splice_jaxpr(jaxpr : Jaxpr, consts : Sequence[Atom], *args):
    env = {}
    partials, intermediates, activations = [], [], []

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Helper to create a partial function that computes up to a given eqn index
    def make_partial_jaxpr(idx):
        """
            We're assuming this function is called at the point of a primitive addition.
            This means the following equation will be an application of the activation function.
            We want to transform the computation graph

            c = dot a b
            d = add c e <<< idx
            f = relu(d)
            ...

            into a jaxpr which takes in the remaining arguments (params) and an intermediate result (d) and
            computes the remainder of the jaxpr. The outvar of the current addition becomes an invar of the
            new jaxpr.
        """
        constvars, outvars, eqns = jaxpr.constvars,  jaxpr.outvars, jaxpr.eqns[idx + 1:]
        
        invars = jaxpr.invars[:-2] + jaxpr.eqns[idx].outvars + [jaxpr.invars[-1]]

        new_jaxpr = Jaxpr(constvars=constvars, invars=invars, outvars=outvars, eqns=eqns)
        return new_jaxpr

    for i, eqn in enumerate(jaxpr.eqns):
        invals = safe_map(read, eqn.invars)
        if "call_jaxpr" in eqn.params:
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
            partials.append(make_partial_jaxpr(i))
            intermediates.append(outvals[0])
        safe_map(write, eqn.outvars, outvals)

    return partials, intermediates, activations

def splice(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        flatargs = list(
            chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args])
        )
        out = flat_splice_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *flatargs)
        return out
    return wrapped

def eval_jaxpr(jaxpr, consts, *args):
    flatargs = list(
        chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args])
    )
    return flat_eval_jaxpr(jaxpr, consts, *flatargs)

def test_splice():
    model = MLP(5, 10, 10, jax.random.PRNGKey(0))
    x, y = jnp.ones(5), jax.nn.one_hot(1, num_classes=10)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def model_fn(params, x):
        model = eqx.combine(params, static)
        return model(x)
    
    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * logits).mean()
        return loss
    
    jaxprs, intermediates, activations = splice(celoss)(params, x, y)
    print([eval_jaxpr(j, [], params, i, y) for j,i in zip(jaxprs, intermediates)])

    def jaxpr_as_fun(jaxpr, params, x, y):
        return eval_jaxpr(jaxpr, [], params, x, y)[0]
    
    print(jax.grad(jaxpr_as_fun, argnums=2)(jaxprs[0], params, intermediates[0], y)) # retrieves the gradient w.r.t. intermediate

class MLP(eqx.Module):
    
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

if __name__ == "__main__":
    test_splice()