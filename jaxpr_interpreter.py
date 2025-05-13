import time

import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial
from jax import jit, grad, vmap
from jax import random

# Importing Jax functions useful for tracing/interpreting.
from functools import wraps

from jax import lax
from jax.extend import core
from jax._src.util import safe_map

def log_jaxpr(jaxpr, consts, *args):
    """
        Given a jax expression which contains dot-products between nn inputs and
        weights, log the activations of that computation.

        Returns:
            accumulator containing the intermediate expressions.
    """
    # Mapping from variable -> value
    env = {}
    accu = []

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

            outvals, _ = log_jaxpr(subjaxpr, sub_consts, *invals)
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)

        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        
        # first quick and dirty hack: assume the linear layers are always using bias.
        if eqn.primitive.name == 'add': # bias addition of linear layer detected
            assert len(outvals) == 1, "Assertion that every add in jaxpr is unary output failed."
            accu.append(outvals[0]) # addition output is unary. Just take the first outval

        safe_map(write, eqn.outvars, outvals)

    return safe_map(read, jaxpr.outvars), accu

def log_activations(fun):

    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = log_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out
    
    return wrapped

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

def model_fn(params, static):
    model = eqx.combine(params, static)
    return lambda x: model(x)

def test_log():
    model = MLP(5, 10, 2, jax.random.PRNGKey(0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    fn = model_fn(params, static)
    log_fn = log_activations(fn)
    out, accu = log_fn(jnp.ones(5))
    print(out, accu)

def test_vmap_log():
    model = MLP(28 * 28, 100, 10, jax.random.PRNGKey(0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    fn = model_fn(params, static)
    vmap_fn = jax.vmap(fn)

    log_fn = jax.jit(log_activations(vmap_fn))
    times = []
    for _ in range(100):
        batches = jnp.ones((100, 32, 28 * 28))
        start = time.time()
        for b in batches:
            out, accu = log_fn(b)
        end = time.time()
        times.append(end - start)
    
    print("Test VMAP with log: mean: ", jnp.mean(jnp.array(times)), "std: ", jnp.std(jnp.array(times)))

def test_vmap():
    model = MLP(28 * 28, 100, 10, jax.random.PRNGKey(0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    fn = model_fn(params, static)
    vmap_fn = jax.jit(jax.vmap(fn))

    times = []
    for _ in range(100):
        batches = jnp.ones((100, 32, 28 * 28))
        start = time.time()
        for b in batches:
            out = vmap_fn(b)
        end = time.time()
        times.append(end - start)
    
    print("Test VMAP without log: mean: ", jnp.mean(jnp.array(times)), "std: ", jnp.std(jnp.array(times)))

if __name__ == "__main__":

    test_vmap_log()
    test_vmap()