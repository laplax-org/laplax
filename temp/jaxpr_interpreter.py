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

def eval_jaxpr(jaxpr, consts, *args):
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

        outvals, _ = eval_jaxpr(subjaxpr, sub_consts, *invals)
    else:
        outvals = eqn.primitive.bind(*invals, **eqn.params)

    if not eqn.primitive.multiple_results:
        outvals = [outvals]

    safe_map(write, eqn.outvars, outvals)
  # Read the final result of the Jaxpr from the environment
  return safe_map(read, jaxpr.outvars)

def splice_jaxpr(jaxpr, consts, *args):
    """
        This jaxpr interpreter splices the given jaxpr into a set of jaxprs.

        We want to be able to compute the intermediate gradients of the function f(params, x) = y
        w.r.t. the inputs x.
        More precisely, we'd like the gradients w.r.t. pre-activations Wx_{t-1} + b = x_t. \partial L / \partial x_t
        for all x_t in the model. This function takes the whole computation, which may look something like:
        
        let relu = { lambda ; a:f32[10]. let b:f32[10] = max a 0.0:f32[] in (b,) } in
        { lambda ; c:f32[10,5] d:f32[10] e:f32[10,10] f:f32[10] g:f32[10,10] h:f32[10] i:f32[5]
            j:f32[10]. let
            k:f32[10] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=float32
            ] c i
            l:f32[10] = add k d                                                 <<<<
            m:f32[10] = custom_jvp_call[
            name=relu
            call_jaxpr={ lambda ; n:f32[10]. let
                o:f32[10] = pjit[name=relu jaxpr=relu] n
                in (o,) }
            jvp=jvp
            symbolic_zeros=False
            ] l
            p:f32[10] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=float32
            ] e m
            q:f32[10] = add p f                                                 <<<<
            r:f32[10] = custom_jvp_call[
            name=relu
            call_jaxpr={ lambda ; s:f32[10]. let
                t:f32[10] = pjit[name=relu jaxpr=relu] s
                in (t,) }
            jvp=jvp
            symbolic_zeros=False
            ] q
            u:f32[10] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=float32
            ] g r
            v:f32[10] = add u h                                                 <<<<
            w:f32[10] = mul j v
            x:f32[] = reduce_sum[axes=(0,)] w
            y:f32[] = div x 10.0:f32[]
            z:f32[] = neg y
        in (z,) }

        and creates partial functions at the marked "<<<<" positions.
    """

    env = {}
    partials = []

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
    def make_partial_fn(idx):
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
        pass

    for i, eqn in enumerate(jaxpr.eqns):
        invals = safe_map(read, eqn.invars)
        if "call_jaxpr" in eqn.params:
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, 'consts') else ()
            outvals = eval_jaxpr(subjaxpr, sub_consts, *invals) # assuming that we're just encountering relus
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        if eqn.primitive.name == 'add':
            partials.append(make_partial_fn(i))
        safe_map(write, eqn.outvars, outvals)

    return partials

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
        flatargs = list(
            chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args])
        )
        out = log_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *flatargs)
        return out
    return wrapped

def splice(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        flatargs = list(
            chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args])
        )
        out = splice_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *flatargs)
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

def laplax_modelfn_test():

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

    jaxprs = splice(celoss)(params, x, y)

    # log_fn = log_activations(model_fn)
    # out, accu = celoss(params, x, y)
    # print(out, accu)

if __name__ == "__main__":
    with jax.disable_jit():
        laplax_modelfn_test()