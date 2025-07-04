from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Any

import jax
import jax.numpy as jnp
from jax._src.util import safe_map  # noqa: PLC2701
from jax.extend import core
from jax.extend.core import Jaxpr
from jaxtyping import Array

"""
    Following: https://github.com/jax-ml/jax/discussions/5336

    1) go through the function and inject "perturbation" additions where needed.
    2) use regular "grad" to collect gradients
"""


def flat_eval_jaxpr(jaxpr: Jaxpr, consts: list[Any], *args) -> list[Array]:
    """Evaluates a JaxPr.

    Args:
        jaxpr (Jaxpr): The Jaxpr to be evaluated.
        consts (List[Any]): Constants used in the Jaxpr.
        args: invars to jaxpr

    Returns:
        List[Array]: The output values of the Jaxpr.
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

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)

        if "call_jaxpr" in eqn.params:  # handle custom definitions (i.e. jax.nn.relu)
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()

            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()

            outvals = flat_eval_jaxpr(subjaxpr, sub_consts, *invals)
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)

        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        safe_map(write, eqn.outvars, outvals)
    # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)


def perturb(
    jaxpr: Jaxpr, consts: list[Any], perturbations: list[Array], *args
) -> Array:
    """To be differentiated w.r.t. perturbations.

    Args:
        jaxpr (Jaxpr): The Jaxpr to be evaluated.
        consts (List[Any]): Constants used in the Jaxpr.
        perturbations (List[Array]): Perturbations to be added to the correct JaxprEqn.
        args: invars to jaxpr

    Returns:
        Regular output of the Jaxpr.
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

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if "call_jaxpr" in eqn.params:  # if its a relu, record the activation
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()
            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()
            outvals = flat_eval_jaxpr(
                subjaxpr, sub_consts, *invals
            )  # assuming that we're just encountering relus
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        if eqn.primitive.name == "add":
            pert = perturbations.pop(0)
            outvals[0] = outvals[0] + pert
        safe_map(write, eqn.outvars, outvals)
    return safe_map(read, jaxpr.outvars)[0]


def flat_inject(
    jaxpr: Jaxpr, consts: list[Any], tagging_rule: Callable | None, *args
) -> tuple[list[Array], list[Array]]:
    """Injects perturbations into the Jaxpr and collects intermediate acts and grads.

    Args:
        jaxpr (Jaxpr): The Jaxpr to be transformed.
        consts (List[Any]): Constants used in the Jaxpr.
        tagging_rule (Callable | None): A function that defines how to tag the
            intermediate variables.
        args: invars to jaxpr

    Returns:
        Tuple[List[Array], List[Array]]: A tuple containing:
            - A list of post-relu activations.
            - A list of gradients of w.r.t. the pre-relu activations.

    We step through the jaxpr and collect post-relu activations.
    Additionally, we construct another jaxpr in which we perturb the weights.
    This jaxpr is interpreted as a function depending on the perturbation inputs and
    can be called with jax.grad to obtain the intermediate gradients.

    TODO: Use `tagging_rule` to tag the correct variables for perturbation.
    """
    del tagging_rule  # unused
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
        if "call_jaxpr" in eqn.params:  # if its a relu, record the activation
            subjaxpr = eqn.params["call_jaxpr"]
            sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()
            if type(subjaxpr) is core.ClosedJaxpr:
                subjaxpr = subjaxpr.jaxpr
                sub_consts = subjaxpr.consts if hasattr(subjaxpr, "consts") else ()
            outvals = flat_eval_jaxpr(
                subjaxpr, sub_consts, *invals
            )  # assuming that we're just encountering relus
            activations.append(outvals[0])
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        if eqn.primitive.name == "add":
            perturbations.append(jnp.zeros_like(outvals[0]))

        safe_map(write, eqn.outvars, outvals)

    def perturbed_fn(perts, *args):
        return perturb(jaxpr, consts, perts, *args)

    grads = jax.grad(perturbed_fn)(perturbations, *args)

    return activations, grads


def intergrad(fun: Callable, tagging_rule: Callable | None = None) -> Callable:
    r"""Function transformation that captures intermediate activations and gradients.

    Args:
        fun (Callable): The function to be transformed. It should accept
            parameters and inputs.
        tagging_rule (Callable | None): A function that defines how to
            tag the intermediate variables.

    Returns:
        Callable: A function that, when called, returns the intermediate
        activations and gradients.

    Usage:
        ```python
        graph, params = nnx.split(model)
        def celoss(params, x, y):
            logits = nnx.merge(graph, params)(x)
            return -(y * jax.nn.log_softmax(logits)).mean()

        intergrad_jit = jax.jit(jax.vmap(intergrad(celoss, tagging_rule=None), \\
                            in_axes=(None, 0, 0)))
        acts, grads = intergrad_jit(celoss)(params, x, y)
        ```

    """
    del tagging_rule  # unused

    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        flatargs = list(chain.from_iterable([jax.tree.flatten(arg)[0] for arg in args]))
        out = flat_inject(closed_jaxpr.jaxpr, closed_jaxpr.literals, None, *flatargs)
        return out

    return wrapped
