import json
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial

from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split
from laplax.util.tree import sub, dot, zeros_like, mul, add
from richard_workspace.helpers import *
from typing import *

def cross_entropy(model, x, y):
    log_y_pred = jax.nn.log_softmax(model(x))
    return -(log_y_pred * y).mean()

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(cross_entropy)(model, x, y)
    optimizer.update(grads)

    return loss

def train_model(model, trainloader, optimizer, num_epochs):
  hist = []
  for epoch in range(num_epochs):
    for xb, yb in trainloader:
      loss = train_step(model, optimizer, xb, yb)
      hist.append(loss.item())
  return hist  
  
def run(num_epochs=100) -> Tuple[List[Any], List[Any]]:
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    KFAC_SAMPLES = 128
    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))
    trainloader = torch_permute_data(_trainloader, seed=0)

    hist = train_model(
        model=model, 
        trainloader=trainloader,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    # get activations and gradients
    from temp.intergrad import intergrad

    _graph, _params = nnx.split(model)

    def model_fn(p, x):
       model = nnx.merge(_graph, p)
       return model(x)
    
    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * jax.nn.log_softmax(logits)).mean() # reduction factor -> 1 / (dim(Y) * batch_size)
        return loss
    
    x, y = collect_numsamples_from_loader(torch_permute_data(_trainloader, seed=0), maxsamples=KFAC_SAMPLES)
    intergrad_mapped = jax.vmap(jax.jit(intergrad(celoss, tagging_rule=None)), in_axes=(None, 0, 0))
    activations, grads = intergrad_mapped(_params, x, y)
    print(jax.tree.map(lambda x: x.shape, activations), jax.tree.map(lambda x: x.shape, grads)) # [(128, 10)] [(128, 10), (128, 10)]

    """
        Following F.Dangel's KFAC tutorial, constructing the KFAC is split up into the following steps:
        1) Perform forward pass, collect inputs.
            -> We still have to pre-pend x to the activations list.
        2)  Compute A^i using the layer inputs.
        3)  Get the backward gradients, compute B^i
        4)  Account for scaling introduced by the loss function.
        5)  Return KFAC for all layers i


        We already have all inputs and gradients. The only thing left is to apply scaling introduced by the loss function
        and to compute the KFAC blocks.
    """

    activations = [x] + activations
    # have to append ones to capture the bias. See F.Dangel page 46 torch code.
    activations = [jnp.concat([arr, jnp.ones((arr.shape[0], 1))], axis=-1) for arr in activations] # append ones along the feature dimension

    def reduce_outers(arrs, factor):
       return jnp.sum(jax.vmap(lambda x: jnp.outer(x, x))(arrs), axis=0) * factor
    
    R = 1 / (KFAC_SAMPLES * y.shape[-1])
    As, Bs = [reduce_outers(arrs, factor=R) for arrs in activations], [reduce_outers(arrs, factor=1/arrs.shape[0]) for arrs in grads]
    blocks = [jax.numpy.kron(a, b) for a, b in zip(As, Bs)]
    
    """
        blocks: [(7850, 7850), (110, 110)], biases included.
    """

    def kfac_inner(v):
        """
            Assume v has pytree like structure. We should flatten & concatenate, then
            split according to the kfac block structure and mutliply out.
        """
        arrs = jax.tree.leaves(v)
        weights, biases = arrs[1::2], arrs[0::2] # assume that we have only linear layers with biases.
        vsplit = [jnp.concat([w.reshape(-1), b]) for w, b in zip(weights, biases)] # flatten the parameter tensors
        Bv = [bi @ vi for bi, vi in zip(blocks, vsplit)]
        vtBv = sum([jnp.dot(vv, bv) for vv, bv in zip(vsplit, Bv)])
        return vtBv

    print(kfac_inner(_params))

# --- Continual Learning Experiment Setup (KFAC) ---
@partial(nnx.jit, static_argnames=['inner_fn'])
def train_step_kfac(model, optimizer, x, y, mode, inner_fn, ell):
    def regularization(model):
        if mode is None:
            return 0.0
        _, params = nnx.split(model)
        distance = sub(params, mode)
        return inner_fn(distance)

    def cross_entropy(model):
        log_y_pred = jax.nn.log_softmax(model(x))
        return -(log_y_pred * y).mean()

    def loss(model):
        ce = cross_entropy(model)
        reg = regularization(model)
        return ce + ell * reg

    loss_val, grads = nnx.value_and_grad(loss)(model)
    optimizer.update(grads)
    return loss_val


def kfac_inner_fn(model, trainloader, batch_size, KFAC_SAMPLES=128, *args, **kwargs):
    # Collect activations and grads as in the main run()
    _graph, _params = nnx.split(model)
    def model_fn(p, x):
        model = nnx.merge(_graph, p)
        return model(x)
    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * jax.nn.log_softmax(logits)).mean()
        return loss
    x, y = collect_numsamples_from_loader(trainloader, maxsamples=KFAC_SAMPLES)
    from temp.intergrad import intergrad
    intergrad_mapped = jax.vmap(jax.jit(intergrad(celoss, tagging_rule=None)), in_axes=(None, 0, 0))
    activations, grads = intergrad_mapped(_params, x, y)
    activations = [x] + activations
    activations = [jnp.concatenate([arr, jnp.ones((arr.shape[0], 1))], axis=-1) for arr in activations]
    def reduce_outers(arrs, factor):
        return jnp.sum(jax.vmap(lambda x: jnp.outer(x, x))(arrs), axis=0) * factor
    R = 1 / (KFAC_SAMPLES * y.shape[-1])
    As = [reduce_outers(arrs, factor=R) for arrs in activations]
    Bs = [reduce_outers(arrs, factor=1/arrs.shape[0]) for arrs in grads]
    blocks = [jax.numpy.kron(a, b) for a, b in zip(As, Bs)]
    def kfac_inner(v):
        arrs = jax.tree.leaves(v)
        weights, biases = arrs[1::2], arrs[0::2]
        vsplit = [jnp.concatenate([w.reshape(-1), b]) for w, b in zip(weights, biases)]
        Bv = [bi @ vi for bi, vi in zip(blocks, vsplit)]
        vtBv = sum([jnp.dot(vv, bv) for vv, bv in zip(vsplit, Bv)])
        return vtBv
    return kfac_inner


def train_model_kfac(model, trainloader, optimizer, mode, inner_fn, num_epochs, ell):
    hist = []
    for epoch in range(num_epochs):
        for xb, yb in trainloader:
            loss = train_step_kfac(model, optimizer, xb, yb, mode, inner_fn, ell)
            hist.append(loss.item())
    return hist


def run_kfac(seeds, kfac_batch_size, ell, num_epochs=100, fn_kwargs: Dict = {}):
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    fn_kwargs.update({'KFAC_SAMPLES': kfac_batch_size})
    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))
    inner_fn, mode = None, None
    results = []
    loss_histories = []
    for i, seed in enumerate(seeds):
        trainloader = torch_permute_data(_trainloader, seed=seed)
        hist = train_model_kfac(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            mode=mode, inner_fn=inner_fn,
            num_epochs=num_epochs, ell=ell
        )
        loss_histories.append(hist)
        res = torch_evaluate_model(model, _testloader, seeds[:i+1])
        res = [r.item() for r in res]
        results.append(res)
        mode = nnx.split(model)[1]
        inner_fn = kfac_inner_fn(model, torch_permute_data(_trainloader, seed=seed), batch_size=kfac_batch_size, **fn_kwargs)
    return results, loss_histories


def train_setup_kfac(ell, batch_size, num_repetitions, fn_kwargs={}):
    raccu, hists_accu = [], []
    for k in range(1, num_repetitions + 1):
        seeds = jnp.arange(5) * k
        r, hists = run_kfac(seeds=seeds, kfac_batch_size=batch_size, ell=ell, fn_kwargs=fn_kwargs)
        raccu.append(r)
        hists_accu.append(hists)
    return raccu, hists_accu


if __name__ == '__main__':
    NUM_REP = 3
    kwlist = [
        {'ell': 0.0,   'batch_size': 128},
        {'ell': 1e+1,  'batch_size': 128},
        {'ell': 1e+2,  'batch_size': 128},
        {'ell': 3e+2,  'batch_size': 128},
        {'ell': 1e+3,  'batch_size': 128},
        {'ell': 3e+3,  'batch_size': 128},
    ]

    d = {}
    for kwds in kwlist:
        print(f"Starting setup: {kwds}")
        raccu, hists = train_setup_kfac(num_repetitions=NUM_REP, fn_kwargs={}, **kwds)
        d[f'ell:{kwds['ell']}batch_size:{kwds['batch_size']}'] = {
            'raccu': raccu,
            'hists': hists
        }
    with open(f'kfac_lambda2.json', 'w') as file:
        json.dump(d, file)
