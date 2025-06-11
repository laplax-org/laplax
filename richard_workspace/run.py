import json
import optax
import jax.numpy as jnp

from functools import partial

from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split
from laplax.util.tree import sub, dot, zeros_like, mul, add
from richard_workspace.helpers import *
from typing import *

@partial(nnx.jit, static_argnames=['inner_fn'])
def train_step(model, optimizer, x, y, mode, inner_fn, ell):

  def regularization(model):
    if not mode:
      return 0
    
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
  
  loss, grads = nnx.value_and_grad(loss)(model)
  optimizer.update(grads)

  return loss

# --- Fisher inner functions updated to use DataLoader batches ---
def emp_fisher_inner(model, trainloader, batch_size, *args, **kwargs):
  def cross_entropy(model, x, y):
    log_y_pred = jax.nn.log_softmax(model(x))
    return -(log_y_pred * y).mean()

  xb, yb = collect_numsamples_from_loader(trainloader, maxsamples=batch_size)
  grads = nnx.grad(cross_entropy)(model, xb, yb)
  sqgrads = jax.tree.map(lambda x : jnp.mean(x**2, axis=0), grads) # square and mean

  def inner(v):
    return dot(v, jax.tree.map(lambda x, y: x * y, v, sqgrads))
      
  return inner

def unscaled_dot_product(*args, **kwargs):
  def inner(v):
    return dot(v, v)
  return inner

def ggn_inner(model, trainloader, batch_size, numsamples_train, *args, **kwargs):
    
    graph_def, params = nnx.split(model)
    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]
    xbatch, ybatch = collect_numsamples_from_loader(trainloader, maxsamples=batch_size)
    data = input_target_split((xbatch, ybatch)) # sample batch

    partial_hvp = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn='cross_entropy',
        num_total_samples=numsamples_train
    )
    def inner(v):
      return dot(v, partial_hvp(v))
    
    return inner

def type1_fisher_inner(model, trainloader, batch_size, M=30, *args, **kwargs):

  def cross_entropy(model, x, y, *, key):
    logits = model(x)
    probs = jax.nn.softmax(logits)
    # Sample one label per example in the batch
    sampled_labels = jax.random.categorical(key, logits, axis=-1)
    ysample = jax.nn.one_hot(sampled_labels, num_classes=probs.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -(log_probs * ysample).mean()

  xb, yb = collect_numsamples_from_loader(trainloader, maxsamples=batch_size)
  running_mean = None
  key = jax.random.PRNGKey(0)

  # running mean accumulation of gradients
  for i in range(M):
    key, curkey = jax.random.split(key)
    grads = nnx.vmap(nnx.grad(partial(cross_entropy, key=curkey)), in_axes=(None, 0, 0))(model, xb, yb) # shape (batch, *param_sizes)
    grads = jax.tree.map(lambda x: (x**2).mean(0), grads)
    
    if running_mean is None:
      running_mean = grads
    else:
      running_mean = add(running_mean, mul(1/(i + 1), sub(grads, running_mean)))

  def inner(v):
    return dot(v, jax.tree.map(lambda x, y: x * y, v, running_mean))
  
  return inner

# --- Training and evaluation using torch DataLoader ---
def train_model(model, trainloader, optimizer, seed, mode, inner_fn, num_epochs, ell):
  hist = []
  for epoch in range(num_epochs):
    for xb, yb in trainloader:
      loss = train_step(model, optimizer, xb, yb, mode, inner_fn=inner_fn, ell=ell)
      hist.append(loss.item())
  return hist  
  
def run(seeds, ggn_batch_size, ell, fisher_fn, num_epochs=100, fn_kwargs : Dict ={}) -> Tuple[List[Any], List[Any]]:
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    fn_kwargs.update({'numsamples_train' : numsamples_train, 'numsamples_test' : numsamples_test})

    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))

    inner_fn = mode = None
    
    results = []
    loss_histories = []
    for i, seed in enumerate(seeds):
        trainloader = torch_permute_data(_trainloader, seed=seed)
        hist = train_model(
          model=model, 
          trainloader=trainloader,
          optimizer=optimizer,
          mode=mode, inner_fn=inner_fn,
          num_epochs=num_epochs, ell=ell, seed=seed
        )

        # Evaluate on all previous and current test-sets
        loss_histories.append(hist)
        res = torch_evaluate_model(model, _testloader, seeds[:i+1])
        res = [r.item() for r in res]
        results.append(res)

        # Update mode and partial_hvp for next task (placeholder)
        mode = nnx.split(model)[1]
        inner_fn = fisher_fn(model, torch_permute_data(_trainloader, seed=seed), batch_size=ggn_batch_size, **fn_kwargs)

    return results, loss_histories

def train_for_different_batchsizes(ell, batch_sizes, num_repetitions, fisher_fn=emp_fisher_inner, fn_kwargs={}):

  print(f'Running ell: {ell}, batch_size: {batch_sizes}, num_rep: {num_repetitions}')

  raccu, hists_accu = \
  { f"batch_size:{bs}" : [] for bs in batch_sizes}, { f"batch_size:{bs}" : [] for bs in batch_sizes}

  for ggn_batch_size in batch_sizes:
    for k in range(1, num_repetitions + 1):
      seeds = jnp.arange(5) * k
      r, hists = run(seeds=seeds, ggn_batch_size=ggn_batch_size, ell=ell, fisher_fn=fisher_fn, fn_kwargs=fn_kwargs)

      raccu[f"batch_size:{ggn_batch_size}"].append(r)
      hists_accu[f"batch_size:{ggn_batch_size}"].append(hists)
      print(f"Iter: {k}, ggn_batchsize: {ggn_batch_size} \n", "\n".join([str(rr) for rr in r]))
  
  return raccu, hists_accu
  
if __name__ == '__main__':
  """
    I do get very bad results for the empirical fisher, which i cannot explain currently,
    especially because the type1 fisher should be equivalent to the ggn under our assumptions
      --> recheck whether thats really true.
      
  """
  ells = [1e-4]
  batch_sizes = [32, 512]
  num_repetitions = 3

  for _type, fn in {'type1_fisher' : type1_fisher_inner, 'ggn' : ggn_inner}.items(): 
    d = dict()
    for ell in ells:
      raccu, hists = train_for_different_batchsizes(ell=ell, 
                                                    batch_sizes=batch_sizes, 
                                                    num_repetitions=num_repetitions,
                                                    fisher_fn=fn)
      d[f'ell:{ell}'] = {
        'raccu' : raccu, 
        'hists' : hists
      }
    
    with open(f'{_type}.json', 'w') as file:
      json.dump(d, file)
