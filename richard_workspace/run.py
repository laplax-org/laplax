from richard_workspace.dataloading import *
from richard_workspace.eval import *
from richard_workspace.innerprods import *
from richard_workspace.models import nnxMLP as MLP
from flax import nnx
from copy import deepcopy
from tqdm import tqdm

import gc
import json
import jax
import jax.tree as jt
import jax.random as jr
import jax.numpy as jnp

import optax


def step(model, optimizer, x, y, mode, inner_fn, _lambda):
  def regularization(model):
    if not mode:
      return 0
    
    _, params = nnx.split(model)
    distance = sub(params, mode)
    return inner_fn(distance)
  
  def cross_entropy(model, x, y):
    log_y_pred = jax.nn.log_softmax(model(x))
    return -(log_y_pred * y).mean()

  def lossfn(model, x, y):
    ce = cross_entropy(model, x, y)
    reg = regularization(model)
    return ce + _lambda * reg
  
  loss, grads = nnx.value_and_grad(lossfn)(model, x, y)
  optimizer.update(grads)

  return loss

def train_model(model, trainloader, optimizer, mode, inner_fn, num_epochs, _lambda):
  hist = []

  # jit the function once inside of the train_model scope instead of
  # using the @nnx.jit decorator on the function. This will make jax release
  # the cached version of the current step function when it's not used any more.
  step_jit = nnx.jit(step, static_argnames=['inner_fn'])
  for epoch in tqdm(range(num_epochs)):
    for xb, yb in trainloader:
      loss = step_jit(model=model, optimizer=optimizer, x=xb, y=yb, mode=mode, inner_fn=inner_fn, _lambda=_lambda)
      hist.append(loss.item())
  return hist

def run(*, seeds, _lambda, inner_fn_factory, dataset_fn, num_epochs=100, curv_samples=128, fn_kwargs : Dict ={}, **run_kwargs) -> Tuple[List[Any], List[Any]]:
    
    root_trainloader, root_testloader, numsample_train, numsample_test = dataset_fn()
    fn_kwargs.update({'numsamples_train' : numsample_train, 'numsamples_test' : numsample_test})
    model = MLP(layer_sizes=[28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(run_kwargs['lr'], weight_decay=5e-4))
    
    results = []
    loss_histories = []
    mode = None
    inner_fn = None
    for i, seed in enumerate(seeds):
        trainloader = permute(root_trainloader, seed=seed)
        hist = train_model(
          model=model, 
          trainloader=trainloader,
          optimizer=optimizer,
          mode=mode, inner_fn=inner_fn,
          num_epochs=num_epochs, _lambda=_lambda
        )

        # Evaluate on all previous and current test-sets
        loss_histories.append(hist)
        res = eval_model(model, root_testloader, seeds[:i+1])
        res = [r.item() for r in res]
        print(res)
        results.append(res)

        # Update mode and partial_hvp for next task (placeholder)
        mode = nnx.split(model)[1]
        inner_fn = inner_fn_factory(model, trainloader, maxsamples=curv_samples, **fn_kwargs)
        
    
    # clears jax caches. necessary to ensure that we're not caching all the different 
    # versions of step_jit compiled for one inner_fn.
    jax.clear_caches()
    
    return results, loss_histories

def run_reps(num_reps, num_tasks=5, **run_kwargs):
    rs, hs = [], []
    seeds = jnp.arange(num_tasks)
    for i in range(1, num_reps + 1):
      scaled_seeds = seeds * i
      r, h = run(seeds=scaled_seeds, **run_kwargs)
      rs.append(r); hs.append(h)
    return rs, hs

if __name__ == '__main__':
  """
    TODO: check and verify (?) KFAC implementation.
  """
  import os
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

  kwlist1, outname1 = [
    {"_lambda" : 0,     "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e-4,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e-3,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e-2,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e-1,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1.0,   "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e+1,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e+2,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e+3,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner},
    {"_lambda" : 1e+4,  "curv_samples": 128,      'inner_fn_factory' : type1_fisher_inner}
  ], "type1_fisher_lambdas"

  kwlist2, outname2 = [
    {"_lambda" : 0,     "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e-4,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e-3,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e-2,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e-1,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1.0,   "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e+1,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e+2,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e+3,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner},
    {"_lambda" : 1e+4,  "curv_samples": 128,      'inner_fn_factory' : emp_fisher_inner}
  ], "emp_fisher_lambdas"


  kwlist3, outname3 = [
    {"_lambda" : 0.0,   "curv_samples": 128,      'inner_fn_factory' : zero},
    {"_lambda" : 0,     "curv_samples": 128,      'inner_fn_factory' : ggn_inner},
    {"_lambda" : 1e-3,  "curv_samples": 128,      'inner_fn_factory' : ggn_inner},
    {"_lambda" : 1.0,   "curv_samples": 128,      'inner_fn_factory' : ggn_inner},
  ], "ggn_lambdas"

  kfackwargs, outname4 = [
    {"_lambda" : 5e+3,  "curv_samples": 128,      'inner_fn_factory' : kfac_inner_fn},
    {"_lambda" : 1e+4,   "curv_samples": 128,      'inner_fn_factory' : kfac_inner_fn},
    {"_lambda" : 1e+5,   "curv_samples": 128,      'inner_fn_factory' : kfac_inner_fn}
  ], "kfac_lambdas"


  defaults = {'num_epochs' : 64, 'lr' : 1e-4}
  for kwlist, name, dataset_fn in [(kfackwargs, outname4, minimnist)]:
    
      d = {}
      for kwds in kwlist:
        params = deepcopy(defaults)
        params.update(kwds)
        
        print(f"Starting setup: {d2name(params)}")

        try:
          raccu, hists = run_reps(num_reps=3, dataset_fn=dataset_fn, **params)
          d[d2name(params)] = {
            'raccu' : raccu,
            'hists' : hists
          }
        except Exception as e:
          print(f'Error in configuration: {name} on {dataset_fn.__name__}')
          print(str(e))
    
      with open(f'{name}_{dataset_fn.__name__}.json', 'w') as file:
        json.dump(fp=file, obj=d)
