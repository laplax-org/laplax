from richard_workspace.dataloading import *
from richard_workspace.eval import *
from richard_workspace.innerprods import *
from richard_workspace.models import nnxMLP as MLP
from flax import nnx
from copy import deepcopy

import json
import jax
import jax.tree as jt
import jax.random as jr
import jax.numpy as jnp

import optax

@partial(nnx.jit, static_argnames=['inner_fn'])
def train_step(model, optimizer, x, y, mode, inner_fn, _lambda):

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

  for epoch in range(num_epochs):
    for xb, yb in trainloader:
      loss = train_step(model, optimizer, xb, yb, mode, inner_fn=inner_fn, _lambda=_lambda)
      hist.append(loss.item())
  return hist

def run(*, seeds, _lambda, inner_fn_factory, dataset_fn, num_epochs=100, curv_samples=128, fn_kwargs : Dict ={}, **run_kwargs) -> Tuple[List[Any], List[Any]]:
    root_trainloader, root_testloader, numsamples_train, numsamples_test = dataset_fn()
    fn_kwargs.update({'numsamples_train' : numsamples_train, 'numsamples_test' : numsamples_test})

    model = MLP(layer_sizes=[28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(run_kwargs['lr'], weight_decay=5e-4))

    inner_fn, mode = None, None
    
    results = []
    loss_histories = []
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
        results.append(res)

        # Update mode and partial_hvp for next task (placeholder)
        mode = nnx.split(model)[1]
        inner_fn = inner_fn_factory(model, trainloader, maxsamples=curv_samples, **fn_kwargs)

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
    14.06 TODO:
    - ablate curv samples again just to be sure.
    - ablate learning rate and num epochs


  """

  kwlist = [
    {"_lambda" : 0, "curv_samples": 16, 'inner_fn_factory' : zero},
    {"_lambda" : 1e-3, "curv_samples": 128, 'inner_fn_factory' : unscaled_dot_product},
    {"_lambda" : 1e-3, "curv_samples": 128, 'inner_fn_factory' : ggn_inner},
    {"_lambda" : 1e-3, "curv_samples": 512, 'inner_fn_factory' : ggn_inner},
  ]


  inner_name_map = {
    zero : "None",
    ggn_inner : "GGN",
    type1_fisher_inner : "Type1Fisher",
    emp_fisher_inner : "EmpFisher",
    unscaled_dot_product : "DotProduct",
  }
  dataset_name_map = {
    minimnist : "Mini-MNIST",
    mnist : "MNIST",
    fashion_mnist : "Fashion-MNIST"
  }

  inner_fn_factory = unscaled_dot_product
  dataset_fn = minimnist

  defaults = {'num_epochs' : 64, 'lr' : 1e-4}

  d = {}
  for kwds in kwlist:
    params = deepcopy(defaults)
    params.update(kwds)
    
    print(f"Starting setup: {params}")

    raccu, hists = run_reps(num_reps=3, dataset_fn=dataset_fn, **params)
    d[d2name(params)] = {
      'raccu' : raccu,
      'hists' : hists
    }

  name = 'unscaled_ggn_comparison' #f"dataset:{dataset_name_map[dataset_fn]}_inner:{inner_name_map[inner_fn_factory]}"
  with open(f'{name}.json', 'w') as file:
    json.dump(fp=file, obj=d)
  

      