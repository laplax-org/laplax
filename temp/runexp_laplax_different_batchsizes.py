import json
import optax
import jax.numpy as jnp

from functools import partial

from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split
from laplax.util.tree import sub, dot, zeros_like
from richard_workspace.helpers import *
from typing import *

@partial(nnx.jit, static_argnames=['partial_hvp'])
def laplax_train_step(model, optimizer, x, y, mode, partial_hvp, ell=0):

  def scaled_distance_to_prev_mode(model):
    if not mode:
      return 0

    _, model_state = nnx.split(model)
    d = sub(model_state, mode)
    hvp_v = partial_hvp(d)
    ret = dot(d, hvp_v)
    return ret

  def cross_entropy(model):
    log_y_pred = jax.nn.log_softmax(model(x))
    return -(log_y_pred * y).mean()

  def loss(model):
    ce = cross_entropy(model)
    reg = scaled_distance_to_prev_mode(model)
    return ce + ell * reg
  
  loss, grads = nnx.value_and_grad(loss)(model)
  optimizer.update(grads)

  return loss

def laplax_compute_hvp(model, trainloader, batch_size, num_total_samples):
    
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
        num_total_samples=num_total_samples
    )
    
    return partial_hvp

def train_model(model, trainloader, optimizer, mode, partial_hvp, num_epochs, ell):
  hist = []
  for kk in range(num_epochs):
    for xb, yb in trainloader:
        loss = laplax_train_step(model, optimizer, xb, yb, mode, partial_hvp, ell=ell)
        hist.append(loss.item())
  return hist  
  
def run_laplax(seeds, ggn_batch_size, ell, num_epochs=100) -> Tuple[List[Any], List[Any]]:
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))

    mode = partial_hvp = None
    
    results = []
    loss_histories = []
    for i, seed in enumerate(seeds):
        trainloader = torch_permute_data(_trainloader, seed=seed)
        hist = train_model(
          model=model, 
          trainloader=trainloader,
          optimizer=optimizer,
          mode=mode, partial_hvp=partial_hvp,
          num_epochs=num_epochs, ell=ell
        )

        # Evaluate on all previous and current test-sets (not implemented here)
        loss_histories.append(hist)
        res = torch_evaluate_model(model, _testloader, seeds[:i+1])
        res = [r.item() for r in res]
        results.append(res)

        # Update mode and partial_hvp for next task (placeholder)
        mode = nnx.split(model)[1]
        partial_hvp = laplax_compute_hvp(model, trainloader=torch_permute_data(_trainloader, seed=seed), batch_size=ggn_batch_size, num_total_samples=numsamples_train)

    return results, loss_histories

def train_for_different_batchsizes(ell, batch_sizes, num_repetitions):

  print(f'Running ell: {ell}, batch_size: {batch_sizes}, num_rep: {num_repetitions}')

  raccu, hists_accu = \
  { f"batch_size:{bs}" : [] for bs in batch_sizes}, { f"batch_size:{bs}" : [] for bs in batch_sizes}

  for ggn_batch_size in batch_sizes:
    for k in range(1, num_repetitions + 1):
      seeds = jnp.arange(5) * k
      r, hists = run_laplax(seeds=seeds, ggn_batch_size=ggn_batch_size, ell=ell)

      raccu[f"batch_size:{ggn_batch_size}"].append(r)
      hists_accu[f"batch_size:{ggn_batch_size}"].append(hists)
      print(f"Iter: {k}, gnn_batchsize: {ggn_batch_size} \n", "\n".join([str(rr) for rr in r]))
  
  return raccu, hists_accu
  
if __name__ == '__main__':
  """
    NOTES: For ell=0.1, regular updates i get 
    gnn_batchsize: 32 
    [[0.8630191683769226], 
    [0.8444488644599915, 0.8418530225753784], 
    [0.7773562073707581, 0.8173921704292297, 0.835563063621521], 
    [0.7585862278938293, 0.7648761868476868, 0.8098043203353882, 0.8535343408584595], 
    [0.7011780738830566, 0.7128593921661377, 0.745207667350769, 0.8376597166061401, 0.851138174533844]]


    For ell=0.0 i get
    gnn_batchsize: 32 
    [[0.8630191683769226], 
    [0.6575478911399841, 0.8184903860092163], 
    [0.4871205985546112, 0.5252595543861389, 0.49790334701538086], 
    [0.2614816129207611, 0.31220048666000366, 0.24231229722499847, 0.09794329106807709], 
    [0.4091453552246094, 0.42801517248153687, 0.1631389707326889, 0.16523562371730804, 0.7391173839569092]]
    These results correlate with other's findings.

    Some weirdness however: In the mnist_permute, which should work equivalently, i get completely different results for ell=0.0 and ell=0.1 respectively... Why?
  """
  ells = [0.0, 1e-5, 1e-4, 1e-3]
  batch_sizes = [128]
  num_repetitions = 5
  d = dict()
  for ell in ells:
    raccu, hists = train_for_different_batchsizes(ell=ell, batch_sizes=batch_sizes, num_repetitions=num_repetitions)
    d[f'ell:{ell}'] = {
      'raccu' : raccu, 
      'hists' : hists
    }
  
  with open('test.json', 'w') as file:
    json.dump(d, file)
