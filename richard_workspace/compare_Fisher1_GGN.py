import json
import optax
import jax.numpy as jnp

from functools import partial

from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split
from laplax.util.tree import sub, dot, zeros_like, mul, add
from laplax.util.mv import diagonal
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
  sqgrads = jax.tree.map(lambda x : jnp.mean(x, axis=0)**2, grads) # mean and square

  def inner(v):
    return dot(v, jax.tree.map(lambda x, y: x * y, v, sqgrads))
      
  return inner

def unscaled_dot_product(*args, **kwargs):
  def inner(v):
    return dot(v, v)
  return inner

def ggn_diagonal(model, trainloader, batch_size, numsamples_train, *args, **kwargs):
    
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
    
    diag = diagonal(partial_hvp, params, mv_jittable=False)

    return diag

def type1_fisher_diagonal(model, trainloader, batch_size, M=30, *args, **kwargs):

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
  
  return running_mean

# --- Training and evaluation using torch DataLoader ---
def train_model(model, trainloader, optimizer, seed, mode, inner_fn, num_epochs, ell):
  hist = []
  for epoch in range(num_epochs):
    for xb, yb in trainloader:
      loss = train_step(model, optimizer, xb, yb, mode, inner_fn=inner_fn, ell=ell)
      hist.append(loss.item())
  return hist  

if __name__ == '__main__':
    """
        I do get very bad results for the empirical fisher, which i cannot explain currently,
        especially because the type1 fisher should be equivalent to the ggn under our assumptions
        
        -> Train the model to convergence. Then compute the GGN and the Fisher1 Diagonals
        and plot them/compute the correlation. They should ~about co-incide.
    """
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))

    inner_fn = mode = None

    results = []
    loss_histories = []
    trainloader = torch_permute_data(_trainloader, seed=0)
    hist = train_model(
        model=model, 
        trainloader=trainloader,
        optimizer=optimizer,
        mode=mode, inner_fn=inner_fn,
        num_epochs=1000, ell=0.0, seed=0
    )
    
    # compute the GGN and the FIsher1
    g = ggn_diagonal(model, trainloader=torch_permute_data(_trainloader, seed=0), numsamples_train=numsamples_train, batch_size=32)
    f1_diag = type1_fisher_diagonal(model, trainloader=torch_permute_data(_trainloader, seed=0), batch_size=32, M=1024)
    f = jnp.concat(jax.tree.map(lambda x: x.reshape(-1), jax.tree.leaves(f1_diag)))

    f = (f - f.mean()) / f.std()
    g = (g - g.mean()) / g.std()
    cor = (f * g).mean()
    print(cor)

    """
        This checks the correlation between the diagonals. This should equal to 1 which would tell us
        that the computed values are proportial.

        M=32 -> cor=0.98299956
        M=256 -> cor=0.99656504
        M=1024 -> cor=0.9992698

        This looks good. The values coincide up to a scalar factor. This would explain why
        the optimal lambda parameter is different for the two approaches in my experiments.
        The scale factor is about 6193706 for M=1024

    """