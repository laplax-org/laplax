from richard_workspace.helpers import *
from functools import partial
from laplax.util.tree import sub, dot, ones_like, zeros_like
from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split

import optax
last_layer_diagonal = None

"""
    We want to check whether large curvature values lead to large gradient norms.
"""

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

    # only get the diagonal for the last layer.
    len_last_layer = jax.tree.leaves(params)[-1].size
    total_len = sum([arr.size for arr in jax.tree.leaves(params)])
    last_layer_trees = [zeros_like(params) for _ in range(len_last_layer)]
    last_layer_trees_aug = []
    
    jnp.arange(total_len - len_last_layer, total_len)
    for i, entry in enumerate(last_layer_trees):
        flat, treedef = jax.tree.flatten(entry)
        flat[-1] = flat[-1].at[i // 10, i % 10].set(1.0)
        unflattened = jax.tree.unflatten(treedef, flat)
        last_layer_trees_aug.append(unflattened)

    hessian_last_rows = [partial_hvp(s) for s in last_layer_trees_aug]
    diagonal_last_layer = [dot(a, b) for a, b in zip(hessian_last_rows, last_layer_trees_aug)]

    def inner(v):
      return dot(v, partial_hvp(v))
    
    dll = jnp.array(diagonal_last_layer)
    dll = (dll - dll.mean()) / dll.std()

    return inner, dll

def large_last_layer(*args, **kwargs):
    def inner(v):
        ones = ones_like(v)
        ones_flat, treedef = jax.tree.flatten(ones)
        ones_flat[-1]*=10000
        scaled = jax.tree.unflatten(treedef, ones_flat)

        return dot(scaled, v)
    return inner

@partial(nnx.jit, static_argnames=['inner_fn'])
def train_step(model, optimizer, x, y, mode, inner_fn, ell):

    global last_layer_diagonal
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
    g = jnp.abs(jax.tree.leaves(grads)[-1].reshape(-1))
    g = (g - g.mean()) / g.std()
    
    optimizer.update(grads)
    # Use Pearson correlation coefficient between standardized gradient and last_layer_diagonal
    cor = jnp.mean(g * last_layer_diagonal)
    return loss, cor

def train_model(model, trainloader, optimizer, seed, mode, inner_fn, num_epochs, ell):
  hist, corhist = [], []
  for epoch in range(num_epochs):
    for xb, yb in trainloader:
      loss, cor = train_step(model, optimizer, xb, yb, mode, inner_fn=inner_fn, ell=ell)
      hist.append(loss.item())
      corhist.append(cor.item())
  return hist, corhist

def main():
    
    global last_layer_diagonal
    _trainloader, _testloader, numsamples_train, numsamples_test = torch_load_mnist()
    model = MLP([28*28, 10, 10], rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))
    _, params = nnx.split(model)
    params_before = jax.tree.leaves(params).copy()


    inner_fn, last_layer_diagonal = ggn_inner(model, torch_permute_data(_trainloader, seed=0), batch_size=128, numsamples_train=numsamples_train)
    mode = zeros_like(params)
    
    results = []
    loss_histories = []
    
    trainloader = torch_permute_data(_trainloader, seed=0)
    hist, corhist = train_model(
        model=model, 
        trainloader=trainloader,
        optimizer=optimizer,
        mode=mode, inner_fn=inner_fn,
        num_epochs=10, ell=10.0, seed=0
    )
    c = jnp.array(corhist)
    print(c.mean())
    print((c > 0).sum() / len(c))

    _, params = nnx.split(model)
    params_after = jax.tree.leaves(params)
    return results, loss_histories

if __name__ == '__main__':
    main()

