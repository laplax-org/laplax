"""
    Get simple curvature out of laplax
"""
import jax
import optax
import jax.numpy as jnp
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from copy import deepcopy

from typing import List
from jaxtyping import PyTree, Array
from flax import nnx
from functools import partial
from laplax.curv import create_ggn_mv
from laplax.util.loader import input_target_split
from laplax.util.tree import sub, dot, zeros_like

class MLP(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.linear(x))
    return self.linear_out(x)

def flat(leaves : List[Array]):
    return jnp.concat([arr.reshape(-1) for arr in leaves])

@partial(nnx.jit, static_argnames=['partial_hvp'])
def train_step(model, optimizer, x, y, mode, partial_hvp, ell=0):

  def scaled_distance_to_prev_mode(model):
    if not mode:
      return 0

    _, model_state = nnx.split(model)
    # model_leaves = jax.tree.leaves(model_state)
    # mode_leaves = jax.tree.leaves(mode)
    
    # model_flat = jnp.concat([leaf.ravel() for leaf in model_leaves])
    # mode_flat = jnp.concat([leaf.ravel() for leaf in mode_leaves])
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

def permute_loaders(train_loader, test_loader, seed):
    """
    Permute the MNIST dataset loaders.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(28 * 28) if seed != 0 else np.arange(28 * 28)

    def permute_data(data_loader):
        for data, target in data_loader:
            data = data.view(data.size(0), -1)  # flatten
            data = jnp.array(data.numpy())
            data = data[:, perm]
            target = jax.nn.one_hot(jnp.array(target.numpy()), num_classes=10)
            yield data, target

    return permute_data(train_loader), permute_data(test_loader)

def evaluate_model(model, train_loader, test_loader, seeds):
    per_seed_acc = []
    for seed in seeds:
        # seed 0 corresponds to the original MNIST
        _, test = permute_loaders(train_loader, test_loader, seed)
        
        y_preds_test = []
        for j, (data, target) in enumerate(test):
            y_preds_test.append(accuracy(model, data, target))
        y_preds_test = jnp.stack(y_preds_test)
        accuracy_value = jnp.mean(y_preds_test)
        per_seed_acc.append(accuracy_value)
    return per_seed_acc

def accuracy(model, x, y):
    log_y_pred = model(x)
    y_pred = jnp.argmax(log_y_pred, axis=-1)
    y_true = jnp.argmax(y, axis=-1)
    return sum(y_pred == y_true).astype(int) / len(y)

def getmode(model):
    graph, state = nnx.split(model)
    return state

def train_model(model, optimizer, train_loader, test_loader, seed, 
                num_epochs, mode, partial_hvp, ell=0.001, batch_size=32):
    
    np.random.seed(seed)
    cur_train_loader, _ = permute_loaders(train_loader=train_loader, test_loader=test_loader, seed=seed)
    loss_history = []

    for epoch in range(num_epochs):
        for data, target in cur_train_loader:
            loss = train_step(model=model, 
                              optimizer=optimizer, 
                              x=data, 
                              y=target, 
                              ell=ell, mode=mode, partial_hvp=partial_hvp)
            
            loss_history.append(loss)

    return loss_history, getmode(model)

def identity(*args, **kwargs):
    return lambda x: x

def load_twomoons(size=200):
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=size)
    X, y = jnp.array(X), jax.nn.one_hot(jnp.array(y), num_classes=2)
    Xtrain, Xtest ,ytrain, ytest = train_test_split(X, y, test_size=0.3)

    model = MLP(din=2, dmid=10, dout=2, rngs=nnx.Rngs(0))
    return model, Xtrain, Xtest, ytrain, ytest

def load_mnist(batch_size=128, num_workers=4):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./laplace/data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./laplace/data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def main():
    
    train_loader, test_loader = load_mnist()
    model = MLP(din = 28 * 28, dmid = 10, dout=10, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3, weight_decay=5e-4))

    graphdef, state = nnx.split(model)
    mode = None #zeros_like(state)
    partial_hvp = None #identity()
    seeds = np.arange(5)
    GGN_DATA_SPLIT = 32
    loss_histories = []

    for i, seed in enumerate(seeds):
      loss_history, _ = train_model(model, optimizer, train_loader, test_loader,
                                      seed=seed, num_epochs=100, mode=mode, partial_hvp=partial_hvp)
      loss_histories.append(loss_history)

      # evaluate
      eval_result = \
        evaluate_model(model, 
                       train_loader=train_loader, 
                       test_loader=test_loader, 
                       seeds=seeds[:i+1])
      
      print(f"ROUND : {i}, " + str([f"Dataset : {j}, Acc: {acc}" for j, acc in enumerate(eval_result)]))
      
    #   graph_def, params = nnx.split(model)
    #   def model_fn(input, params):
    #       return nnx.call((graph_def, params))(input)[0]
    
    #   dummy_train, _ = permute_loaders(train_loader, test_loader, seed)
    #   xbatch, ybatch = next(iter(dummy_train))
    #   data = input_target_split((xbatch, ybatch)) # sample batch

    #   partial_hvp = create_ggn_mv(
    #       model_fn=model_fn,
    #       params=params,
    #       data=data,
    #       loss_fn='cross_entropy'
    #   )

if __name__ == '__main__':
   main()