"""
    Get simple curvature out of laplax
"""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from typing import *
from jaxtyping import Array
from torchvision import datasets, transforms
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
  
def load_twomoons(size=200):
    X, y = make_moons(n_samples=size)
    X = jnp.array(X).reshape(size, -1)
    y = jax.nn.one_hot(jnp.array(y), num_classes=2)
    return train_test_split(X, y, test_size=0.3)

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    def dataset_to_arrays(dataset):
        data = [np.array(img).reshape(-1) for img, _ in dataset]
        targets = [target for _, target in dataset]
        data = jnp.array(np.stack(data))
        targets = jax.nn.one_hot(jnp.array(np.array(targets)), num_classes=10)
        return data, targets

    return (*dataset_to_arrays(train_dataset), *dataset_to_arrays(test_dataset))

def evaluate_model(model, X_test, y_test, seeds):
    per_seed_acc = []
    batch_size = 32
    for seed in seeds:
        X_test_perm, y_test_perm = permute_data(X_test, y_test, seed)
        y_preds_test = []
        num_samples = X_test_perm.shape[0]
        for i in range(0, num_samples, batch_size):
            x_batch = X_test_perm[i:i+batch_size]
            y_batch = y_test_perm[i:i+batch_size]
            y_preds_test.append(accuracy(model, x_batch, y_batch))
        y_preds_test = jnp.stack(y_preds_test)
        accuracy_value = jnp.mean(y_preds_test)
        per_seed_acc.append(accuracy_value)
    return per_seed_acc

def torch_evaluate_model(model, _testloader, seeds):
    per_seed_acc = []
    for seed in seeds:
        testloader = torch_permute_data(_testloader, seed)
        y_preds_test = []
        for x_batch, y_batch in testloader:
            y_preds_test.append(accuracy(model, x_batch, y_batch))
        y_preds_test = jnp.stack(y_preds_test)
        accuracy_value = jnp.mean(y_preds_test)
        per_seed_acc.append(accuracy_value)
    return per_seed_acc

def flat(leaves : List[Array]):
    return jnp.concat([arr.reshape(-1) for arr in leaves])

def accuracy(model, x, y):
    log_y_pred = model(x)
    y_pred = jnp.argmax(log_y_pred, axis=-1)
    y_true = jnp.argmax(y, axis=-1)
    return sum(y_pred == y_true).astype(int) / len(y)

def permute_data(X, y, seed):
    perm = np.random.RandomState(seed).permutation(X.shape[1]) \
      if seed != 0 else np.arange(X.shape[1])
    X_perm = np.array(X)
    X_perm = jnp.array(X_perm[:, perm])
    return X_perm, y

def torch_permute_data(loader, seed):
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

    return permute_data(loader)


def torch_load_mnist():
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=32), DataLoader(test_dataset, batch_size=32), len(train_dataset), len(test_dataset)

class MLP(nnx.Module):
    def __init__(self, layer_sizes, rngs: nnx.Rngs):
        self.linears = []
        for i in range(len(layer_sizes) - 1):
            self.linears.append(nnx.Linear(layer_sizes[i], layer_sizes[i+1], rngs=rngs))

    def __call__(self, x):
        for linear in self.linears[:-1]:
            x = jax.nn.relu(linear(x))
        x = self.linears[-1](x)
        return x

def collect_numsamples_from_loader(loader, maxsamples):
      x, y = [], []
      cur = 0
      for xx, yy in loader:
        x.append(xx); y.append(yy)
        cur += len(xx)
        if cur >= maxsamples:
           break
        
      xret, yret = jnp.concat(x, axis=0), jnp.concat(y, axis=0)
      return xret[:maxsamples], yret[:maxsamples]
