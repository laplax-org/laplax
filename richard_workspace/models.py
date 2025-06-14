import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import equinox as eqx

class nnxMLP(nnx.Module):
    def __init__(self, layer_sizes, rngs: nnx.Rngs):
        self.linears = []
        for i in range(len(layer_sizes) - 1):
            self.linears.append(nnx.Linear(layer_sizes[i], layer_sizes[i+1], rngs=rngs))

    def __call__(self, x):
        for linear in self.linears[:-1]:
            x = jax.nn.relu(linear(x))
        x = self.linears[-1](x)
        return x
    
class eqxMLP(eqx.Module):

    lin1 : eqx.nn.Linear
    lin2 : eqx.nn.Linear
    lin3 : eqx.nn.Linear

    def __init__(self, key: jr.PRNGKey):
        keys = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(28*28, 10, key=keys[0])
        self.lin2 = eqx.nn.Linear(10, 10, key=keys[1])
        self.lin3 = eqx.nn.Linear(10, 10, key=keys[2])

    def __call__(self, x):
        x = jax.nn.relu(self.lin1(x))
        x = jax.nn.relu(self.lin2(x))
        x = self.lin3(x)
        return x