from flax import nnx
import jax
import types

import jax.numpy as jnp

class RecordActivations(nnx.Module):
    def __init__(self, mod : nnx.Module):
        self.mod = mod
        self.activations = nnx.Variable([])

    def __call__(self, x):
        out = self.mod(x)
        self.activations.value.append(out)
        return out
    
class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dmid, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear(x))  # <- we want to log what's inside this
        return self.linear_out(x)

def celoss(model, x, y):
    log_ypred = jax.nn.log_softmax(model(x))
    return -(y * log_ypred).mean()

def overwrite_linears():
    """
        This would be possible for nnx, but impossible for equinox, as in eqx
        models are frozen dataclasses and we cannot overwrite a frozen field...
    """
    model = MLP(10, 20, 10, rngs=nnx.Rngs(0))
    model.linear, model.linear_out = RecordActivations(model.linear), RecordActivations(model.linear_out)
    x, y = jax.numpy.arange(10), jax.nn.one_hot(1, 10)
    loss, grads = nnx.value_and_grad(celoss)(model, x, y)
    print(model.linear.activations)
    print(grads)

if __name__=='__main__':
    overwrite_linears()