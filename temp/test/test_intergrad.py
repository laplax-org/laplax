from temp.intergrad import intergrad

from flax import nnx

import unittest
import jax
import jax.numpy as jnp


# --- NNX MLP ---
class nnx_mlp(nnx.Module):
    def __init__(self, in_dim, mid_dim, out_dim, key):
        self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
        self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x

# --- Equinox MLP ---
import equinox as eqx
import jax.nn as jnn

class eqx_mlp(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self, in_dim, mid_dim, out_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_dim, mid_dim, key=k1)
        self.linear2 = eqx.nn.Linear(mid_dim, mid_dim, key=k2)
        self.linear3 = eqx.nn.Linear(mid_dim, out_dim, key=k3)

    def __call__(self, x):
        x = self.linear1(x)
        x = jnn.relu(x)
        x = self.linear2(x)
        x = jnn.relu(x)
        x = self.linear3(x)
        return x

# --- Flax Linen MLP ---
import flax.linen as nn

class linen_mlp(nn.Module):
    in_dim: int
    mid_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mid_dim)(x)
        x = jnn.relu(x)
        x = nn.Dense(self.mid_dim)(x)
        x = jnn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x

class p_nnx_mlp(nnx.Module):

  def __init__(self, in_dim, mid_dim, out_dim, key):
    self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
    self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
    self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

  def __call__(self, x):

    x = self.linear1(x)
    x = self.perturb('xgrad1', x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    x = self.perturb('xgrad2', x)
    x = jax.nn.relu(x)
    x = self.linear3(x)
    x = self.perturb('xgrad3', x)

    return x

class nnx_intermediate_mlp(nnx.Module):
    """MLp which returns intermediate activations"""
    def __init__(self, in_dim, mid_dim, out_dim, key):
        self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
        self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

    def __call__(self, x):
        x = self.linear1(x)
        act1 = jax.nn.relu(x)
        x = self.linear2(act1)
        act2 = jax.nn.relu(x)
        x = self.linear3(act2)
        return x, [act1, act2]
   
class TestIntergradNNX(unittest.TestCase):

    def test_eqx_intergrad_shapes(self):
        key = jax.random.PRNGKey(0)
        model = eqx_mlp(5, 12, 10, key)
        params, static = eqx.partition(model, eqx.is_array)
        x = jax.random.normal(key, (10, 5))
        y = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), num_classes=10)

        def celoss(params, x, y):
            model = eqx.combine(params, static)
            logits = model(x)
            return -(y * jax.nn.log_softmax(logits)).mean()


        a, g = jax.vmap(intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0))(params, x, y)
        
        assert len(a) == 2 and len(g) == 3, "Expected 2 activations and 3 gradients"
        assert jax.tree.map(lambda x: x.shape, a) == [(10, 12), (10, 12)] # activation shapes
        assert jax.tree.map(lambda x: x.shape, g) == [(10, 12), (10, 12), (10, 10)] # gradient shapes


    def test_linen_intergrad_shapes(self):
        key = jax.random.PRNGKey(0)
        model = linen_mlp(5, 12, 10)
        x = jax.random.normal(key, (10, 5))
        y = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), num_classes=10)

        variables = model.init(key, x)
        params = variables['params']

        def celoss(params, x, y):
            logits = model.apply({'params': params}, x)
            return -(y * jax.nn.log_softmax(logits)).mean()

        a, g = jax.vmap(intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0))(params, x, y)
        
        assert len(a) == 2 and len(g) == 3, "Expected 2 activations and 3 gradients"
        assert jax.tree.map(lambda x: x.shape, a) == [(10, 12), (10, 12)] # activation shapes
        assert jax.tree.map(lambda x: x.shape, g) == [(10, 12), (10, 12), (10, 10)] # gradient shapes


    def test_nnx_activations(self):
        
        model = nnx_mlp(5, 12, 10, nnx.Rngs(0))
        dummy_model = nnx_intermediate_mlp(5, 12, 10, nnx.Rngs(0))
        
        key = jax.random.PRNGKey(0)
        x, y = jax.random.normal(key=key, shape=(10, 5)), jax.nn.one_hot(jnp.ones(10), num_classes=10)

        graph, params = nnx.split(model)
        def model_fn(p, x):
            model = nnx.merge(graph, p)
            return model(x)
        
        def celoss(params, x, y):
            logits = model_fn(params, x)
            return -(y * jax.nn.log_softmax(logits)).mean()
        
        activations, grads = jax.vmap(intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0))(params, x, y)

        # collect activations for dummy model
        _, dummy_acts = dummy_model(x)

        # check that shapes match
        self.assertEqual(len(activations), len(dummy_acts))
        assert all([jnp.allclose(a, b) for a, b in zip(activations, dummy_acts)])



    def test_nnx_gradients(self):
        model = nnx_mlp(5, 12, 10, nnx.Rngs(0))
        pmodel = p_nnx_mlp(5, 12, 10, nnx.Rngs(0))
        x, y = jnp.ones(5), jax.nn.one_hot(1, num_classes=10)

        # make sample run
        _ = pmodel(x)
        _ = model(x)

        graph, params = nnx.split(model)
        def model_fn(p, x):
            model = nnx.merge(graph, p)
            return model(x)
        
        @nnx.grad(argnums=nnx.DiffState(argnum=0, filter=nnx.Any(nnx.Param, nnx.Perturbation)))
        def p_celoss(model, x, y):
            preds = jax.nn.log_softmax(model(x))
            return -(preds * y).mean()

        def value_p_celoss(model, x, y):
            preds = jax.nn.log_softmax(model(x))
            return -(preds * y).mean()
        
        def celoss(params, x, y):
            logits = model_fn(params, x)
            loss = -(y * jax.nn.log_softmax(logits)).mean()
            return loss
        
        allgrads = p_celoss(pmodel, x, y)
        intm_grads = [allgrads.xgrad1.value, allgrads.xgrad2.value, allgrads.xgrad3.value]
        activations, grads = intergrad(celoss, tagging_rule=None)(params, x, y)

        assert all([jnp.allclose(a, b).all() for a, b in zip(intm_grads, grads)])

if __name__ == "__main__":
    unittest.main()
