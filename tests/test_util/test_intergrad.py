import equinox as eqx
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

from laplax.util.intergrad import intergrad


class NnxMLP(nnx.Module):
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


class EqxMLP(eqx.Module):
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


class LinenMLP(nn.Module):
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


class PerturbMLP(nnx.Module):
    def __init__(self, in_dim, mid_dim, out_dim, key):
        self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
        self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.perturb("xgrad1", x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = self.perturb("xgrad2", x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        x = self.perturb("xgrad3", x)

        return x


class IntermediateMLP(nnx.Module):
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


# ------------------- TESTS -------------------


def test_eqx_intergrad_shapes():

    key = jax.random.PRNGKey(0)
    model = EqxMLP(5, 12, 10, key)
    params, static = eqx.partition(model, eqx.is_array)
    x = jax.random.normal(key, (10, 5))
    y = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), num_classes=10)

    def celoss(params, x, y):
        model_ = eqx.combine(params, static)
        logits = model_(x)
        return -(y * jax.nn.log_softmax(logits)).mean()

    a, g = jax.vmap(intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0))(
        params, x, y
    )
    assert len(a) == 2, "Expected 2 activations"
    assert len(g) == 3, "Expected 3 gradients"
    assert jax.tree.map(lambda x: x.shape, a) == [(10, 12), (10, 12)]
    assert jax.tree.map(lambda x: x.shape, g) == [(10, 12), (10, 12), (10, 10)]


def test_linen_intergrad_shapes():
    key = jax.random.PRNGKey(0)
    model = LinenMLP(5, 12, 10)
    x = jax.random.normal(key, (10, 5))
    y = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), num_classes=10)

    variables = model.init(key, x)
    params = variables["params"]

    def celoss(params, x, y):
        logits = model.apply({"params": params}, x)
        return -(y * jax.nn.log_softmax(logits)).mean()

    a, g = jax.vmap(intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0))(
        params, x, y
    )
    assert len(a) == 2, "Expected 2 activations"
    assert len(g) == 3, "Expected 3 gradients"
    assert jax.tree.map(lambda x: x.shape, a) == [(10, 12), (10, 12)]
    assert jax.tree.map(lambda x: x.shape, g) == [(10, 12), (10, 12), (10, 10)]


def test_nnx_activations():
    model = NnxMLP(5, 12, 10, nnx.Rngs(0))
    dummy_model = IntermediateMLP(5, 12, 10, nnx.Rngs(0))
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key=key, shape=(10, 5))
    y = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), num_classes=10)

    graph, params = nnx.split(model)

    def model_fn(p, x):
        model_ = nnx.merge(graph, p)
        return model_(x)

    def celoss(params, x, y):
        logits = model_fn(params, x)
        return -(y * jax.nn.log_softmax(logits)).mean()

    activations, _ = jax.vmap(
        intergrad(celoss, tagging_rule=None), in_axes=(None, 0, 0)
    )(params, x, y)
    _, dummy_acts = dummy_model(x)
    assert len(activations) == len(dummy_acts)
    assert all(list(map(jnp.allclose, activations, dummy_acts)))


def test_nnx_gradients():
    model = NnxMLP(5, 12, 10, nnx.Rngs(0))
    pmodel = PerturbMLP(5, 12, 10, nnx.Rngs(0))
    x, y = jnp.ones(5), jax.nn.one_hot(1, num_classes=10)

    # make sample run
    _ = pmodel(x)
    _ = model(x)

    graph, params = nnx.split(model)

    def model_fn(p, x):
        model_ = nnx.merge(graph, p)
        return model_(x)

    @nnx.grad(
        argnums=nnx.DiffState(argnum=0, filter=nnx.Any(nnx.Param, nnx.Perturbation))
    )
    def p_celoss(model, x, y):
        preds = jax.nn.log_softmax(model(x))
        return -(preds * y).mean()

    def celoss(params, x, y):
        logits = model_fn(params, x)
        loss = -(y * jax.nn.log_softmax(logits)).mean()
        return loss

    allgrads = p_celoss(pmodel, x, y)
    intm_grads = [allgrads.xgrad1.value, allgrads.xgrad2.value, allgrads.xgrad3.value]
    _, grads = intergrad(celoss, tagging_rule=None)(params, x, y)
    assert all([jnp.allclose(a, b).all() for a, b in    # noqa: C419
                zip(intm_grads, grads, strict=False)])
