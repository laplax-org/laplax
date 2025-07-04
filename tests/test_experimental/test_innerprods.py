import jax
import jax.numpy as jnp
from flax import nnx

from laplax.curv.ggn import create_ggn_mv
from laplax.experimental.innerprods import (
    ggn_inner,
    unscaled_dot_product,
    zero,
)
from laplax.util.mv import to_dense


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


def f(x):
    return jnp.sum(x**2)


def test_zero_inner_grad():
    inner = zero()

    def reg_f(x):
        return f(x) + inner(x)

    x = jnp.array([1.0, 2.0, 3.0])
    g = jax.grad(reg_f)(x)
    assert jnp.allclose(g, 2 * x), "Zero inner product should not affect gradient"


def test_unscaled_dot_product_grad():
    inner = unscaled_dot_product()

    def reg_f(x):
        return f(x) + inner(x)

    x = jnp.array([1.0, 2.0, 3.0])
    g = jax.grad(reg_f)(x)
    assert jnp.allclose(g, 4 * x), "Unscaled dot product should scale the gradient by 2"


def test_ggn_inner():
    params = jnp.ones((3,))

    def model_fn(input, params):
        return jnp.multiply(input, params)

    data = {"input": jnp.array([[1.0, 2.0, 3.0]]), "target": jnp.array([[1, 0, 0]])}

    ggn_inner_fn = ggn_inner(
        params=params, model_fn=model_fn, data=data, num_total_samples=1
    )
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn="cross_entropy",
        num_total_samples=1,
    )
    ggn_diag = to_dense(ggn_mv, layout=3)

    def vgv(x):
        return jnp.dot(x, ggn_diag @ x)

    xs = jax.random.normal(key=jax.random.key(0), shape=(1_000, 3))
    a, b = jax.vmap(vgv)(xs), jax.vmap(ggn_inner_fn)(xs)
    assert jnp.allclose(a, b, atol=1e-5)
