from functools import partial

import jax
import jax.numpy as jnp
import optax

from laplax.curv.loss import create_loss_hessian_mv
from laplax.enums import LossFn

# ---------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------


def test_binary_cross_entropy_loss_hessian():
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (1,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        optax.sigmoid_binary_cross_entropy,
    )(logits, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv("binary_cross_entropy")
    hess_laplax = jax.vmap(partial(hess_mv, pred=logits))(jnp.eye(1))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_cross_entropy_loss_hessian():
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        optax.softmax_cross_entropy_with_integer_labels,
    )(logits, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv("cross_entropy")
    hess_laplax = jax.vmap(partial(hess_mv, pred=logits))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_mse_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        lambda pred, target: jnp.sum((pred - target) ** 2),
    )(pred, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv(LossFn.MSE)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_callable_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set random loss function
    random_arr = jax.random.normal(keys[2], (10,))

    def loss_func(pred, target):
        return jnp.sum(random_arr @ (pred - target) ** 3)

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(loss_func)(pred, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv(loss_func)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred, target=target))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)
