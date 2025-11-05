from functools import partial

import jax
import jax.numpy as jnp
import optax

from laplax.curv.loss import create_loss_hessian_mv, fetch_loss_gradient_fn
from laplax.enums import LossFn

# ---------------------------------------------------------------
# Loss Gradients
# ---------------------------------------------------------------


def test_single_binary_cross_entropy_loss_gradient():
    key = jax.random.key(0)
    target = jnp.zeros(1)
    logits = jax.random.normal(key, (1,))

    # Set loss gradient via autodiff
    def BCE(f, y):
        return optax.sigmoid_binary_cross_entropy(f, y)[0]

    grad_autodiff = jax.grad(
        BCE,
    )(logits, target)

    # Set loss gradient via laplax
    grad_fn_laplax = fetch_loss_gradient_fn(
        LossFn.BINARY_CROSS_ENTROPY, None, vmap_over_data=False
    )
    grad_laplax = grad_fn_laplax(logits, target)
    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


def test_binary_cross_entropy_loss_gradient_vmap():
    key = jax.random.key(0)
    target = jnp.zeros(5)
    logits = jax.random.normal(key, (5,))

    # Set loss gradient via autodiff
    def BCE(f, y):
        return optax.sigmoid_binary_cross_entropy(f, y)

    grad_autodiff = jax.vmap(
        jax.grad(
            BCE,
        )
    )(logits, target)

    # Set loss gradient via laplax
    grad_fn_laplax = fetch_loss_gradient_fn(
        LossFn.BINARY_CROSS_ENTROPY, None, vmap_over_data=True
    )
    grad_laplax = grad_fn_laplax(logits, target)

    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


def test_cross_entropy_loss_gradient():
    key = jax.random.key(0)
    target = jnp.asarray([2], dtype=int)
    logits = jax.random.normal(key, (3))

    # Set loss gradient via autodiff
    def fn(f, y):
        return optax.softmax_cross_entropy_with_integer_labels(f[None, :], y)[0]

    grad_autodiff = jax.grad(
        fn,
    )(logits, target)  # (3)

    # Set loss gradient via laplax
    grad_fn_laplax = fetch_loss_gradient_fn("cross_entropy", None, vmap_over_data=False)
    grad_laplax = grad_fn_laplax(logits, target)

    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


def test_cross_entropy_loss_gradient_vmap():
    key = jax.random.key(0)
    target = jnp.zeros(5, dtype=int)
    target.at[3].set(2)
    logits = jax.random.normal(key, (5, 3))

    # Set loss gradient via autodiff
    grad_autodiff = jax.vmap(
        jax.grad(
            optax.softmax_cross_entropy_with_integer_labels,
        )
    )(logits, target)  # (5,3)

    # Set loss gradient via laplax
    grad_fn_laplax = fetch_loss_gradient_fn("cross_entropy", None, vmap_over_data=True)
    grad_laplax = grad_fn_laplax(logits, target)
    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


def test_mean_sqared_error_loss_gradient_vmap():
    key = jax.random.key(0)
    target = jnp.zeros((5, 3))
    values = jax.random.normal(key, (5, 3))

    # Set loss gradient via autodiff
    grad_autodiff = jax.vmap(
        jax.grad(
            lambda pred, target: jnp.sum((pred - target) ** 2),
        )
    )(values, target)  # (5,3)

    # Set loss gradient via laplax
    grad_fn_laplax = fetch_loss_gradient_fn(LossFn.MSE, None, vmap_over_data=True)
    grad_laplax = grad_fn_laplax(values, target)
    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


def test_callable_loss_gradient():
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)
    pred = jax.random.normal(keys[0], (10, 3))
    target = jax.random.normal(keys[1], (10, 3))

    # Set random loss function
    random_arr = jax.random.normal(keys[2], (3,))

    def loss_func(pred, target):
        return jnp.sum(random_arr @ (pred - target) ** 3)

    # Set loss hessian via autodiff
    grad_autodiff = jax.vmap(jax.grad(loss_func))(pred, target)

    # Set loss hessian via laplax mv
    grad_fn = fetch_loss_gradient_fn(loss_func, None, vmap_over_data=True)
    grad_laplax = grad_fn(pred, target)

    assert jnp.allclose(grad_autodiff, grad_laplax, atol=1e-8)


# ---------------------------------------------------------------
# Loss Hessians
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
