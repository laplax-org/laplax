"""Fisher Matrix Vector Product."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.random import bernoulli, categorical

from laplax.curv.loss import fetch_loss_gradient_fn
from laplax.enums import LossFn
from laplax.types import (
    Data,
    Float,
    Int,
    KeyType,
    ModelFn,
    Params,
)
from laplax.util.tree import mean, mul


def _fisher_calculation(f_ns, jvp, ys, loss_grad_fn, vec):
    r"""Performs concrete Fisher calculation.

    Calculates
    $$
    \text{jvp}^\top (\text{grad}(\text{grad}^\top(\text{jvp}(\text{vec}))))
    $$
    where 'grad' is the gradient of the loss function evaluated at 'f_n' and 'y'.

    Args:
        f_ns: Output of the model's forward pass of shape (o,) or (n,o)
        jvp: Jacobian mvp of the models forward pass
        ys: The labels to use of shape (o,), (m,o), (n,o) or (n,m,o)
        loss_grad_fn: The gradient of the loss function
        vec: A PyTree that can be consumed by jvp

    Returns:
        The unscaled Fisher matrix vector poduct
    """

    def _jvp(vec):
        return jnp.atleast_2d(jvp(vec))

    vjp = jax.linear_transpose(_jvp, vec)

    # Calculate grads and expand as necessary to shape (n,m,o) where
    # n: Number of datapoints
    # m: Number of mc_samples
    # o: Output dimension of network (Length of gradient vector)

    if ys.ndim == 1:
        # no batch or mc_samples dim
        grads = jnp.expand_dims(loss_grad_fn(f_ns, ys), axis=(0, 1))

    elif ys.ndim == 3:
        # batch and mc_sample dim
        f_ns_expanded = jnp.expand_dims(f_ns, 1)
        grads = loss_grad_fn(f_ns_expanded, ys)

    elif f_ns.ndim == 1:
        # mc_samples dim, but no batch dim
        f_ns_expanded = jnp.expand_dims(f_ns, 0)
        grads = jnp.expand_dims(loss_grad_fn(f_ns_expanded, ys), axis=0)
    else:
        # batch dim, but no mc_samples dim
        grads = jnp.expand_dims(loss_grad_fn(f_ns, ys), 1)
    
    Jv = _jvp(vec)
    GtJv = jnp.einsum("nmo,no->nm", grads, Jv)
    GGtJv = jnp.einsum("nm,nmo->no", GtJv, grads)
    JtGGtJv = vjp(GGtJv)[0]
    return JtGGtJv


def create_empirical_fisher_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str | Callable | None,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    loss_grad_fn: Callable | None = None,
) -> Callable[[Params, Data], Params]:
    r"""Create empirical Fisher matrix-vector product without fixed data.

    The resulting matrix vector product computes:
    $$
    \text{factor} \cdot \sum_n J_n^\top \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    where $J_n$ is the Jacobian of the model w.r.t the parameters
    evaluated at data point $n$, $c(y=y_n,\hat{y}=f_n)$ is the
    loss function evaluated at data point $n$, and $v$ is the vector.
    The `factor` is a scaling factor that is used to scale the Fisher matrix.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the Fisher computation.
        factor: Scaling factor for the Fisher computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_grad_fn: The loss gradient function.

    Returns:
        A function that takes a vector and a batch of data,
        and computes the empirical Fisher matrix-vector product.

    Note:
        If 'vmap_over_data'=True (default), the computation is vmapped over the batch
        of data. If the computation should be performed for a singe datum
        (or a singleton batch dimension), pass 'vmap_over_data'=False.
        If 'vmap_over_data'=False and a non-singleton batch dimension is present,
        the batch dimension is handled by vectorization.
        Compared to 'vmap_over_data'=True, this can be faster especially for large batches,
        but the passed 'model_fn' and (optional) 'loss_grad_fn'
        must accept batches of data.
    """

    def empirical_fisher_mv(vec, data):
        grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn)
        if vmap_over_data:
            if not data["input"].ndim > 1:
                msg = "Could not find a leading batch dimension. If this is intentional, pass vmap_over_data=False"
                raise ValueError(msg)

            def fisher_calculation_for_vmap(datum):
                x, y = datum["input"], datum["target"]
                # Calculate forward pass and its derivative
                f_n, jvp = jax.linearize(lambda p: model_fn(x, p), params)
                return _fisher_calculation(f_n, jvp, y, grad_fn, vec)

            vmap = jax.vmap(fisher_calculation_for_vmap)(data)
            fisher = mean(vmap, axis=0)  # over batch dimension
            batch_size = data["input"].shape[0]
            return mul(factor * batch_size, fisher)

        if data["input"].ndim == 1:
            # No leading batch dim => Calculate for single datum
            xs, ys = data["input"], data["target"]

        elif data["input"].shape[0] == 1:
            # Only one datum in batch
            xs, ys = data["input"][0], data["target"][0]

        else:
            # Handle batch dimension of data explicitly
            xs, ys = data["input"], data["target"]
            grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn, handle_batches=True)

        f_ns, jvp = jax.linearize(lambda p: model_fn(xs, p), params)

        fisher = _fisher_calculation(f_ns, jvp, ys, grad_fn, vec)
        return mul(factor, fisher)

    return empirical_fisher_mv


def create_empirical_fisher_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str | Callable | None = None,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    loss_grad_fn: Callable | None = None,
) -> Callable[[Params], Params]:
    r"""Creates the empirical Fisher matrix-vector product with data.

    The resulting matrix vector product computes:
    $$
    \frac{\text{num_total_samples}}{\text{num_curv_samples}} 
    \cdot \sum_n J_n^\top \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    where $J_n$ is the Jacobian of the model w.r.t the parameters
    evaluated at data point $n$, $c(y=y_n,\hat{y}=f_n)$ is the
    loss function evaluated at data point $n$, and $v$ is the vector.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset
    remains fixed.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the Fisher computation.
        num_curv_samples: Number of samples used to calculate the Fisher.
            Defaults to None, in which case it is inferred from `data`
            as its batch size.
            Note that for losses that contain sums even for a single input
            (e.g., pixel-wise semantic segmentation losses),
            this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_curv_samples`'s description. Defaults to None, in which case
            it is set to equal `num_curv_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_grad_fn: The loss gradient function.
            If not provided, it is computed using the 'loss_fn'.

    Returns:
        A function that takes a vector and computes
        the empirical Fisher matrix-vector product for the given data.

    Note:
        If 'vmap_over_data'=True (default), the computation is vmapped over the batch
        of data. If the computation should be performed for a singe datum
        (or a singleton batch dimension), pass 'vmap_over_data'=False.
        If 'vmap_over_data'=False and a non-singleton batch dimension is present,
        the batch dimension is handled by vectorization.
        Compared to 'vmap_over_data'=True, this can be faster especially for large batches,
        but the passed 'model_fn' and (optional) 'loss_grad_fn'
        must accept batches of data.
    """
    if num_curv_samples is None:
        # infer num_curv_samples from batch dimension
        if data["input"].ndim == 1:
            # no batch dimension present
            num_curv_samples = 1
        else:
            num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    fisher_mv = create_empirical_fisher_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        vmap_over_data=vmap_over_data,
        loss_grad_fn=loss_grad_fn,
    )

    def wrapped_fisher_mv(vec: Params) -> Params:
        return fisher_mv(vec, data)

    return wrapped_fisher_mv


def sample_likelihood(loss_fn, f_ns, mc_samples, key):
    # sample mc_samples values $\tilde{y} from e^{-\text{loss_fn}(y, f_n)}$
    *n, o = f_ns.shape
    if loss_fn == LossFn.MSE:
        unit_samples = jax.random.normal(key, shape=(*n, mc_samples, o))
        return unit_samples * jnp.sqrt(0.5) + f_ns[..., None, :]

    if loss_fn == LossFn.CROSS_ENTROPY:
        f_ns = jnp.expand_dims(f_ns, axis=-2)
        return categorical(key, f_ns, shape=(*n, mc_samples), replace=True)[..., None]

    if loss_fn == LossFn.BINARY_CROSS_ENTROPY:
        f_ns = jnp.expand_dims(f_ns, axis=-2)
        bool_samples = bernoulli(key, f_ns, shape=(*n, mc_samples, 1))
        return jnp.astype(bool_samples, jnp.float32)

    msg = f"Unsupported LossFn {loss_fn} to sample from."
    raise ValueError(msg)


def create_MC_fisher_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | str,
    factor: Float,
    *,
    vmap_over_data: bool = True,
    mc_samples: Int | None = 1,
) -> Callable[[Params, Data], Params]:
    r"""Create Monte-Carlo approximated Fisher matrix-vector product without fixed data.

    The resulting matrix vector product computes:
    $$
    \text{factor} \cdot \frac{1}{\text{mc_samples}\sum_n,m J_n^\top \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    where $J_n$ is the Jacobian of the model w.r.t the parameters evaluated at data point $n$,
    $c(y,\hat{y})$ is the loss function, and $v$ is the vector.
    $\tilde{y}_{n,m}$ is the m-th Monte Carlo sample of the label under the likelihood
    induced by the loss function: $r(y|f_n) = \exp(-c(y,\hat{y}=f_n))$ at data point $n$.
    The `factor` is a scaling factor that is used to scale the Fisher matrix.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the Fisher computation.
        factor: Scaling factor for the Fisher computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        mc_samples: Number of MC samples to use. Defaults to 1.

    Returns:
        A function that takes a vector, a batch of data and a key,
        and computes the Monte-Carlo Fisher matrix-vector product.

    Note:
        If vmap_over_data is true (default), the computation is vmapped over the batch
        of data. If the computation should be performed for a singe datum
        (or a singleton batch dimension), pass 'vmap_over_data'=False.
        If 'vmap_over_data'=False and a non-singleton batch dimension is passed,
        the batch dimension is handled explicitly.
        In this case, the passed model_fn must accept batches of data.
    """

    def mc_fisher_mv(vec, data, key):
        grad_fn = fetch_loss_gradient_fn(loss_fn, None, handle_batches=True)
        if vmap_over_data:
            if not data["input"].ndim > 1:
                msg = "Could not find a leading batch dimension. If this is intentional, pass vmap_over_data=False"
                raise ValueError(msg)

            batch_size = data["input"].shape[0]
            keys = jax.random.split(key, batch_size)

            def fisher_calculation_for_vmap(datum, key):
                x = datum["input"]
                f_n, jvp = jax.linearize(lambda p: model_fn(x, p), params)
                y_samples = sample_likelihood(loss_fn, f_n, mc_samples, key)
                return _fisher_calculation(f_n, jvp, y_samples, grad_fn, vec)

            vmap = jax.vmap(fisher_calculation_for_vmap)(data, keys)
            fisher = mean(vmap, axis=0)  # over batch dimension
            return mul(factor * batch_size / mc_samples, fisher)

        if data["input"].ndim == 1:
            # No leading batch dim => Calculate for single datum
            xs = data["input"]

        elif data["input"].shape[0] == 1:
            # Only one datum in batch
            xs = data["input"][0]
        else:
            xs = data["input"]

        f_ns, jvp = jax.linearize(lambda p: model_fn(xs, p), params)
        y_samples = sample_likelihood(loss_fn, f_ns, mc_samples, key)

        fisher = _fisher_calculation(f_ns, jvp, y_samples, grad_fn, vec)
        return mul(factor / mc_samples, fisher)

    return mc_fisher_mv


def create_MC_fisher_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    mc_samples: Int | None = 1,
) -> Callable[[Params], Params]:
    r"""Create Monte-Carlo approximated Fisher matrix-vector product.

    The resulting matrix vector product computes:
    $$
    \text{factor} \cdot \frac{1}{\text{mc_samples}\sum_n,m J_n^\top \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    where $J_n$ is the Jacobian of the model w.r.t the parameters
    evaluated at data point $n$, $c(y,\hat{y})$ is the
    loss function, and $v$ is the vector.
    $\tilde{y}_{n,m}$ is the m-th Monte Carlo sample of the label under the likelihood
    induced by the loss function: $r(y|f_n) = \exp(-c(y,\hat{y}=f_n))$ at data point $n$.
    The `factor` is a scaling factor that is used to scale the Fisher matrix.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset remains fixed.
    
    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the Fisher computation.
        num_curv_samples: Number of samples used to calculate the Fisher.
            Defaults to None, in which case it is inferred from `data`
            as its batch size.
            Note that for losses that contain sums even for a single input
            (e.g., pixel-wise semantic segmentation losses),
            this number is _not_ the batch size.
        num_total_samples: Number of total samples the model was trained on. See the
            remark in `num_curv_samples`'s description. Defaults to None, in which case
            it is set to equal `num_curv_samples`.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        mc_samples: Number of MC samples to use. Defaults to 1.

    Returns:
        A function that takes a vector and key and computes
        the Monte-Carlo Fisher matrix-vector product.

    Note:
        If vmap_over_data is true (default), the computation is vmapped over the batch
        of data. If the computation should be performed for a singe datum
        (or a singleton batch dimension), pass 'vmap_over_data'=False.
        If 'vmap_over_data'=False and a non-singleton batch dimension is passed,
        the batch dimension is handled explicitly.
        In this case, the passed model_fn must accept batches of data.
    """
    if num_curv_samples is None:
        # infer num_curv_samples from batch dimension
        if data["input"].ndim == 1:
            # no batch dimension present
            num_curv_samples = 1
        else:
            num_curv_samples = data["input"].shape[0]

    if num_total_samples is None:
        num_total_samples = num_curv_samples

    curv_scaling_factor = num_total_samples / num_curv_samples

    fisher_mv = create_MC_fisher_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=curv_scaling_factor,
        vmap_over_data=vmap_over_data,
        mc_samples=mc_samples,
    )

    def wrapped_fisher_mv(vec, key):
        return fisher_mv(vec, data, key)

    return wrapped_fisher_mv
