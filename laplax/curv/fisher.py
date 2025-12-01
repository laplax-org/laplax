"""Fisher Matrix Vector Product."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

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


def _fisher_calculation(model_fn, params, xs, ys, loss_grad_fn, vec):
    r"""Performs concrete Fisher calculation.

    Calculates
    $$
    \text{jvp}^\top (\text{grad}(\text{grad}^\top(\text{jvp}(\text{vec}))))
    $$
    where 'grad' is the gradient of the loss function evalutated at 'f_n' and 'y'.
    
    Args:
        model_fn: models forward function
        params: model parameters
        xs: model input
        ys: The labels to use 
        loss_grad_fn: The gradient of the loss function
        vec: A PyTree that can be consumed by jvp

    Returns:
        The unscaled Fisher matrix vector poduct
    """
    # Calculate forward pass and its derivative
    f_ns, _jvp = jax.linearize(lambda p: model_fn(xs, p), params)
    jvp = lambda vec: jnp.atleast_2d(_jvp(vec))
    vjp = jax.linear_transpose(jvp, vec)

    grads = jnp.atleast_2d(loss_grad_fn(f_ns, ys))

    Jv = jvp(vec)
    GtJv = jnp.einsum("no,no->n", grads, Jv)
    GGtJv = jnp.einsum("n,no->no", GtJv, grads)
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
    \text{factor} \frac{1}{N}\cdot \sum_n J_n^\top \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    #where $J_n$ is the Jacobian of the model w.r.t the parameters
    #evaluated at data point $n$, $c(y=y_n,\hat{y}=f_n)$ is the
    #loss function evaluated at data point $n$, and $v$ is the vector.
    #The `factor` is a scaling factor that
    #is used to scale the Fisher matrix.

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
        If vmap_over_data is true (default), the computation is vmapped over the batch
        of data. If the computation should be performed for a singe datum 
        (or a singleton batch dimension), pass 'vmap_over_data'=False.
        If 'vmap_over_data'=False and a non-singleton batch dimension is passed, 
        the batch dimension is handled explicitly. 
        In this case, the passed model_fn and (optional) loss_grad_fn 
        must accept batches of data.

    """
    def empirical_fisher_mv(vec, data):
        fisher_calculation_partial = partial(_fisher_calculation, 
        model_fn=model_fn, 
        params=params,
        vec=vec)

        if vmap_over_data:
            msg = "vmap_over_data=True could not find a leading batch dimension"
            assert data["input"].ndim > 1, msg
            grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn)
            def fisher_calculation_for_vmap(datum):
                xs, ys = datum["input"], datum["target"]
                return fisher_calculation_partial(xs=xs, ys=ys, loss_grad_fn=grad_fn)

            vmap = jax.vmap(fisher_calculation_for_vmap)(data)
            fisher = mean(vmap, axis=0)  # over batch dimension
        else:
            if data["input"].ndim == 1:
                # No leading batch dim => Calculate for single datum
                x, y = data["input"], data["target"]
                grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn)
                fisher = fisher_calculation_partial(xs = x, ys = y, loss_grad_fn=grad_fn)
            elif data["input"].shape[0] == 1:
                # Only one datum in batch
                x, y = data["input"][0], data["target"][0]
                grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn)
                fisher = fisher_calculation_partial(xs=x, ys=y, loss_grad_fn=grad_fn)
            else:
                # Handle batch dimension of data explicitly
                x, y = data["input"], data["target"]
                grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn, handle_batches=True)
                fisher = fisher_calculation_partial(xs = x, ys = y, loss_grad_fn=grad_fn)
                fisher = mul(1./len(data["input"]), fisher) # divide by batch len to get mean
        
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
    \text{factor} \frac{1}{N}\cdot \sum_n J_n^\top \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=y_n,\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    #where $J_n$ is the Jacobian of the model w.r.t the parameters
    #evaluated at data point $n$, $c(y=y_n,\hat{y}=f_n)$ is the
    #loss function evaluated at data point $n$, and $v$ is the vector.
    #The `factor` is a scaling factor that is used to scale the Fisher matrix.

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
        The function assumes as a default that the data has a batch dimension.
    """
    if num_curv_samples is None:
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
    \text{factor} \cdot \frac{1}{NM}\sum_n,m J_n^\top \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    #where $J_n$ is the Jacobian of the model w.r.t the parameters
    #evaluated at data point $n$, $c(y,\hat{y})$ is the
    #loss function, and $v$ is the vector.
    #$\tilde{y}_{n,m}$ is the m-th Monte Carlo sample of the label under the liklihood
    # induced by the loss function: $r(y|f_n) = \exp(-c(y,\hat{y}=f_n))$
    #at data point $n$.
    #The `factor` is a scaling factor that
    #is used to scale the Fisher matrix.

    This function computes the above expression efficiently without hardcoding the
        dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the Fisher computation.
        factor: Scaling factor for the Fisher computation.
        key: PRNG Key to use for sampling
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        mc_samples: Number of MC samples to use. Defaults to 1.

    Returns:
        A function that takes a vector, a batch of data and a key,
        and computes the Monte-Carlo Fisher matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    loss_grad_fn = fetch_loss_gradient_fn(loss_fn, None, vmap_over_data=False)

            
    def mc_fisher_mv(vec, data, key):

        def mc_fisher_single_datum(datum, key):
            x = datum["input"]
            # Calculate forward pass and its derivative
            f_n, jvp = jax.linearize(lambda p: model_fn(x, p), params)

            def mc_fisher_single_label(y_sample):
                return fisher_calculation(f_n, jvp, y_sample, loss_grad_fn, vec)
                

            y_samples = sample_likelihood(loss_fn, f_n, mc_samples, key)
            vmap = jax.vmap(mc_fisher_single_label)(y_samples)
            return mean(vmap, axis=0) # over mc_samples dimension
        
        batch_size = data["input"].shape[0]
        keys = jax.random.split(key, batch_size)
        vmap = jax.vmap(mc_fisher_single_datum)(data, keys)
        fisher = mean(vmap, axis=0)  # over data batch dimension
        return mul(factor, fisher)

    return mc_fisher_mv


def sample_likelihood(loss_fn, f_n, mc_samples, key):
    # sample mc_samples values $\tilde{y} from e^{-\text{loss_fn}(y, f_n)}$
    if loss_fn == LossFn.MSE:
        unit_samples = jax.random.normal(key, shape=(mc_samples, *(f_n.shape)))
        return unit_samples + f_n[None,...]

    if loss_fn == LossFn.CROSS_ENTROPY:
        return jax.random.categorical(key, f_n, shape=(mc_samples, 1), replace=True)

    if loss_fn == LossFn.BINARY_CROSS_ENTROPY:
        bool_samples = jax.random.bernoulli(key, f_n, shape=(mc_samples, 1))
        return jnp.astype(bool_samples, jnp.float32)

    msg = f"Unsupported LossFn {loss_fn} to sample from."
    raise ValueError(msg)


def create_MC_fisher_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | str,
    key: KeyType,
    *,
    num_curv_samples: Int | None = None,
    num_total_samples: Int | None = None,
    vmap_over_data: bool = True,
    mc_samples: Int | None = 1,
) -> Callable[[Params], Params]:
    r"""Create Monte-Carlo approximated Fisher matrix-vector product without fixed data.

    The resulting matrix vector product computes:
    $$
    \text{factor} \cdot \frac{1}{NM}\sum_n,m J_n^\top \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right) \left(\nabla_{f_n}
    c(y=\tilde{y}_{n,m},\hat{y}=f_n)\right)^\top J_n \cdot v
    $$

    #where $J_n$ is the Jacobian of the model w.r.t the parameters
    #evaluated at data point $n$, $c(y,\hat{y})$ is the
    #loss function, and $v$ is the vector.
    #$\tilde{y}_{n,m}$ is the m-th Monte Carlo sample of the label under the liklihood
    # induced by the loss function: $r(y|f_n) = \exp(-c(y,\hat{y}=f_n))$
    #at data point $n$.
    #The `factor` is a scaling factor that
    #is used to scale the Fisher matrix.

    This function computes the above expression efficiently without hardcoding the
        dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the Fisher computation.
        key: PRNG Key to use for sampling
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
        A function that takes a vector and computes
        the Monte-Carlo Fisher matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    if num_curv_samples is None:
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

    def wrapped_fisher_mv(vec: Params) -> Params:
        return fisher_mv(vec, data, key)

    return wrapped_fisher_mv
