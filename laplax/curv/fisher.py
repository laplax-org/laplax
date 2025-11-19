"""Fisher Matrix Vector Product."""

from collections.abc import Callable

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


def transpose(linop, example_input):
    # Simple wrapper around jax.linear_transpose to
    # directly return transpose of linop with single argument
    t = jax.linear_transpose(linop, example_input)
    return lambda v: t(v)[0]


def fisher_structure_calculation(jvp, grads_vp, vec, M=1):
    r"""Nests matrix vector product calls as needed for fisher calculation.

    Calculates
    $$
    \text{jvp}^\top (\text{grads_vp}(\text{grads_vp}^\top(\text{jvp}(\text{vec}))))
    $$

    Args:
        jvp: A callable mapping a PyTree to a vector of shape (O,)
        grads_vp: A callable mapping a vector of shape ('M',) to  vector of shape (O,)
        vec: A PyTree that can be consumed by jvp
        M: Number of gradients provided

    Returns:
        The unscaled fisher matrix vector poduct for one datum
    """
    vjp = transpose(jvp, vec)
    v_grads_p = transpose(grads_vp, jnp.zeros((M, 1)))

    Jv = jvp(vec)
    GtJv = v_grads_p(Jv)
    GGtJv = grads_vp(GtJv)
    JtGGtJv = vjp(GGtJv)
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
        The function assumes as a default that the data has a batch dimension.
    """
    loss_grad_fn = fetch_loss_gradient_fn(loss_fn, loss_grad_fn, vmap_over_data=False)

    def empirical_fisher_mv(vec, data):
        def emp_fisher_single_datum(datum):
            x, y = datum["input"], datum["target"]
            # Forward pass
            f_evaluated = model_fn(input=x, params=params).squeeze()

            # Construct jvp of forward pass
            # atleast_2d ensures jvp has signature expected by fisher calculation
            jvp = jax.linearize(lambda p: jnp.atleast_2d(model_fn(x, p)), params)[1]

            # Construct gradient mv
            grad = loss_grad_fn(f_evaluated, y)[:, None]

            fisher = fisher_structure_calculation(jvp, lambda v: grad @ v, vec)
            return mul(factor, fisher)

        if vmap_over_data:
            vmap = jax.vmap(emp_fisher_single_datum)(data)
            return mean(vmap, axis=0)
        return emp_fisher_single_datum(data, vec)

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
    loss_fn: LossFn | str | Callable | None,
    factor: Float,
    key: KeyType,
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
        A function that takes a vector and a batch of data,
        and computes the Monte-Carlo Fisher matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """
    loss_grad_fn = fetch_loss_gradient_fn(loss_fn, None, vmap_over_data=False)

    def mc_fisher_mv(vec, data):
        def mc_fisher_single_datum(datum):
            x = datum["input"]  # actual y never used for MC Fisher
            # Forward pass
            f_evaluated = model_fn(input=x, params=params)

            # Construct jvp of forward pass
            # atleast_2d ensures jvp has signature expected by fisher calculation
            jvp = jax.linearize(lambda p: jnp.atleast_2d(model_fn(x, p)), params)[1]

            # Construct would-be-gradients mv
            y_samples = sample_likelihood(loss_fn, f_evaluated, mc_samples, key)

            # TODO @Luis: Is this (O,M) = loss_grad_fn( (O,1), (O,M) ) ?
            grad = loss_grad_fn(f_evaluated, y_samples)

            fisher = fisher_structure_calculation(jvp, lambda v: grad @ v, vec)
            return mul(factor, fisher)

        if vmap_over_data:
            vmap = jax.vmap(mc_fisher_single_datum)(data)
            return mean(vmap, axis=0)
        return mc_fisher_single_datum(data["input"], data["target"], vec)

    return mc_fisher_mv


def sample_likelihood(loss_fn, f_n, mc_samples, key):
    # sample M values $\tilde{y} from e^{-\text{loss_fn}(y, f_n)}$
    if loss_fn is LossFn.MSE:
        unit_samples = jax.random.normal(key, shape=(f_n.shape[0], mc_samples))
        return unit_samples + f_n[:, None]

    if loss_fn is LossFn.CROSS_ENTROPY:
        return jax.random.categorical(key, f_n, shape=(1, mc_samples), replace=True)
    msg = f"Unsupported LossFn {loss_fn} to sample from."
    raise ValueError(msg)
