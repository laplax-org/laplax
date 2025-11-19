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
    ModelFn,
    Params,
)
from laplax.util.tree import mul,mean


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

    def transpose(linop, example_input):
        # SImple wrapper around jax.linear_transpose to 
        # directly return transpose of linop with single argument 
        t = jax.linear_transpose(linop, example_input)
        return lambda v: t(v)[0]


    def emp_fisher_single_datum(x,y, vec):
        # Forward pass
        f_evaluated = model_fn(input=x, params=params).squeeze()
        
        # Construct jvp/vjp of forward pass
        # atleast_2d ensures jvp/vjp have signature expected by fisher calculation
        model_as_fn_of_params = lambda p: jnp.atleast_2d(model_fn(input=x, params=p))
        jvp = jax.linearize(model_as_fn_of_params, params)[1]
        vjp = transpose(jvp, vec)
        
        # Construct gradient mv and its transpose
        grad = loss_grad_fn(f_evaluated, y)[:,None]
        grad_mv = lambda v: grad @ v
        grad_T_mv = transpose(grad_mv, jnp.zeros((1,1)))

        # nest matrix vector product calls
        Jv = jvp(vec)
        GtJv = grad_T_mv(Jv)
        GGtJv = grad_mv(GtJv)
        JtGGtJv = vjp(GGtJv)
        return mul(factor, JtGGtJv)

    def empirical_fisher_mv(vec, data):
        if vmap_over_data:
            return mean(jax.vmap(lambda datum: emp_fisher_single_datum(datum["input"],datum["target"], vec))(data), axis=0)
        return emp_fisher_single_datum(data["input"], data["target"], vec)
    
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
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        mc_samples: Number of MC samples to use. Defaults to 1.

    Returns:
        A function that takes a vector and a batch of data,
        and computes the Monte-Carlo Fisher matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.
    """

    def mc_fisher_mv(vec, data):

        def fwd(p):
            # Step 1: Single jvp for entire batch, if vmap_over_data is True
            if vmap_over_data:
                return jax.vmap(lambda x: model_fn(input=x, params=p))(data["input"])
            return model_fn(input=data["input"], params=p)
        
        _, jvp = jax.linearize(fwd, params)
        vjp = jax.linear_transpose(jvp, vec)

        S_mv = would_be_grad_mv(loss_fn, mc_samples, data)
        St_mv = jax.linear_transpose(S_mv, jnp.zeros(mc_samples))
        
        StJv = St_mv(Jv)[0]
        SStJv = S_mv(StJv)
        JtSStJv = vjp(SStJv)[0]
        return mul(factor, JtSStJv)

    return mc_fisher_mv
