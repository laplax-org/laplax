"""Fisher Matrix Vector Product"""

from collections.abc import Callable

import jax

from loguru import logger

from laplax.curv.loss import fetch_loss_gradient_fn


from laplax.enums import LossFn
from laplax.enums import FisherType
from laplax.types import (
    Data,
    Float,
    Int,
    ModelFn,
    Params,
)

def create_fisher_mv_without_data(
	type: FisherType | str,
	model_fn: ModelFn,
	params: Params,
	loss_fn: LossFn | str | Callable | None,
	factor: Float,
	*,
	vmap_over_data: bool = True,
	loss_gradient_fn: Callable | None = None,
	mc_samples: Int | None = None,
) -> Callable[[Params, Data], Params]:
	r"""Create Fisher matrix-vector product without fixed data.
	Uses either Empirical or Monte Carlo approximation.
    #$$
    #\text{factor} \cdot \sum_i J_i^\top \nabla^2_{f(x_i, \theta), f(x_i, \theta)}
    #\mathcal{L}(f(x_i, \theta), y_i) J_i \cdot v
    #$$

    #where $J_i$ is the Jacobian of the model at data point $i$, $H_{L, i}$ is the
    #Hessian of the loss, and $v$ is the vector. The `factor` is a scaling factor that
    #is used to scale the GGN matrix.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
    	type: The type of Fisher approximation. Either 'EMPIRICAL' or 'MC'.
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the Fisher computation.
        factor: Scaling factor for the Fisher computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_grad_fn: The loss gradient function.
		mc_samples: Number of MC samples to use for type 'MC'. Defaults to 1.

    Returns:
        A function that takes a vector and a batch of data, and computes the Fisher
        matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.

    """

	# Create loss gradient product
	loss_grad_fn = fetch_loss_gradient_fn(loss_fn, loss_gradient_fn, vmap_over_data)

	def empirical_fisher_mv(vec, data):
		def fwd(p):
			# Step 1: Single jvp for entire batch, if vmap_over_data is True
			if vmap_over_data:
				return jax.vmap(lambda x: model_fn(input=x, params=p))(data["input"])
			return model_fn(input=data["input"], params=p)

		# Step 2: Linearize the forward pass
		z, jvp = jax.linearize(fwd, params)

		grad = loss_grad_fn(fwd(params), data["target"])
		vjp = jax.linear_transpose(jvp, vec)

		Jv = jvp(vec)
		GtJv = grad.T @ Jv
		GGtJv = grad @ GtJv
		JtGGtJv = vjp(GGtJv)
		return JtGGtJv

	def mc_fisher_mv(vec, data):
		raise NotImplementedError

	if type == FisherType.EMPIRICAL:
		if mc_samples is not None:
			# This is not an error (does not prevent computation), but is likely unintended -> Warning
			logger.warning("Parameter 'mc_samples' does not affect FisherType 'EMPIRICAL'. Did you mean to use FisherType 'MC'?")
		return empirical_fisher_mv

	if type == FisherType.MC:
		mc_samples = mc_samples or 1
		return mc_fisher_mv

	msg = f"Fisher Type must be either 'EMPIRICAL' or 'MC'. Got {type} instead."
	raise ValueError(msg)