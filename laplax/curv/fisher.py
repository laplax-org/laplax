"""Fisher Matrix Vector Product"""

from collections.abc import Callable

import jax

from loguru import logger

from laplax.curv.loss import create_loss_hessian_mv


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
	loss_hessian_mv: Callable | None = None,
	mc_samples: Int | None = None,
) -> Callable[[Params, Data], Params]:
	r"""Create Fisher matrix-vector product without fixed data.

    ## TODO: Add mathy description

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
    	type: The type of Fisher approximation.
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        vmap_over_data: Whether to vmap over the data. Defaults to True.
        loss_hessian_mv: The loss Hessian matrix-vector product.
		mc_samples: Number of MC samples to use for type "MC". Defaults to 1.

    Returns:
        A function that takes a vector and a batch of data, and computes the Fisher
        matrix-vector product.

    Note:
        The function assumes as a default that the data has a batch dimension.

    """

    # Create loss Hessian-vector product
	loss_hessian_mv = loss_hessian_mv or create_loss_hessian_mv(loss_fn)
	
	if vmap_over_data:
		loss_hessian_mv = jax.vmap(loss_hessian_mv)

	def empirical_fisher_mv(vec, data):
		raise NotImplementedError

	def mc_fisher_mv(vec, data):
		raise NotImplementedError

	if type == FisherType.EMPIRICAL:
		if mc_samples is not None:
			logger.warning("Parameter 'mc_samples' does not affect FisherType 'EMPIRICAL'. Did you mean to use FisherType 'MC'?")
		return empirical_fisher_mv

	if type == FisherType.MC:
		mc_samples = mc_samples or 1
		return mc_fisher_mv

	msg = f"Fisher Type must be either 'EMPIRICAL' or 'MC'. Got {type} instead."
	raise ValueError(msg)