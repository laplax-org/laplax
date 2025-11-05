import jax.numpy as jnp
import pytest
import pytest_cases

from laplax.curv.fisher import create_fisher_mv_without_data
from laplax.enums import FisherType

@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def empirical_fisher(rosenbrock):
	fisher_mvp = create_fisher_mv_without_data(
		FisherType.EMPIRICAL,
		model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        #data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        factor=1.0,
        vmap_over_data=True)
	fisher_mvp(jnp.eye(10))
