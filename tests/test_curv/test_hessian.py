import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from laplax.curv.hessian import create_hessian_mv

from .cases.rosenbrock import RosenbrockCase

# ---------------------------------------------------------------
# Hessian - Rosenbrock
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def test_hessian_rosenbrock(rosenbrock):
    hessian_mv = create_hessian_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        num_total_samples=1,
        vmap_over_data=False,
    )

    hessian_calc = jax.lax.map(hessian_mv, jnp.eye(2))
    hessian_manual = rosenbrock.hessian_manual
    assert jnp.allclose(hessian_calc, hessian_manual)
