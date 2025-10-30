import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from laplax.curv.ggn import create_ggn_mv

from .cases.rosenbrock import RosenbrockCase


# ---------------------------------------------------------------
# GGN - Rosenbrock
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def test_ggn_rosenbrock(rosenbrock):
    # Setup ggn_mv
    ggn_mv = create_ggn_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        num_curv_samples=1,
        num_total_samples=1,
    )

    # Compute the GGN
    ggn_calc = jax.lax.map(ggn_mv, jnp.eye(2))

    # Compare with the manual GGN
    ggn_manual = rosenbrock.ggn_manual
    assert jnp.allclose(ggn_calc, ggn_manual)
