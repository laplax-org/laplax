import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

from laplax import util
from laplax.curv.cov import (
    CURVATURE_METHODS,
    CURVATURE_PRECISION_METHODS,
    CURVATURE_STATE_TO_COV,
    CURVATURE_STATE_TO_SCALE,
    CURVATURE_TO_POSTERIOR_STATE,
)
from laplax.curv.full import full_prec_to_scale

from .cases.covariance import case_posterior_covariance


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def key(request) -> np.random.Generator:
    return jax.random.key(request.param)


@pytest_cases.fixture
def prior_prec():
    """Fixture for precision matrix.

    Returns:
        A valid precision matrix.
    """
    return jnp.array([[3.0, 0.5], [0.5, 2.0]])


@pytest_cases.fixture
def invalid_prec():
    """Fixture for invalid precision matrix.

    Returns:
        An invalid precision matrix.
    """
    return jnp.array([[1.0, 2.0], [2.0, 1.0]])


# --------------------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------------------


def test_prec_to_scale(prior_prec):
    """Test `prec_to_scale` for valid input."""
    scale = full_prec_to_scale(prior_prec)
    assert scale.shape == prior_prec.shape, (
        "Scale matrix shape should match precision matrix shape."
    )
    assert jnp.all(jnp.linalg.eigvals(scale) > 0), (
        "Scale matrix should be positive definite."
    )
    assert jnp.allclose(
        scale @ scale.T @ prior_prec, jnp.eye(prior_prec.shape[0]), atol=1e-6, rtol=1e-6
    )


# ValueError message removed from code due to jit compilation issues.
# def test_prec_to_scale_invalid(invalid_prec):
#     """Test `prec_to_scale` for invalid input."""
#     with pytest.raises(ValueError, match="matrix is not positive definite"):
#         full_prec_to_scale(invalid_prec)


@pytest_cases.parametrize_with_cases("task", cases=case_posterior_covariance)
def test_posterior_covariance_est(task):
    # Get low rank terms
    curv_estimate = CURVATURE_METHODS[task.method](
        mv=task.arr_mv,
        layout=task.layout,
        key=task.key_curv_estimate,
        rank=task.rank,
    )
    assert jnp.allclose(
        task.adjust_curv_estimate(curv_estimate),
        task.true_curv,
        atol=task.atol,
        rtol=task.rtol,
    )

    # Get and test precision matrix
    prec = CURVATURE_PRECISION_METHODS[task.method](
        curv_estimate=curv_estimate,
        prior_arguments={"prior_prec": 1.0},
    )
    prec_dense = task.adjust_prec(prec)
    assert jnp.allclose(
        prec_dense,
        task.true_curv + jnp.eye(task.size),
        atol=task.atol,
        rtol=task.rtol,
    )

    # Create posterior state
    state = CURVATURE_TO_POSTERIOR_STATE[task.method](prec)

    # Get and test scale matrix
    scale_mv = CURVATURE_STATE_TO_SCALE[task.method](state)
    scale_dense = util.mv.to_dense(scale_mv, layout=task.layout)
    assert jnp.allclose(
        scale_dense @ scale_dense.T @ prec_dense,
        jnp.eye(task.size),
        atol=task.atol,
        rtol=task.rtol,
    )

    # Get and test covariance matrix
    cov_mv = CURVATURE_STATE_TO_COV[task.method](state)
    cov_dense = util.mv.to_dense(cov_mv, layout=task.layout)
    assert jnp.allclose(
        cov_dense @ prec_dense,
        jnp.eye(task.size),
        atol=task.atol,
        rtol=task.rtol,
    )
