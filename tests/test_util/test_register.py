from unittest.mock import Mock

import pytest

from laplax.register import (
    CURVATURE_MARGINAL_LOG_LIKELIHOOD,
    CURVATURE_METHODS,
    CURVATURE_PRECISION_METHODS,
    CURVATURE_STATE_TO_COV,
    CURVATURE_STATE_TO_SCALE,
    CURVATURE_TO_POSTERIOR_STATE,
    register_curvature_method,
)


@pytest.mark.parametrize(
    (
        "name",
        "create_curvature_fn",
        "curvature_to_precision_fn",
        "prec_to_posterior_fn",
        "posterior_state_to_scale_fn",
        "posterior_state_to_cov_fn",
        "marginal_log_likelihood_fn",
        "default",
    ),
    [
        (
            "test_method",
            Mock(name="create_curvature_fn"),
            Mock(name="curvature_to_precision_fn"),
            Mock(name="prec_to_posterior_fn"),
            Mock(name="posterior_state_to_scale_fn"),
            Mock(name="posterior_state_to_cov_fn"),
            Mock(name="marginal_log_likelihood_fn"),
            None,
        ),
    ],
)
def test_register_curvature_method(
    name,
    create_curvature_fn,
    curvature_to_precision_fn,
    prec_to_posterior_fn,
    posterior_state_to_scale_fn,
    posterior_state_to_cov_fn,
    marginal_log_likelihood_fn,
    default,
):
    register_curvature_method(
        name=name,
        create_curvature_fn=create_curvature_fn,
        curvature_to_precision_fn=curvature_to_precision_fn,
        prec_to_posterior_fn=prec_to_posterior_fn,
        posterior_state_to_scale_fn=posterior_state_to_scale_fn,
        posterior_state_to_cov_fn=posterior_state_to_cov_fn,
        marginal_log_likelihood_fn=marginal_log_likelihood_fn,
        default=default,
    )

    assert CURVATURE_METHODS[name] == create_curvature_fn
    assert CURVATURE_PRECISION_METHODS[name] == curvature_to_precision_fn
    assert CURVATURE_TO_POSTERIOR_STATE[name] == prec_to_posterior_fn
    assert CURVATURE_STATE_TO_SCALE[name] == posterior_state_to_scale_fn
    assert CURVATURE_STATE_TO_COV[name] == posterior_state_to_cov_fn
    assert CURVATURE_MARGINAL_LOG_LIKELIHOOD[name] == marginal_log_likelihood_fn


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("default_test", "lanczos"),
    ],
)
def test_register_curvature_method_with_default(name, default):
    register_curvature_method(name=name, default=default)

    assert CURVATURE_METHODS[name] == CURVATURE_METHODS[default]
    assert CURVATURE_PRECISION_METHODS[name] == CURVATURE_PRECISION_METHODS[default]
    assert CURVATURE_TO_POSTERIOR_STATE[name] == CURVATURE_TO_POSTERIOR_STATE[default]
    assert CURVATURE_STATE_TO_SCALE[name] == CURVATURE_STATE_TO_SCALE[default]
    assert CURVATURE_STATE_TO_COV[name] == CURVATURE_STATE_TO_COV[default]
    assert (
        CURVATURE_MARGINAL_LOG_LIKELIHOOD[name]
        == CURVATURE_MARGINAL_LOG_LIKELIHOOD[default]
    )


def test_register_curvature_method_missing_functions():
    with pytest.raises(ValueError, match="must be specified"):
        register_curvature_method(name="incomplete_test")


@pytest.mark.parametrize(
    ("name", "create_curvature_fn", "default"),
    [
        ("partial_test", Mock(name="create_curvature_fn"), "lanczos"),
    ],
)
def test_register_curvature_method_partial(name, create_curvature_fn, default):
    register_curvature_method(
        name=name, create_curvature_fn=create_curvature_fn, default=default
    )

    assert CURVATURE_METHODS[name] == create_curvature_fn
    assert CURVATURE_PRECISION_METHODS[name] == CURVATURE_PRECISION_METHODS[default]
    assert CURVATURE_TO_POSTERIOR_STATE[name] == CURVATURE_TO_POSTERIOR_STATE[default]
    assert CURVATURE_STATE_TO_SCALE[name] == CURVATURE_STATE_TO_SCALE[default]
    assert CURVATURE_STATE_TO_COV[name] == CURVATURE_STATE_TO_COV[default]
    assert (
        CURVATURE_MARGINAL_LOG_LIKELIHOOD[name]
        == CURVATURE_MARGINAL_LOG_LIKELIHOOD[default]
    )
