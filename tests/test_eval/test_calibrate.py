"""Tests for grid search calibration."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_fn
from laplax.curv.ggn import create_ggn_mv
from laplax.enums import CurvApprox
from laplax.eval.calibrate import (
    evaluate_for_given_prior_arguments,
    optimize_prior_prec,
)
from laplax.eval.metrics import chi_squared_zero
from laplax.eval.pushforward import set_lin_pushforward

from .cases.regression import case_regression


@pytest_cases.parametrize("curv_op", [CurvApprox.FULL])
@pytest_cases.parametrize_with_cases("task", cases=case_regression)
def test_lin_pushforward(curv_op, task):
    """Test for pipeline integration of calibration function."""
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    num_training_samples = 1000
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # Set get posterior function
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=task.loss_fn_type,
        num_total_samples=num_training_samples,
    )
    posterior_fn = create_posterior_fn(
        curv_op,
        ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward for calibration objective
    set_prob_predictive = partial(
        set_lin_pushforward,
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        num_samples=5,  # TODO(2bys): Find a better way of setting this.
    )

    def calibration_objective(prior_arguments):
        return evaluate_for_given_prior_arguments(
            prior_arguments=prior_arguments,
            data=data,
            set_prob_predictive=set_prob_predictive,
            calibration_metric=chi_squared_zero,
        )

    # Optimize
    prior_prec = optimize_prior_prec(
        objective=calibration_objective,
        grid_size=10,
    )

    # Calculate values for comparison.
    prior_prec_interval = jnp.logspace(
        start=-5.0,  # Default values
        stop=6.0,
        num=10,
    )

    # Calculate
    true_val = calibration_objective(prior_arguments={"prior_prec": prior_prec})
    comparison_prec = calibration_objective(
        prior_arguments={"prior_prec": prior_prec_interval[-1]}
    )
    assert true_val <= comparison_prec
