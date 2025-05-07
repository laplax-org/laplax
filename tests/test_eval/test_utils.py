"""Test for utility functions."""

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_fn
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.metrics import (
    DEFAULT_REGRESSION_METRICS,
    DEFAULT_REGRESSION_METRICS_DICT,
)
from laplax.eval.pushforward import set_lin_pushforward
from laplax.eval.utils import evaluate_metrics_on_dataset

from .cases.regression import case_regression


@pytest_cases.parametrize(
    "curv_op",
    ["full"],
)
@pytest_cases.parametrize_with_cases("task", cases=case_regression)
def test_eval_metrics(curv_op, task):
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

    # Set pushforward
    pushforward = set_lin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        prior_arguments={"prior_prec": 99999999.0},
        num_samples=5,  # TODO(2bys): Find a better way of setting this.
    )

    results_metrics_dict = evaluate_metrics_on_dataset(
        pushforward,
        data,
        metrics_dict=DEFAULT_REGRESSION_METRICS_DICT,
    )
    results_metrics = evaluate_metrics_on_dataset(
        pushforward,
        data,
        metrics=DEFAULT_REGRESSION_METRICS,
    )

    # Check metrics are positive
    assert all(results_metrics["rmse"] > 0)
    assert all(results_metrics["chi^2"] > 0)

    # Check shapes match within each dict
    comparison = next(iter(results_metrics.values())).shape
    assert all(k.shape == comparison for k in results_metrics.values())

    # Check metrics match between both approaches
    assert jnp.allclose(results_metrics["rmse"], results_metrics_dict["rmse"])
    assert jnp.allclose(results_metrics["chi^2"], results_metrics_dict["chi^2"])
    assert jnp.allclose(results_metrics["nll"], results_metrics_dict["nll"])
    assert jnp.allclose(results_metrics["crps"], results_metrics_dict["crps"])
