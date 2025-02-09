"""Calibration utilities for optimizing prior precision in probabilistic models.

This script provides utilities for optimizing prior precision in probabilistic models.
It includes functions to:

- Evaluate metrics for given prior arguments and datasets.
- Perform grid search to optimize prior precision using objective functions.
- Optimize prior precision over a logarithmic grid interval.

The script leverages JAX for numerical operations, Loguru for logging, and custom
utilities from the `laplax` package.
"""

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from laplax.eval.metrics import estimate_q
from laplax.types import Array, Data, Float, PriorArguments


# Calibrate prior
def calibration_metric(**predictions) -> Float:
    r"""Computes a calibration metric for a given set of predictions.

    The calculated metric is the ratio between the error of the prediction and
    the variance of the output uncertainty.

    Args:
        **predictions: Keyword arguments representing the model predictions,
        typically including mean, variance, and target.

    Returns:
        The calibration metric value.
    """
    return jnp.abs(estimate_q(**predictions) - 1)


def evaluate_for_given_prior_arguments(
    *,
    data: Data,
    set_prob_predictive: Callable,
    metric: Callable = calibration_metric,
    **kwargs,
) -> Float:
    """Evaluate the metric for a given set of prior arguments and data.

    This function computes predictions for the input data using a probabilistic
    predictive function generated by `set_prob_predictive`. It then evaluates a
    specified metric using these predictions.

    Args:
        data: Dataset containing inputs and targets.
        set_prob_predictive: A callable that generates a probabilistic predictive
            function.
        metric: A callable metric function to evaluate the predictions
            (default: `calibration_metric`).
        **kwargs: Additional arguments passed to `set_prob_predictive` and
            `jax.lax.map`.

    Returns:
        The evaluated metric value.
    """
    prob_predictive = set_prob_predictive(**kwargs)

    def evaluate_data(dp: Data) -> dict[str, Array]:
        return {**prob_predictive(dp["input"]), "target": dp["target"]}

    res = metric(
        **jax.lax.map(
            evaluate_data,
            data,
            batch_size=kwargs.get(
                "evaluate_for_given_prior_arguments_batch_size",
                kwargs.get("data_batch_size"),
            ),
        )
    )
    return res


def grid_search(
    prior_prec_interval: Array,
    objective: Callable[[PriorArguments], float],
    patience: int = 5,
    max_iterations: int | None = None,
) -> Float:
    """Perform grid search to optimize prior precision.

    This function iteratively evaluates an objective function over a range of
    prior precisions. It tracks the performance and stops early if results
    increase consecutively for a specified number of iterations (`patience`).
    The prior precision which scores the lowest is returned.

    Args:
        prior_prec_interval: An array of prior precision values to search.
        objective: A callable objective function that takes `PriorArguments` as input
            and returns a float result.
        patience: The number of consecutive iterations with increasing results to
            tolerate before stopping (default: 5).
        max_iterations: The maximum number of iterations to perform (default: None).

    Returns:
        The prior precision value that minimizes the objective function.

    Raises:
        ValueError: If the objective function returns invalid results.
    """
    results, prior_precs = [], []
    increasing_count = 0
    previous_result = None

    for iteration, prior_prec in enumerate(prior_prec_interval):
        start_time = time.perf_counter()
        try:
            result = objective({"prior_prec": prior_prec})
        except ValueError as error:
            logger.warning(f"Caught an exception in validate {error}")
            result = float("inf")

        if jnp.isnan(result):
            logger.info("Caught nan, setting result to inf.")
            result = float("inf")

        # Logging for performance and tracking
        logger.info(
            f"Took {time.perf_counter() - start_time:.4f} seconds, "
            f"prior prec: {prior_prec:.4f}, "
            f"result: {result:.6f}",
        )

        results.append(result)
        prior_precs.append(prior_prec)

        # If we have a previous result, check if the result has increased
        if previous_result is not None:
            if result > previous_result:
                increasing_count += 1
                logger.info(f"Result increased, increasing_count = {increasing_count}")
            else:
                increasing_count = 0

            # Stop if the results have increased for `patience` consecutive iterations
            if increasing_count >= patience:
                break

        previous_result = result

        # Check if maximum iterations reached
        if max_iterations is not None and iteration >= max_iterations:
            logger.info(f"Stopping due to reaching max iterations = {max_iterations}")
            break

    best_prior_prec = prior_precs[np.nanargmin(results)]
    logger.info(f"Chosen prior prec = {best_prior_prec:.4f}")

    return best_prior_prec


def optimize_prior_prec(
    objective: Callable[[PriorArguments], float],
    log_prior_prec_min: float = -5.0,
    log_prior_prec_max: float = 6.0,
    grid_size: int = 300,
) -> Float:
    """Optimize prior precision using logarithmic grid search.

    This function creates a logarithmically spaced interval of prior precision
    values and performs a grid search to find the optimal value that minimizes
    the specified objective function.

    Args:
        objective: A callable objective function that takes `PriorArguments` as input
            and returns a float result.
        log_prior_prec_min: The base-10 logarithm of the minimum prior precision
            value (default: -5.0).
        log_prior_prec_max: The base-10 logarithm of the maximum prior precision
            value (default: 6.0).
        grid_size: The number of points in the grid interval (default: 300).

    Returns:
        The optimized prior precision value.
    """
    prior_prec_interval = jnp.logspace(
        start=log_prior_prec_min,
        stop=log_prior_prec_max,
        num=grid_size,
    )
    prior_prec = grid_search(
        prior_prec_interval,
        objective,
    )

    return prior_prec
