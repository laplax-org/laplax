"""`eval` module contains pushforwards, calibration and metrics."""

from .calibrate import evaluate_for_given_prior_arguments, optimize_prior_prec
from .likelihood import (
    joint_log_likelihood,
    marginal_log_likelihood,
)
from .pushforward import set_lin_pushforward, set_nonlin_pushforward
from .utils import (
    apply_fns,
    evaluate_metrics_on_dataset,
    evaluate_metrics_on_generator,
    evaluate_on_dataset,
    transfer_entry,
)

__all__ = [
    "apply_fns",
    "evaluate_for_given_prior_arguments",
    "evaluate_metrics_on_dataset",
    "evaluate_metrics_on_generator",
    "evaluate_on_dataset",
    "joint_log_likelihood",
    "marginal_log_likelihood",
    "optimize_prior_prec",
    "set_lin_pushforward",
    "set_nonlin_pushforward",
    "transfer_entry",
]
