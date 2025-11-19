from loguru import logger

from laplax.api import calibration_options
from laplax.curv.cov import (
    CURVATURE_METHODS,
    CURVATURE_PRECISION_METHODS,
    CURVATURE_STATE_TO_COV,
    CURVATURE_STATE_TO_SCALE,
    CURVATURE_TO_POSTERIOR_STATE,
)
from laplax.eval.likelihood import CURVATURE_MARGINAL_LOG_LIKELIHOOD
from laplax.types import (
    Any,
    Callable,
    CurvApprox,
    CurvatureMV,
    FlatParams,
    Layout,
    PosteriorState,
)

# ----------------------------------------------------------------------------------
# Register new calibration methods
# ----------------------------------------------------------------------------------


def register_calibration_method(
    method_name: str,
    method_fn: Callable,
) -> None:
    """Register a new calibration method.

    Args:
        method_name: Name of the calibration method.
        method_fn: Function implementing the calibration method.

    Notes:
        The method function should have signature
        `method_fn(objective: Callable, **kwargs) -> float`
    """
    calibration_options[method_name] = method_fn
    logger.info(f"Registered new calibration method: {method_name}")


# ----------------------------------------------------------------------------------
# Register new curvature methods
# ----------------------------------------------------------------------------------


def register_curvature_method(
    name: str,
    *,
    create_curvature_fn: Callable[[CurvatureMV, Layout, Any], Any] | None = None,
    curvature_to_precision_fn: Callable | None = None,
    prec_to_posterior_fn: Callable | None = None,
    posterior_state_to_scale_fn: Callable[
        [PosteriorState], Callable[[FlatParams], FlatParams]
    ]
    | None = None,
    posterior_state_to_cov_fn: Callable[
        [PosteriorState], Callable[[FlatParams], FlatParams]
    ]
    | None = None,
    marginal_log_likelihood_fn: Callable | None = None,
    default: CurvApprox | None = None,
) -> None:
    """Register a new curvature method with optional custom functions.

    This function allows adding new curvature methods with their corresponding
    functions for creating curvature estimates, adding prior information,
    computing posterior states, and deriving matrix-vector product functions
    for scale and covariance.

    Args:
        name: Name of the new curvature method.
        create_curvature_fn: Custom function to create the curvature
            estimate. Defaults to None.
        curvature_to_precision_fn: Custom function to convert the curvature
            estimate to a posterior precision matrix. Defaults to None.
        prec_to_posterior_fn: Custom function to convert the posterior precision
            matrix to a posterior state. Defaults to None.
        posterior_state_to_scale_fn: Custom function to compute scale
            matrix-vector products. Defaults to None.
        posterior_state_to_cov_fn: Custom function to compute covariance
            matrix-vector products. Defaults to None.
        marginal_log_likelihood_fn: Custom function to compute the marginal
            log-likelihood. Defaults to None.
        default: Default method to inherit missing
            functionality from. Defaults to None.

    Raises:
        ValueError: If no default is provided and required functions are missing.
    """
    # Check whether default is given
    if default is None and not all((
        create_curvature_fn,
        curvature_to_precision_fn,
        prec_to_posterior_fn,
        posterior_state_to_scale_fn,
        posterior_state_to_cov_fn,
        marginal_log_likelihood_fn,
    )):
        missing_functions = [
            fn_name
            for fn_name, fn in zip(
                [
                    "create_curvature_fn",
                    "curvature_to_precision_fn",
                    "prec_to_posterior_fn",
                    "posterior_state_to_scale_fn",
                    "posterior_state_to_cov_fn",
                    "marginal_log_likelihood_fn",
                ],
                [
                    create_curvature_fn,
                    curvature_to_precision_fn,
                    prec_to_posterior_fn,
                    posterior_state_to_scale_fn,
                    posterior_state_to_cov_fn,
                    marginal_log_likelihood_fn,
                ],
                strict=True,
            )
            if fn is None
        ]
        msg = (
            "Either a default method must be provided or the following functions must "
            f"be specified: {', '.join(missing_functions)}."
        )
        raise ValueError(msg)

    CURVATURE_METHODS[name] = create_curvature_fn or CURVATURE_METHODS[default]
    CURVATURE_PRECISION_METHODS[name] = (
        curvature_to_precision_fn or CURVATURE_PRECISION_METHODS[default]
    )
    CURVATURE_TO_POSTERIOR_STATE[name] = (
        prec_to_posterior_fn or CURVATURE_TO_POSTERIOR_STATE[default]
    )
    CURVATURE_STATE_TO_SCALE[name] = (
        posterior_state_to_scale_fn or CURVATURE_STATE_TO_SCALE[default]
    )
    CURVATURE_STATE_TO_COV[name] = (
        posterior_state_to_cov_fn or CURVATURE_STATE_TO_COV[default]
    )
    CURVATURE_MARGINAL_LOG_LIKELIHOOD[name] = (
        marginal_log_likelihood_fn or CURVATURE_MARGINAL_LOG_LIKELIHOOD[default]
    )
