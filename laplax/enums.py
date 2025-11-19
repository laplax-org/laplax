"""Enums for Laplace approximations."""

from enum import StrEnum


class LossFn(StrEnum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    NONE = "none"


class CurvApprox(StrEnum):
    FULL = "full"
    DIAGONAL = "diagonal"
    LANCZOS = "lanczos"
    LOBPCG = "lobpcg"


class CovarianceStructure(StrEnum):
    """Structure of the prior covariance used for FSP inference."""

    NONE = "none"
    KRONECKER = "kronecker"


class LowRankMethod(StrEnum):
    LANCZOS = "lanczos"
    LOBPCG = "lobpcg"


class CalibrationErrorNorm(StrEnum):
    L1 = "l1"
    INF = "inf"


# ------------------------------------------------------------------------------
# Additional laplax.api specific enumerations
# ------------------------------------------------------------------------------


class CalibrationObjective(StrEnum):
    """Supported calibration objectives (minimisation!)."""

    NLL = "nll"
    CHI_SQUARED = "chi_squared"
    MARGINAL_LOG_LIKELIHOOD = "marginal_log_likelihood"
    ECE = "ece"


class CalibrationMethod(StrEnum):
    """Supported calibration methods."""

    GRID_SEARCH = "grid_search"


class Pushforward(StrEnum):
    """Supported pushforward types for pushing forward weight space uncertainty."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class Predictive(StrEnum):
    """Supported predictive types for crossing softmax transformation."""

    MC_BRIDGE = "mc_bridge"
    LAPLACE_BRIDGE = "laplace_bridge"
    MEAN_FIELD_0 = "mean_field_0"
    MEAN_FIELD_1 = "mean_field_1"
    MEAN_FIELD_2 = "mean_field_2"
    NONE = "none"  # Intended for regression flag


class DefaultMetrics(StrEnum):
    """Supported default metric settings."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
