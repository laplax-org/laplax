"""Enums for Laplace approximations."""

from enum import StrEnum


class LossFn(StrEnum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"


class CurvApprox(StrEnum):
    FULL = "full"
    DIAGONAL = "diagonal"
    LANCZOS = "lanczos"
    LOBPCG = "lobpcg"

class LowRankMethod(StrEnum):
    LANCZOS = "lanczos"
    LOBPCG = "lobpcg"


class CalibrationErrorNorm(StrEnum):
    L1 = "l1"
    INF = "inf"
