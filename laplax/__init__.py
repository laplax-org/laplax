"""Package for Laplace approximations in JAX."""

import importlib.metadata

from laplax.api import calibration, evaluation, laplace

__all__ = ["calibration", "evaluation", "laplace"]
__version__ = importlib.metadata.version("laplax")
