"""Package for Laplace approximations in JAX.

Avoid importing heavy submodules at package import time to keep import side
effects minimal (e.g., JAX PRNG initialization). Expose top-level symbols via
lazy attribute access.
"""

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

__all__ = ["calibration", "evaluation", "laplace"]
__version__ = importlib.metadata.version("laplax")

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .api import calibration, evaluation, laplace


def __getattr__(name: str):  # pragma: no cover - trivial passthrough
    if name in {"calibration", "evaluation", "laplace"}:
        _api = importlib.import_module("laplax.api")  # noqa: RUF052
        return getattr(_api, name)
    msg = f"module 'laplax' has no attribute {name!r}"
    raise AttributeError(msg)
