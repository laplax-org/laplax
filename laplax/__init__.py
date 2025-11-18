"""Package for Laplace approximations in JAX.

Avoid importing heavy submodules at package import time to keep import side
effects minimal (e.g., JAX PRNG initialization). Expose top-level symbols via
lazy attribute access.
"""

import importlib.metadata

__all__ = ["calibration", "evaluation", "laplace"]
__version__ = importlib.metadata.version("laplax")


def __getattr__(name: str):  # pragma: no cover - trivial passthrough
    if name in {"calibration", "evaluation", "laplace"}:
        from . import api as _api

        return getattr(_api, name)
    msg = f"module 'laplax' has no attribute {name!r}"
    raise AttributeError(msg)
