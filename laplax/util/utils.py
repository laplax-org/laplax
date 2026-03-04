"""General utility functions."""

from laplax.types import Any


def identity(x: Any) -> Any:
    """Identity function.

    Args:
        x: Any input.

    Returns:
        The input itself.
    """
    return x


def input_target_split(batch: Any) -> tuple[Any, Any]:
    """Split batch into input and target.

    Args:
        batch: A tuple or list containing (input, target).

    Returns:
        Tuple of (input, target).
    """
    return batch[0], batch[1]
