"""Contains operations for flexible/adaptive compute."""

import operator

import jax

from laplax.types import Callable, Iterable, Kwargs

# -------------------------------------------------------------------------
# Default values
# -------------------------------------------------------------------------

DEFAULT_PARALLELISM = None
DEFAULT_DTYPE = "float32"
DEFAULT_PRECOMPUTE_LIST = True

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    """Convert a string representation of a boolean to a boolean value.

    Args:
        value: A string representation of a boolean ("True" or "False").

    Returns:
        The corresponding boolean value.

    Raises:
        ValueError: If the string does not represent a valid boolean value.
    """
    valid_values = {"True": True, "False": False}
    if value not in valid_values:
        msg = "invalid string representation of a boolean value"
        raise ValueError(msg)
    return valid_values[value]


# -------------------------------------------------------------------------
# Adaptive operations
# -------------------------------------------------------------------------


def precompute_list(
    func: Callable, items: Iterable, precompute: bool | None = None, **kwargs: Kwargs
) -> Callable:
    """Precompute results for a list of items or return the original function.

    If `option` is enabled, this function applies `func` to all items in `items`
    and stores the results for later retrieval. Otherwise, it returns `func` as-is.

    Args:
        func: The function to apply to each item in the list.
        items: An iterable of items to process.
        precompute: Determines whether to precompute results:
            - None: Use the default precompute setting.
            - bool: Specify directly whether to precompute.
        **kwargs: Additional keyword arguments, including:
            - precompute_list_batch_size: Batch size for precomputing results.

    Returns:
        A function to retrieve precomputed elements by index, or the original
            `func` if precomputation is disabled.
    """
    if precompute is None:
        precompute = DEFAULT_PRECOMPUTE_LIST

    if precompute:
        precomputed = jax.lax.map(
            func, items, batch_size=kwargs.get("precompute_list_batch_size")
        )

        def get_element(index: int):
            return jax.tree.map(operator.itemgetter(index), precomputed)

        return get_element

    return func
