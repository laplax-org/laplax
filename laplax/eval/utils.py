# utils.py

"""Pushforward utilities for evaluating probabilistic predictions on datasets.

This module provides utilities for evaluating probabilistic models on datasets and
managing metric computations.

Key features include:

- Wrapping functions to store outputs in a structured format.
- Finalizing multiple functions and collecting results in a dictionary.
- Applying prediction functions across datasets to generate predictions and evaluating
  them against their targets.
- Computing and transforming evaluation metrics for datasets using custom or default
  metrics.

These utilities streamline dataset evaluation workflows and ensure flexibility in metric
computation and result aggregation.
"""

from collections.abc import Iterator

import jax
from loguru import logger

from laplax.types import Any, Array, Callable, Data, InputArray, Kwargs
from laplax.util.utils import identity


def finalize_fns(
    fns: list[Callable],
    results: dict,  # Typing must allow empty dict for initializations
    aux: dict[str, Any] | None = None,
    low_rank: bool = False,
    **kwargs: Kwargs,
) -> dict:
    """Execute a set of functions and store their results in a dictionary.

    This function iterates over a list of functions, executes each
    function with the provided keyword arguments, and updates the `results`
    dictionary with their outputs. The functions know what key they should update the
    results dict with.

    Args:
        fns: A list of callables to execute.
        results: A dictionary to store the outputs of the functions.
        aux: Auxiliary data passed to the functions.
        **kwargs: Additional arguments passed to each function.

    Returns:
        The updated `results` dictionary containing the outputs of all
            executed functions.
    """
    for func in fns:
        results, aux = func(results=results, aux=aux, low_rank=low_rank, **kwargs)
    return results


def evaluate_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]],
    data: Data,
    **kwargs: Kwargs,
) -> dict:
    """Evaluate a prediction function on a dataset.

    This function applies a probabilistic predictive function (`pred_fn`) to
    each data point in the dataset, combining the predictions with the target
    labels.

    Args:
        pred_fn: A callable that takes an input array and returns predictions
            as a dictionary.
        data: A dataset, where each data point is a dictionary containing
            "input" and "target".
        **kwargs: Additional arguments, including:

            - `evaluate_on_dataset_batch_size`: Batch size for processing data
              (default: `data_batch_size`).

    Returns:
        A dictionary containing predictions and target labels for the entire dataset.
    """

    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        return {**pred_fn(dp["input"]), "target": dp["target"]}

    return jax.lax.map(
        evaluate_data_point,
        data,
        batch_size=kwargs.get(
            "evaluate_on_dataset_batch_size", kwargs.get("data_batch_size")
        ),
    )


def apply_fns(
    *funcs: Callable,
    names: list[str] | None = None,
    field: str = "results",
    **kwargs: Kwargs,
) -> Callable:
    """Apply multiple functions and store their results in a dictionary.

    This function takes a sequence of functions, applies them to the provided inputs,
    and stores their results in either the 'results' or 'aux' dictionary under
    specified names. This function is useful for applying multiple metrics to the
    results of a pushforward function.

    Args:
        *funcs: Variable number of callable functions to be applied.
        names: Optional list of names for the functions' results. If None,
            function names will be used.
        field: String indicating where to store results, either 'results' or 'aux'
            (default: 'results').
        **kwargs: Mapping of argument names to keys in results/aux dictionaries
            that will be passed to the functions.

    Returns:
        A function that takes 'results' and 'aux' dictionaries along with
            additional kwargs, applies the functions, and returns the updated
            dictionaries.

    Raises:
        TypeError: If any of the provided functions is not callable.
    """
    # Validate all funcs are callable
    for i, func in enumerate(funcs):
        if not callable(func):
            msg = f"Argument {i} is not callable. Type is {type(func)}"
            raise TypeError(msg)

    def apply(results, aux, **local_kwargs):
        # Create key-value pair for functions
        key_value_pairs = {}
        if kwargs:
            for k, v in kwargs.items():
                if v in results:
                    key_value_pairs[k] = results[v]
                elif v in aux:
                    key_value_pairs[k] = aux[v]
                else:
                    msg = f"Key {k} not found in results or aux."
                    raise ValueError(msg)
        else:
            logger.warning("No kwargs provided, using aux dictionary as input")
            key_value_pairs = aux

        # Ensure we have names for all functions
        if names is None:
            # Store under the function name
            func_names = [func.__name__ for func in funcs]
        else:
            if len(names) != len(funcs):
                msg = (
                    f"Number of names ({len(names)}) does not match number "
                    f"of functions ({len(funcs)})"
                )
                raise ValueError(msg)
            func_names = names

        # Apply each function and store results
        for func, name in zip(funcs, func_names, strict=True):
            res = func(**key_value_pairs, **local_kwargs)

            if field == "results":
                results[name] = res
            elif field == "aux":
                aux[name] = res
            else:
                msg = f"Field {field} must be either 'results' or 'aux'."
                raise ValueError(msg)

        return results, aux

    return apply


def transfer_entry(
    mapping: dict[str, str] | list[str],
    field: str = "results",
    access_from: str = "aux",
) -> Callable:
    """Transfer entries between results and auxiliary dictionaries.

    This function creates a callable that copies values between the results and
    auxiliary dictionaries based on the provided mapping.

    Args:
        mapping: Either a dictionary mapping destination keys to source keys,
            or a list of keys to copy with the same names.
        field: String indicating where to store entries, either 'results' or 'aux'
            (default: 'results').
        access_from: String indicating which dictionary to read from, either
            'results' or 'aux' (default: 'aux').

    Returns:
        A function that takes 'results' and 'aux' dictionaries,
            transfers the specified entries, and returns the updated dictionaries.

    Raises:
        ValueError: If field is not 'results' or 'aux'.
    """
    # Convert list to dict if needed
    if isinstance(mapping, list):
        mapping = {k: k for k in mapping}

    # Check if field and access_from are valid
    dict_options = ("results", "aux")
    if field not in dict_options or access_from not in dict_options:
        msg = f"Field {field} must be either 'results' or 'aux'."
        raise ValueError(msg)

    # Transfer the entry
    def transfer(results, aux, **kwargs):
        del kwargs
        options = {"results": results, "aux": aux}
        for k, v in mapping.items():
            options[field][k] = options[access_from][v]
        return options["results"], options["aux"]

    return transfer


def evaluate_metrics_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]],
    data: Data,
    *,
    metrics: list | None = None,
    metrics_dict: dict[str, Callable] | None = None,
    reduce: Callable = identity,
    **kwargs: Kwargs,
) -> dict:
    """Evaluate a set of metrics on a dataset.

    This function computes specified metrics for predictions generated by a
    probabilistic predictive function (`pred_fn`) over a dataset. The results
    can optionally be transformed using an `apply` function.

    Args:
        pred_fn: A callable that takes an input array and returns predictions
            as a dictionary.
        data: A dataset, where each data point is a dictionary containing
            "input" and "target".
        metrics: A list of metrics to compute, this should use the `apply_fns`
            function to apply the metrics and `transfer_entry` function to transfer
            entries between results and auxiliary dictionaries.
        metrics_dict: A dictionary of metrics to compute, where keys are metric
            names and values are callables.
        reduce: A callable to transform the evaluated metrics (default: identity).
        **kwargs: Additional arguments, including:

            - `evaluate_metrics_on_dataset_batch_size`: Batch size for processing data
              (default: `data_batch_size`).

    Returns:
        A dictionary containing the evaluated metrics for the entire dataset.

    Raises:
        ValueError: When metrics and metrics_dict are both None.
    """
    # Initialize metrics list from metric_dict if provided
    metrics_from_dict = []
    if metrics_dict is not None:
        metrics_from_dict = [
            apply_fns(*metrics_dict.values(), names=list(metrics_dict.keys()))
        ]

    # Initialize final metrics list
    if metrics is None and metrics_dict is None:
        msg = "Either metrics or metric_dict must be provided."
        raise ValueError(msg)
    if metrics is None:
        metrics = metrics_from_dict
    elif metrics_dict is not None:
        metrics.extend(metrics_from_dict)

    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        pred = {**pred_fn(dp["input"]), "target": dp["target"]}
        return finalize_fns(fns=metrics, results={}, aux=pred, **kwargs)

    # Evaluate metrics
    evaluated_metrics = jax.lax.map(
        evaluate_data_point,
        data,
        batch_size=kwargs.get(
            "evaluate_metrics_on_dataset_batch_size", kwargs.get("data_batch_size")
        ),
    )
    return {metric: reduce(evaluated_metrics[metric]) for metric in evaluated_metrics}


def evaluate_metrics_on_generator(
    pred_fn: Callable[[InputArray], dict[str, Array]],
    data_generator: Iterator[Data],
    *,
    metrics: list | None = None,
    metrics_dict: dict[str, Callable] | None = None,
    transform: Callable = identity,
    reduce: Callable = identity,
    vmap_over_data: bool = True,
    low_rank: bool = False,
    **kwargs: Kwargs,
) -> dict:
    """Evaluate a set of metrics on a data generator.

    Similar to evaluate_metrics_on_dataset, but works with a generator of data points
    instead of a dataset array. This is useful for cases where the data doesn't fit
    in memory or is being streamed.

    Args:
        pred_fn: A callable that takes an input array and returns predictions
            as a dictionary.
        data_generator: An iterator yielding data points, where each data point
            is a dictionary containing "input" and "target".
        metrics: A list of metrics to compute, this should use the `apply_fns`
            function to apply the metrics and `transfer_entry` function to transfer
            entries between results and auxiliary dictionaries.
        metrics_dict: A dictionary of metrics to compute, where keys are metric
            names and values are callables.
        transform: The transform over individual data points.
        reduce: A callable to transform the evaluated metrics (default: identity).
        vmap_over_data: Data batches from generator have unaccounted batch dimension
            (default: True).
        low_rank: (default: False).
        **kwargs: Additional keyword arguments passed to the metrics functions.

    Returns:
        A dictionary containing the evaluated metrics for all data points.

    Raises:
        ValueError: If neither metrics nor metric_dict is provided.
    """
    # Initialize metrics list from metric_dict if provided
    metrics_from_dict = []
    if metrics_dict is not None:
        metrics_from_dict = [
            apply_fns(*metrics_dict.values(), names=list(metrics_dict.keys()))
        ]

    # Initialize final metrics list
    if metrics is None and metrics_dict is None:
        msg = "Either metrics or metric_dict must be provided."
        raise ValueError(msg)
    if metrics is None:
        metrics = metrics_from_dict
    elif metrics_dict is not None:
        metrics.extend(metrics_from_dict)

    def evaluate_data(dp: Data) -> dict[str, Array]:
        pred = {**pred_fn(dp["input"]), "target": dp["target"]}
        return finalize_fns(
            fns=metrics, results={}, aux=pred, low_rank=low_rank, **kwargs
        )

    # Vmap over batch dimension, if necessary.
    if vmap_over_data:
        evaluate_data = jax.vmap(evaluate_data)
    if not kwargs.get("debug"):
        evaluate_data = jax.jit(evaluate_data)

    # Evaluate metrics by iterating over the generator
    all_results = [evaluate_data(transform(dp)) for dp in data_generator]

    # Combine and reduce results
    if not all_results:
        return {}

    def flatten_dict(d: dict, sep: str = "_") -> dict[str, any]:
        items: dict[str, any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                items.update(flatten_dict(v, sep=sep))
            else:
                items[k] = v
        return items

    # Flatten the results if they are nested dictionaries
    all_results = [flatten_dict(result) for result in all_results]

    # Get all metric names from the first result
    metric_names = all_results[0].keys()

    # Collect and reduce metrics
    return {
        metric: reduce([result[metric] for result in all_results])
        for metric in metric_names
    }
