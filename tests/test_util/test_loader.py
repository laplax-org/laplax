"""Test the loader utilities."""

from itertools import starmap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.loader import (
    execute_with_data_loader,
    reduce_add,
    reduce_concat,
    reduce_online_mean,
    reduce_sum,
    wrap_function_with_data_loader,
)


def dummy_transform(x):
    """Simple transform that returns the data as is.

    Returns:
        The input.
    """
    return x


def sum_function(data, param):
    return data + param * jnp.ones_like(data)


@pytest.fixture
def seed():
    return 0


@pytest.fixture
def sample_iterable(seed):
    """Create sample data: list of arrays.

    Returns:
        The list of arrays.
    """
    shapes = [(5, 4, 2), (5, 4, 2), (5, 4, 2)]
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    return list(starmap(jax.random.normal, zip(keys, shapes, strict=True)))


@pytest.fixture
def full_data(sample_iterable):
    """Create concatenated version of the sample data.

    Returns:
        The concatenated data.
    """
    return jnp.concatenate(sample_iterable, axis=0)


@pytest.mark.parametrize("param", [2.7])
@pytest.mark.parametrize(
    ("reduce_fn", "expected_transform"),
    [
        (reduce_add, lambda x, p: sum_function(x, p).reshape(3, 5, 4, 2).sum(axis=0)),
        (reduce_sum, lambda x, p: jnp.sum(sum_function(x, p), axis=0, keepdims=True)),
        (reduce_concat, sum_function),
        (reduce_online_mean, lambda x, p: jnp.mean(sum_function(x, p), axis=0)),
    ],
    ids=["reduce_add", "reduce_sum", "reduce_concat", "reduce_online_mean"],
)
def test_reduce_functions_with_wrap_function(
    sample_iterable, full_data, reduce_fn, expected_transform, param
):
    """Test different reduce functions using both wrap_function."""
    # Test wrap_function_with_data_loader
    wrapped_fn = wrap_function_with_data_loader(
        sum_function, sample_iterable, transform=dummy_transform, reduce=reduce_fn
    )
    result_wrapped = wrapped_fn(param=param)
    expected = expected_transform(full_data, param)
    np.testing.assert_allclose(result_wrapped, expected, atol=5e-6)

    # Test execute_with_data_loader
    result_execute = execute_with_data_loader(
        sum_function,
        sample_iterable,
        transform=dummy_transform,
        reduce=reduce_fn,
        param=param,
    )

    np.testing.assert_allclose(result_execute, expected, rtol=1e-6, atol=5e-6)
