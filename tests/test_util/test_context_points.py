# /tests/test_util/test_context_points.py

"""Tests for context point selection utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.context_points import (
    _generate_low_discrepancy_sequence,
    _sample_uniform_like,
    select_context_points,
)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def sample_data(seed):
    """Create sample data batch.

    Returns:
        Dictionary with 'input' and 'target' arrays.
    """
    key = jax.random.PRNGKey(seed)
    batch_size = 10
    input_shape = (batch_size, 5, 3)
    target_shape = (batch_size, 2)

    key_x, key_y = jax.random.split(key)
    inputs = jax.random.uniform(key_x, input_shape, minval=0.0, maxval=2 * jnp.pi)
    targets = jax.random.uniform(key_y, target_shape, minval=-1.0, maxval=1.0)

    return {"input": inputs, "target": targets}


@pytest.fixture
def sample_dataloader(sample_data):
    """Create a simple dataloader from sample data.

    Returns:
        SimpleLoader instance.
    """

    class SimpleLoader:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            yield self.data

    return SimpleLoader(sample_data)


@pytest.mark.parametrize(
    "sequence_type",
    ["sobol", "halton", "latin_hypercube"],
)
def test_generate_low_discrepancy_sequence(sequence_type, seed):
    """Test low-discrepancy sequence generation."""
    n_dims = 3
    n_points = 10

    points = _generate_low_discrepancy_sequence(
        n_dims=n_dims,
        n_points=n_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    assert points.shape == (n_points, n_dims)
    assert jnp.all(points >= 0.0)
    assert jnp.all(points <= 1.0)
    assert not jnp.any(jnp.isnan(points))
    assert not jnp.any(jnp.isinf(points))


def test_generate_low_discrepancy_sequence_invalid_type():
    """Test that invalid sequence type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sequence type"):
        _generate_low_discrepancy_sequence(
            n_dims=2,
            n_points=5,
            sequence_type="invalid",
            seed=42,
        )


def test_sample_uniform_like(sample_data, seed):
    """Test uniform sampling in data bounding box."""
    data = sample_data["input"]
    n_points = 5
    key = jax.random.PRNGKey(seed)

    samples = _sample_uniform_like(data, n_points, key)

    assert samples.shape == (n_points, *data.shape[1:])
    assert not jnp.any(jnp.isnan(samples))
    assert not jnp.any(jnp.isinf(samples))

    data_flat = data.reshape(data.shape[0], -1)
    samples_flat = samples.reshape(n_points, -1)

    data_min = jnp.min(data_flat, axis=0)
    data_max = jnp.max(data_flat, axis=0)

    assert jnp.all(samples_flat >= data_min - 1e-6)
    assert jnp.all(samples_flat <= data_max + 1e-6)


@pytest.mark.parametrize(
    "method",
    ["random", "sobol", "halton", "latin_hypercube", "pca"],
)
def test_select_context_points_batch(method, sample_data, seed):
    """Test context point selection from a single batch."""
    n_context_points = 5

    context_x, context_y = select_context_points(
        data=sample_data,
        method=method,
        n_context_points=n_context_points,
        seed=seed,
    )

    assert context_x.shape[0] == n_context_points
    assert context_x.shape[1:] == sample_data["input"].shape[1:]
    assert context_y.shape[0] == n_context_points
    assert context_y.shape[1:] == sample_data["target"].shape[1:]

    assert not jnp.any(jnp.isnan(context_x))
    assert not jnp.any(jnp.isnan(context_y))
    assert not jnp.any(jnp.isinf(context_x))
    assert not jnp.any(jnp.isinf(context_y))


@pytest.mark.parametrize(
    "method",
    ["random", "sobol", "halton", "latin_hypercube", "pca"],
)
def test_select_context_points_dataloader(method, sample_dataloader, seed):
    """Test context point selection from a dataloader."""
    n_context_points = 5

    context_x, context_y = select_context_points(
        data=sample_dataloader,
        method=method,
        n_context_points=n_context_points,
        seed=seed,
    )

    assert context_x.shape[0] == n_context_points
    assert context_y.shape[0] == n_context_points

    assert not jnp.any(jnp.isnan(context_x))
    assert not jnp.any(jnp.isnan(context_y))
    assert not jnp.any(jnp.isinf(context_x))
    assert not jnp.any(jnp.isinf(context_y))


def test_select_context_points_invalid_method(sample_data):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown context selection method"):
        select_context_points(
            data=sample_data,
            method="invalid_method",
            n_context_points=5,
            seed=42,
        )


def test_select_context_points_reproducibility(sample_data, seed):
    """Test that same seed produces same results."""
    n_context_points = 5

    context_x1, context_y1 = select_context_points(
        data=sample_data,
        method="random",
        n_context_points=n_context_points,
        seed=seed,
    )

    context_x2, context_y2 = select_context_points(
        data=sample_data,
        method="random",
        n_context_points=n_context_points,
        seed=seed,
    )

    np.testing.assert_allclose(context_x1, context_x2, rtol=1e-6)
    np.testing.assert_allclose(context_y1, context_y2, rtol=1e-6)


def test_select_context_points_different_methods_produce_different_results(
    sample_data, seed
):
    """Test that different methods produce different results."""
    n_context_points = 5

    context_random = select_context_points(
        data=sample_data,
        method="random",
        n_context_points=n_context_points,
        seed=seed,
    )[0]

    context_sobol = select_context_points(
        data=sample_data,
        method="sobol",
        n_context_points=n_context_points,
        seed=seed,
    )[0]

    assert not jnp.allclose(context_random, context_sobol, rtol=1e-3)


@pytest.fixture
def pde_2d_image_data(seed):
    """Create 2D PDE-like image data (wave pattern).

    Returns:
        Dictionary with 'input' and 'target' arrays.
    """
    key = jax.random.PRNGKey(seed)
    batch_size = 4
    height, width = 32, 32

    x = jnp.linspace(0, 2 * jnp.pi, width)
    y = jnp.linspace(0, 2 * jnp.pi, height)
    X, Y = jnp.meshgrid(x, y)

    images = []
    for i in range(batch_size):
        freq = 1.0 + 0.5 * i
        phase = jax.random.uniform(key, shape=(), minval=0, maxval=2 * jnp.pi)
        image = jnp.sin(freq * X + phase) * jnp.cos(freq * Y + phase)
        images.append(image)

    inputs = jnp.stack(images, axis=0)
    inputs = inputs[..., jnp.newaxis]

    targets = jnp.ones((batch_size, height, width, 1))

    return {"input": inputs, "target": targets}
