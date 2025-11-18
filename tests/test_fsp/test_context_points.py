import jax
import jax.numpy as jnp
import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from laplax.util.context_points import (
    _flatten_spatial_dims,  # noqa: PLC2701
    _generate_low_discrepancy_sequence,  # noqa: PLC2701
    _halton_context_points,  # noqa: PLC2701
    _latin_hypercube_context_points,  # noqa: PLC2701
    _load_all_data_from_dataloader,  # noqa: PLC2701
    _make_grid_from_data_shape,  # noqa: PLC2701
    _normalize_to_unit_cube,  # noqa: PLC2701
    _pca_context_points,  # noqa: PLC2701
    _pca_transform_jax,  # noqa: PLC2701
    _random_context_points,  # noqa: PLC2701
    _sobol_context_points,  # noqa: PLC2701
    select_context_points,
)


class TensorDataset(Dataset):
    """A simple Dataset wrapping tensors."""

    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


class DummyWaveDataset(Dataset):
    """Minimal dummy dataset mimicking wave/heat trajectories.

    Generates synthetic spatiotemporal data in 1D or 2D with an explicit
    time axis and a channel axis, keeping shapes simple and deterministic.

    X shape per-sample:
    - 1D: (S, T, Cx)
    - 2D: (Sx, Sy, T, Cx)

    Y shape per-sample (single-channel target matching spatial/time layout):
    - 1D: (S, T, 1)
    - 2D: (Sx, Sy, T, 1)
    """

    def __init__(
        self,
        n_samples: int = 100,
        spatial_size: int | tuple[int, int] = 32,
        n_timesteps: int = 10,
        channels_in: int = 2,
        is_2d: bool | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Determine 1D vs 2D and spatial shape
        if isinstance(spatial_size, tuple):
            sx, sy = spatial_size
            self.is_2d = True
            spatial_shape = (int(sx), int(sy))
        else:
            self.is_2d = bool(is_2d) if is_2d is not None else False
            spatial_shape = (
                (int(spatial_size),)
                if not self.is_2d
                else (int(spatial_size), int(spatial_size))
            )

        t = int(max(1, n_timesteps))
        c_in = int(max(1, channels_in))
        c_out = 1

        # Build full sample shapes
        if self.is_2d:
            self.x = rng.standard_normal((
                n_samples,
                spatial_shape[0],
                spatial_shape[1],
                t,
                c_in,
            )).astype(np.float32)
            # Simple target derived from inputs to keep relation but avoid heavy logic
            self.y = (
                self.x[..., :1]
                + 0.1
                * rng.standard_normal((
                    n_samples,
                    spatial_shape[0],
                    spatial_shape[1],
                    t,
                    c_out,
                ))
            ).astype(np.float32)
        else:
            self.x = rng.standard_normal((n_samples, spatial_shape[0], t, c_in)).astype(
                np.float32
            )
            self.y = (
                self.x[..., :1]
                + 0.1 * rng.standard_normal((n_samples, spatial_shape[0], t, c_out))
            ).astype(np.float32)

        self.n_samples = int(n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _create_dummy_wave_dataloader(
    n_samples: int = 100,
    spatial_size: int | tuple[int, int] = 32,
    n_timesteps: int = 10,
    channels_in: int = 2,
    is_2d: bool | None = None,
    batch_size: int = 16,
    seed: int = 42,
) -> DataLoader:
    dataset = DummyWaveDataset(
        n_samples=n_samples,
        spatial_size=spatial_size,
        n_timesteps=n_timesteps,
        channels_in=channels_in,
        is_2d=is_2d,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _create_dummy_dataloader(
    n_samples: int = 100,
    input_shape: tuple = (32,),
    output_shape: tuple = (10,),
    batch_size: int = 16,
) -> DataLoader:
    """Create a dummy DataLoader for testing.

    Returns:
        DataLoader: A DataLoader yielding random input/target pairs.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_samples, *input_shape)).astype(np.float32)
    y = rng.standard_normal((n_samples, *output_shape)).astype(np.float32)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_load_all_data_from_dataloader():
    """Test loading all data from a DataLoader."""
    dataloader = _create_dummy_wave_dataloader(
        n_samples=50, spatial_size=32, n_timesteps=10, channels_in=2, batch_size=10
    )
    all_x, all_y = _load_all_data_from_dataloader(dataloader)

    assert all_x.shape == (50, 32, 10, 2)
    assert all_y.shape == (50, 32, 10, 1)
    assert isinstance(all_x, jax.Array)
    assert isinstance(all_y, jax.Array)


def test_flatten_spatial_dims():
    """Test flattening of spatial dimensions."""
    data = jnp.ones((4, 8, 8, 3))
    flattened, original_shape = _flatten_spatial_dims(data)

    assert flattened.shape == (4, 64)
    assert original_shape == (4, 8, 8, 3)


def test_flatten_spatial_dims_1d():
    """Test flattening with 1D spatial data."""
    data = jnp.ones((10, 32, 2))
    flattened, original_shape = _flatten_spatial_dims(data)

    assert flattened.shape == (10, 32)
    assert original_shape == (10, 32, 2)


def test_pca_transform():
    """Test PCA transformation."""
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (100, 50))

    transformed, pca = _pca_transform_jax(data, n_components=10)

    assert transformed.shape == (100, 10)
    assert pca.n_components_ == 10


def test_pca_transform_variance_threshold():
    """Test PCA transformation with variance threshold."""
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (100, 50))

    transformed, pca = _pca_transform_jax(data, variance_threshold=0.95)

    assert transformed.shape[0] == 100
    assert pca.n_components_ <= 50
    assert np.sum(pca.explained_variance_ratio_) >= 0.95


def test_generate_low_discrepancy_sobol():
    """Test Sobol sequence generation."""
    points = _generate_low_discrepancy_sequence(
        n_dims=5, n_points=32, sequence_type="sobol", seed=42
    )

    assert points.shape == (32, 5)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)


def test_generate_low_discrepancy_halton():
    """Test Halton sequence generation."""
    points = _generate_low_discrepancy_sequence(
        n_dims=5, n_points=50, sequence_type="halton", seed=42
    )

    assert points.shape == (50, 5)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)


def test_generate_low_discrepancy_latin_hypercube():
    """Test Latin Hypercube sequence generation."""
    points = _generate_low_discrepancy_sequence(
        n_dims=5, n_points=50, sequence_type="latin_hypercube", seed=42
    )

    assert points.shape == (50, 5)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)


def test_generate_low_discrepancy_invalid_type():
    """Test that invalid sequence type raises error."""
    with pytest.raises(ValueError, match="Unknown sequence type"):
        _generate_low_discrepancy_sequence(
            n_dims=5, n_points=50, sequence_type="invalid"
        )


def test_normalize_to_unit_cube():
    """Test normalization to unit hypercube."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    normalized, data_min, data_max = _normalize_to_unit_cube(data)

    assert normalized.shape == data.shape
    np.testing.assert_array_equal(data_min, [1.0, 2.0])
    np.testing.assert_array_equal(data_max, [5.0, 6.0])
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


def test_random_context_points():
    """Test random context point selection."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y = _random_context_points(dataloader, n_context, seed=42)

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_random_context_points_reproducibility():
    """Test that random selection is reproducible with same seed."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    cx1, cy1 = _random_context_points(dataloader, n_context, seed=42)
    cx2, cy2 = _random_context_points(dataloader, n_context, seed=42)

    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_random_context_points_all_data():
    """Test random selection when requesting all or more points."""
    dataloader = _create_dummy_dataloader(n_samples=50)

    context_x, context_y = _random_context_points(
        dataloader, n_context_points=60, seed=42
    )

    assert context_x.shape == (50, 32)
    assert context_y.shape == (50, 10)


def test_sobol_context_points():
    """Test Sobol-based context point selection."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 16

    context_x, context_y = _sobol_context_points(
        dataloader, n_context, n_pca_components=5, seed=42
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_halton_context_points():
    """Test Halton-based context point selection."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y = _halton_context_points(
        dataloader, n_context, n_pca_components=5, seed=42
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_latin_hypercube_context_points():
    """Test Latin Hypercube-based context point selection."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y = _latin_hypercube_context_points(
        dataloader, n_context, n_pca_components=5, seed=42
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_pca_context_points_reproducibility():
    """Test that PCA-based selection is reproducible with same seed."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 16

    cx1, cy1 = _pca_context_points(
        dataloader, n_context, sequence_type="sobol", n_pca_components=5, seed=42
    )
    cx2, cy2 = _pca_context_points(
        dataloader, n_context, sequence_type="sobol", n_pca_components=5, seed=42
    )

    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_pca_context_points_return_pca():
    """Test that PCA model can be returned."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 16

    result = _pca_context_points(
        dataloader, n_context, n_pca_components=5, seed=42, return_pca=True
    )

    assert len(result) == 3
    context_x, context_y, pca = result
    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)
    assert pca.n_components_ == 5


def test_make_grid_from_data_shape_1d():
    """Test 1D grid generation."""
    data_shape = (10, 32, 2)
    grid, dx = _make_grid_from_data_shape(data_shape)

    assert grid.shape == (32,)
    assert dx > 0


def test_make_grid_from_data_shape_2d():
    """Test 2D grid generation."""
    data_shape = (10, 16, 16, 3)
    grid, dx = _make_grid_from_data_shape(data_shape)

    assert grid.shape == (16, 16, 2)
    assert dx > 0


def test_make_grid_from_data_shape_3d():
    """Test 3D grid generation."""
    data_shape = (10, 8, 8, 8, 4)
    grid, dx = _make_grid_from_data_shape(data_shape)

    assert grid.shape == (8, 8, 8, 3)
    assert dx > 0


def test_select_context_points_random():
    """Test select_context_points with random strategy."""
    dataloader = _create_dummy_dataloader(n_samples=100, input_shape=(32,))
    n_context = 20

    context_x, context_y, grid = select_context_points(
        dataloader, context_selection="random", n_context_points=n_context, seed=42
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)
    assert grid is not None


def test_select_context_points_sobol():
    """Test select_context_points with Sobol strategy."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 16

    context_x, context_y, _grid = select_context_points(
        dataloader,
        context_selection="sobol",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_select_context_points_halton():
    """Test select_context_points with Halton strategy."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y, _grid = select_context_points(
        dataloader,
        context_selection="halton",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_select_context_points_latin_hypercube():
    """Test select_context_points with Latin Hypercube strategy."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y, _grid = select_context_points(
        dataloader,
        context_selection="latin_hypercube",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_select_context_points_combined_strategy():
    """Test select_context_points with combined strategy."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    context_x, context_y, _grid = select_context_points(
        dataloader,
        context_selection="random+sobol",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_select_context_points_combined_three_strategies():
    """Test select_context_points with three combined strategies."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 30

    context_x, context_y, _grid = select_context_points(
        dataloader,
        context_selection="random+sobol+halton",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)
    assert context_y.shape == (n_context, 10)


def test_select_context_points_combined_uneven_split():
    """Test that combined strategy handles uneven splits correctly."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 25

    context_x, _context_y, _grid = select_context_points(
        dataloader,
        context_selection="random+sobol",
        n_context_points=n_context,
        seed=42,
    )

    assert context_x.shape == (n_context, 32)


def test_select_context_points_reproducibility_across_strategies():
    """Test that different strategies produce different results but are reproducible."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    cx_random, _cy_random, _ = select_context_points(
        dataloader, context_selection="random", n_context_points=n_context, seed=42
    )

    cx_sobol, _cy_sobol, _ = select_context_points(
        dataloader,
        context_selection="sobol",
        n_context_points=n_context,
        n_pca_components=5,
        seed=42,
    )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(cx_random, cx_sobol)


def test_select_context_points_invalid_strategy():
    """Test that invalid strategy raises error."""
    dataloader = _create_dummy_dataloader(n_samples=100)

    with pytest.raises(ValueError, match="Unknown context_selection"):
        select_context_points(
            dataloader, context_selection="invalid_strategy", n_context_points=20
        )


def test_select_context_points_with_grid_stride_1d():
    """Test grid striding with 1D data."""
    dataloader = _create_dummy_dataloader(n_samples=100, input_shape=(64, 2))
    n_context = 20

    context_x, _context_y, grid = select_context_points(
        dataloader,
        context_selection="random",
        n_context_points=n_context,
        grid_stride=2,
        seed=42,
    )

    assert context_x.shape[1] == 32
    assert grid.shape == (32,)


def test_select_context_points_with_time_keep():
    """Test time_keep parameter."""
    dataloader = _create_dummy_dataloader(n_samples=100, input_shape=(32, 10, 2))
    n_context = 20

    context_x, _context_y, _grid = select_context_points(
        dataloader,
        context_selection="random",
        n_context_points=n_context,
        time_keep=5,
        seed=42,
    )

    assert context_x.shape[-2] == 5


def test_pca_aliases():
    """Test that PCA aliases work correctly."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 16

    cx1, cy1, _ = select_context_points(
        dataloader, context_selection="pca", n_context_points=n_context, seed=42
    )

    cx2, cy2, _ = select_context_points(
        dataloader,
        context_selection="pca_sobol",
        n_context_points=n_context,
        seed=42,
    )

    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_context_y_values_match_input_data():
    """Test that context_y values actually come from the dataset."""
    dataloader = _create_dummy_dataloader(n_samples=50, batch_size=50)
    _all_x, all_y = _load_all_data_from_dataloader(dataloader)

    n_context = 10
    _context_x, context_y, _ = select_context_points(
        dataloader, context_selection="random", n_context_points=n_context, seed=42
    )

    for cy in context_y:
        found = False
        for ay in all_y:
            if np.allclose(cy, ay):
                found = True
                break
        assert found, "context_y contains values not in original dataset"


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    dataloader = _create_dummy_dataloader(n_samples=100)
    n_context = 20

    cx1, _cy1, _ = select_context_points(
        dataloader, context_selection="random", n_context_points=n_context, seed=42
    )

    cx2, _cy2, _ = select_context_points(
        dataloader, context_selection="random", n_context_points=n_context, seed=123
    )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(cx1, cx2)
