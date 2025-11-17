from collections.abc import Callable
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from laplax.types import Array, Int


def _load_all_data_from_dataloader(
    dataloader: DataLoader,
) -> tuple[Array, Array]:
    """Load all batches from a DataLoader into JAX arrays."""
    x_list = []
    y_list = []

    for batch_x, batch_y in dataloader:
        x_list.append(jnp.array(batch_x))
        y_list.append(jnp.array(batch_y))

    all_x = jnp.array(jnp.concatenate(x_list, axis=0))
    all_y = jnp.array(jnp.concatenate(y_list, axis=0))

    return all_x, all_y


def _flatten_spatial_dims(data: Array) -> tuple[Array, tuple]:
    """Flatten all axes except batch and last channel axis.

    Avoids using jax.numpy on Python tuples (which can carry tracers in tests)
    by computing the product with numpy to obtain a plain integer.
    """
    original_shape = data.shape
    batch_size = int(original_shape[0])
    # Product of all spatial-and-temporal dims excluding the last channel dim
    middle = original_shape[1:-1]
    n_spatial = int(np.prod(middle)) if len(middle) > 0 else 1
    flattened = data.reshape(batch_size, n_spatial)

    return flattened, original_shape


def _pca_transform_jax(
    y_data: Array,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[Array, PCA]:
    """Standardize features, then run PCA (SVD-backed) and return scores.

    - Centers and scales each original feature to unit variance prior to PCA.
    - Uses sklearn's PCA which centers again internally; pre-scaling is the key.
    """
    y_np = np.array(y_data)
    # Standardize features to zero mean, unit variance
    feat_mean = y_np.mean(axis=0, keepdims=True)
    feat_std = y_np.std(axis=0, keepdims=True) + 1e-8
    y_np_std = (y_np - feat_mean) / feat_std

    if n_components is None:
        pca = PCA(n_components=variance_threshold, svd_solver="full")
    else:
        pca = PCA(n_components=n_components)

    pca.fit(y_np_std)
    transformed = pca.transform(y_np_std)

    return jax.device_put(transformed), pca


def _generate_low_discrepancy_sequence(
    n_dims: int,
    n_points: int,
    sequence_type: str = "sobol",
    seed: Int | None = None,
) -> np.ndarray:
    """Generate low-discrepancy quasi-random sequences."""
    if sequence_type.lower() == "sobol":
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        # SciPy's Sobol supports base-2 samples; caller should choose n_points accordingly.
        points = sampler.random_base2(n_points)
    elif sequence_type.lower() == "halton":
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=seed)
        points = sampler.random(n_points)
    elif sequence_type.lower() == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        points = sampler.random(n_points)
    else:
        msg = (
            f"Unknown sequence type: {sequence_type}. "
            "Choose from 'sobol', 'halton', 'latin_hypercube'"
        )
        raise ValueError(msg)

    return points


def _normalize_to_unit_cube(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data to the unit hypercube [0, 1]^d."""
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    normalized = (data - data_min) / (data_max - data_min + 1e-10)
    return normalized, data_min, data_max


def _find_nearest_neighbors(
    query_points: np.ndarray,
    data_points: np.ndarray,
) -> np.ndarray:
    """Find nearest neighbors for each query point in data points."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(data_points)

    _, indices = nn.kneighbors(query_points)
    return indices.flatten()


def _pca_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    sequence_type: str = "sobol",
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
    return_pca: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, PCA]:
    """Select context points using PCA-based low-discrepancy sampling."""
    all_x, all_y = _load_all_data_from_dataloader(dataloader)
    y_flat, _ = _flatten_spatial_dims(all_y)
    x_flat, _ = _flatten_spatial_dims(all_x)

    y_pca, pca = _pca_transform_jax(
        y_flat,
        n_components=n_pca_components,
        variance_threshold=pca_variance_threshold,
    )
    y_pca_norm, _, _ = _normalize_to_unit_cube(np.array(y_pca))

    ld_points = _generate_low_discrepancy_sequence(
        n_dims=pca.n_components_,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    if sequence_type.lower() == "sobol":
        centered = 2.0 * (ld_points - 0.5)
        variances = pca.explained_variance_  # shape (n_components,)
        scales = 2.0 * variances
        ld_scaled = centered * scales
        indices = _find_nearest_neighbors(ld_scaled, np.array(y_pca))
    else:
        # Default: operate in normalized [0,1]^d cube
        indices = _find_nearest_neighbors(ld_points, y_pca_norm)

    # Handle duplicates
    unique_indices = np.unique(indices)
    if len(unique_indices) < n_context_points:
        remaining = n_context_points - len(unique_indices)
        available = np.setdiff1d(np.arange(len(all_y)), unique_indices)

        # Check if we have enough available points
        if len(available) < remaining:
            # Use all available points
            indices = np.concatenate([unique_indices, available])
        else:
            rng = np.random.default_rng(seed)
            additional = rng.choice(available, size=remaining, replace=False)
            indices = np.concatenate([unique_indices, additional])
    else:
        indices = unique_indices[:n_context_points]

    # Extract context points
    context_x = all_x[indices]
    context_y = all_y[indices]

    if return_pca:
        return context_x, context_y, pca
    return context_x, context_y


def _sobol_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Sobol-based PCA context point selection."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="sobol",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _latin_hypercube_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Latin Hypercube-based PCA context point selection."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="latin_hypercube",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _halton_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Halton-based PCA context point selection."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="halton",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _random_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Randomly sample context points from the dataset.

    Extra PCA-related arguments are accepted for API compatibility with
    other context selection functions but are ignored.
    """
    del n_pca_components, pca_variance_threshold

    all_x, all_y = _load_all_data_from_dataloader(dataloader)

    effective_seed = seed if seed is not None else 0
    key = jax.random.PRNGKey(effective_seed)
    n_total = len(all_y)

    if n_context_points >= n_total:
        return all_x, all_y

    indices = jax.random.choice(key, n_total, shape=(n_context_points,), replace=False)

    context_x = all_x[indices]
    context_y = all_y[indices]

    return context_x, context_y


ContextSelectionFn = Callable[
    [DataLoader, Int, Int | None, float, Int | None], tuple[Array, Array]
]

CONTEXT_SELECTION_METHODS: dict[str, ContextSelectionFn] = {
    "random": _random_context_points,
    "sobol": _sobol_context_points,
    "pca_sobol": _sobol_context_points,
    "halton": _halton_context_points,
    "pca_halton": _halton_context_points,
    "latin_hypercube": _latin_hypercube_context_points,
    "pca_lhs": _latin_hypercube_context_points,
    # Alias 'pca' to Sobol-based PCA selection
    "pca": _sobol_context_points,
}


def _make_grid_from_data_shape(
    data_shape, min_domain: float = 0.0, max_domain: float = 2 * np.pi
) -> tuple[jnp.ndarray, float]:
    """Construct a 1D/2D/3D spatial grid from a data shape."""
    spatial_dims = tuple(dim for dim in data_shape[1:] if dim > 1)

    if len(spatial_dims) > 3:
        spatial_dims = spatial_dims[1:]

    num_spatial_dims = len(spatial_dims)
    domain_extent = max_domain - min_domain

    if num_spatial_dims == 1:
        num_points = spatial_dims[0]
        dx = domain_extent / num_points
        grid = jnp.linspace(min_domain, max_domain - dx, num_points)

    elif num_spatial_dims == 2:
        num_points_x, num_points_y = spatial_dims
        dx = domain_extent / num_points_x
        dy = domain_extent / num_points_y

        x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
        y = jnp.linspace(min_domain, max_domain - dy, num_points_y)

        X, Y = jnp.meshgrid(x, y, indexing="xy")
        grid = jnp.stack([X, Y], axis=-1)

    elif num_spatial_dims == 3:
        num_points_x, num_points_y, num_points_z = spatial_dims
        dx = domain_extent / num_points_x
        dy = domain_extent / num_points_y
        dz = domain_extent / num_points_z

        x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
        y = jnp.linspace(min_domain, max_domain - dy, num_points_y)
        z = jnp.linspace(min_domain, max_domain - dz, num_points_z)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        grid = jnp.stack([X, Y, Z], axis=-1)

    else:
        msg = f"Unsupported number of spatial dimensions: {num_spatial_dims}"
        raise ValueError(msg)

    return grid, dx


def _make_grid_from_loader(
    dataloader: DataLoader,
    min_domain: float = 0.0,
    max_domain: float = 2 * np.pi,
) -> tuple[jnp.ndarray, float]:
    """Infer grid from a single batch of the dataloader."""
    _, y = next(iter(dataloader))
    data_shape = y.shape
    return _make_grid_from_data_shape(
        data_shape, min_domain=min_domain, max_domain=max_domain
    )


def _apply_grid_stride(
    grid: jnp.ndarray,
    context_x: Array,
    grid_stride: Int | Sequence[int] | None,
) -> tuple[jnp.ndarray, Array]:
    """Apply spatial striding to grid and corresponding context inputs."""
    if grid_stride is None or grid_stride == 1:
        return grid, context_x

    stride = (
        grid_stride
        if isinstance(grid_stride, (tuple, list))
        else (int(grid_stride),)
    )

    # 1D grid: (S,)
    if grid.ndim == 1:
        s = max(1, int(stride[0]))
        grid = grid[::s]
        # context shape (..., S, T, C) with S at axis=1
        context_x = context_x[:, ::s, ...]
    # 2D grid: (Sy, Sx, 2)
    elif grid.ndim == 3:
        s_x = s_y = max(1, int(stride[0]))
        if len(stride) >= 2:
            s_x = max(1, int(stride[0]))
            s_y = max(1, int(stride[1]))
        # Grid axes are (Sy, Sx, 2) due to indexing="xy"
        grid = grid[::s_y, ::s_x, :]
        # Context axes are (n_ctx, Sx, Sy, T, C)
        context_x = context_x[:, ::s_x, ::s_y, ...]
    # 3D grid: (Sx, Sy, Sz, 3)
    elif grid.ndim == 4:
        s_x = s_y = s_z = max(1, int(stride[0]))
        if len(stride) >= 3:
            s_x = max(1, int(stride[0]))
            s_y = max(1, int(stride[1]))
            s_z = max(1, int(stride[2]))
        grid = grid[::s_x, ::s_y, ::s_z, :]
        context_x = context_x[:, ::s_x, ::s_y, ::s_z, ...]
    else:
        msg = f"Unsupported grid dimension for striding: {grid.ndim}"
        raise ValueError(msg)

    return grid, context_x


def select_context_points(
    dataloader: DataLoader,
    context_selection: str,
    n_context_points: Int = 50,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
    time_keep: Int | None = None,
    grid_stride: Int | None = None,
) -> tuple[Array, Array, Array | None]:
    """Top-level context point selection API."""
    # Handle combined strategies (e.g., "random+sobol", "sobol+latin_hypercube")
    if "+" in context_selection:
        strategies = [s.strip() for s in context_selection.split("+")]

        # Divide points among strategies
        points_per_strategy = n_context_points // len(strategies)
        remainder = n_context_points % len(strategies)

        all_context_x: list[Array] = []
        all_context_y: list[Array] = []

        for i, strategy in enumerate(strategies):
            n_points = points_per_strategy + (1 if i < remainder else 0)
            strategy_seed = None if seed is None else seed + i
            if strategy not in CONTEXT_SELECTION_METHODS:
                msg = (
                    f"Unknown context_selection: {strategy}. "
                    "Choose from 'random', 'sobol', 'halton', "
                    "'latin_hypercube', 'pca', 'pca_sobol', "
                    "'pca_halton', 'pca_lhs'"
                )
                raise ValueError(msg)
            cx, cy = CONTEXT_SELECTION_METHODS[strategy](
                dataloader,
                n_points,
                n_pca_components,
                pca_variance_threshold,
                strategy_seed,
            )
            all_context_x.append(cx)
            all_context_y.append(cy)

        context_x = jnp.concatenate(all_context_x, axis=0)
        context_y = jnp.concatenate(all_context_y, axis=0)

        grid, _ = _make_grid_from_loader(dataloader)
        grid, context_x = _apply_grid_stride(grid, context_x, grid_stride)

        return context_x, context_y, grid

    # Single strategy selection
    if context_selection not in CONTEXT_SELECTION_METHODS:
        msg = (
            f"Unknown context_selection: {context_selection}. "
            "Choose from 'random', 'sobol', 'halton', "
            "'latin_hypercube', 'pca', 'pca_sobol', "
            "'pca_halton', 'pca_lhs'"
        )
        raise ValueError(msg)

    context_x, context_y = CONTEXT_SELECTION_METHODS[context_selection](
        dataloader,
        n_context_points,
        n_pca_components,
        pca_variance_threshold,
        seed,
    )

    if time_keep is not None:
        t_keep = max(1, int(time_keep))
        if context_x.shape[-2] > t_keep:
            context_x = context_x[..., :t_keep, :]

    grid, _ = _make_grid_from_loader(dataloader)
    grid, context_x = _apply_grid_stride(grid, context_x, grid_stride)

    return context_x, context_y, grid
