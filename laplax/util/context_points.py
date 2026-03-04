# /laplax/util/context_points.py

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA

from laplax.util.loader import is_data_loader, validate_data_transform

if TYPE_CHECKING:
    from collections.abc import Callable

    from laplax.types import (
        Array,
        Data,
        Float,
        InputArray,
        Int,
        Iterable,
        KeyType,
        Kwargs,
    )

# -----------------------------------------------------------------------------
# Grid helpers (no classes/dataclasses)
# -----------------------------------------------------------------------------

# A lightweight “struct” for grid metadata (no dataclass/class).
# Keys:
# - min_val: list[float]
# - max_val: list[float]
# - shape: tuple[int, ...]
# - spacing: list[float]
GridInfo = dict[str, object]


def _make_grid_info(
    *,
    min_val: list[float],
    max_val: list[float],
    shape: tuple[int, ...],
    spacing: list[float],
) -> GridInfo:
    """Create a grid-info dictionary with a unified structure.

    Args:
        min_val: Minimum value per dimension.
        max_val: Maximum value per dimension.
        shape: Grid resolution per dimension.
        spacing: Grid spacing per dimension.

    Returns:
        Dictionary containing grid metadata.
    """
    return {
        "min_val": min_val,
        "max_val": max_val,
        "shape": shape,
        "spacing": spacing,
    }


def _compute_grid_spacing(
    min_val: list[float] | np.ndarray,
    max_val: list[float] | np.ndarray,
    shape: tuple[int, ...],
) -> list[float]:
    """Compute grid spacing from bounds and shape.

    Args:
        min_val: Minimum value per dimension.
        max_val: Maximum value per dimension.
        shape: Grid resolution per dimension.

    Returns:
        Spacing per dimension. If a dimension has resolution 1, spacing is 0.0.
    """
    return [
        (max_v - min_v) / (s - 1) if s > 1 else 0.0
        for min_v, max_v, s in zip(min_val, max_val, shape, strict=False)
    ]


def _load_all_data_from_dataloader(
    dataloader: Iterable,
) -> tuple[InputArray, InputArray]:
    """Load all data from a dataloader into arrays.

    Args:
        dataloader: Iterable yielding batches of data.

    Returns:
        Tuple of (all_inputs, all_targets) concatenated across all batches.
    """
    x_list: list[Array] = []
    y_list: list[Array] = []

    it = iter(dataloader)
    first_batch = next(it)
    transform = validate_data_transform(first_batch)

    # include the first batch we already pulled
    first = transform(first_batch)
    x_list.append(jnp.array(first["input"]))
    y_list.append(jnp.array(first["target"]))

    for batch in it:
        data = transform(batch)
        x_list.append(jnp.array(data["input"]))
        y_list.append(jnp.array(data["target"]))

    all_x = jnp.concatenate(x_list, axis=0)
    all_y = jnp.concatenate(y_list, axis=0)
    return all_x, all_y


def _make_grid_from_data_shape(
    data_shape: tuple[int, ...],
    *,
    bounds: tuple[float, float] | None = None,
    n_points_per_dim: int = 10,
) -> GridInfo:
    """Create grid information based on data shape.

    Args:
        data_shape: Feature shape (without batch dimension).
        bounds: Lower/upper bound to use for all dimensions.
        n_points_per_dim: Number of grid points per dimension.

    Returns:
        Grid metadata dictionary.
    """
    n_dims = int(np.prod(data_shape))
    lo, hi = bounds if bounds is not None else (-3.0, 3.0)

    min_val = [lo] * n_dims
    max_val = [hi] * n_dims
    shape = tuple([n_points_per_dim] * n_dims)
    spacing = _compute_grid_spacing(min_val, max_val, shape)

    return _make_grid_info(
        min_val=min_val, max_val=max_val, shape=shape, spacing=spacing
    )


def _make_grid_from_loader(data_loader: Iterable, *, n_points: int) -> GridInfo:
    """Infer grid from a data loader (heuristic).

    Args:
        data_loader: Iterable yielding batches.
        n_points: Target total number of grid points (approximate).

    Returns:
        Grid metadata dictionary.
    """
    x_all, _ = _load_all_data_from_dataloader(data_loader)
    n_dims = int(x_all.shape[-1])

    # points_per_dim ** n_dims ≈ n_points
    points_per_dim = int(np.power(n_points, 1.0 / max(n_dims, 1)))
    points_per_dim = max(points_per_dim, 2)

    min_val = np.min(np.array(x_all), axis=0).tolist()
    max_val = np.max(np.array(x_all), axis=0).tolist()

    # Add padding
    range_val = np.array(max_val) - np.array(min_val)
    min_val = (np.array(min_val) - 0.1 * range_val).tolist()
    max_val = (np.array(max_val) + 0.1 * range_val).tolist()

    shape = tuple([points_per_dim] * n_dims)
    spacing = _compute_grid_spacing(min_val, max_val, shape)

    return _make_grid_info(
        min_val=min_val, max_val=max_val, shape=shape, spacing=spacing
    )


def _grid_context_points(
    *,
    data: Data | Iterable,
    n_context_points: Int,
    seed: Int | None = None,
    bounds: tuple[float, float] | None = None,
) -> tuple[InputArray, InputArray]:
    """Generate context points on a grid inferred from data bounds (or explicit bounds).

    Produces exactly `n_context_points` by subsampling or padding with uniform samples.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to return (exact).
        seed: Random seed (used for subsample/padding).
        bounds: Optional (lo, hi) used for *all* dimensions.

    Returns:
        (context_x, context_y)

    Raises:
        ValueError: If the grid inference produces incompatible feature dimension.
    """
    key = jax.random.PRNGKey(int(seed)) if seed is not None else jax.random.PRNGKey(0)

    if is_data_loader(data):
        all_x, all_y = _load_all_data_from_dataloader(data)
    else:
        transform = validate_data_transform(data)
        batch_data = transform(data)
        all_x = jnp.asarray(batch_data["input"])
        all_y = jnp.asarray(batch_data["target"])

    feature_shape = tuple(int(s) for s in all_x.shape[1:])
    n_features = int(np.prod(feature_shape))

    # infer grid and generate coords in flattened feature space
    grid_info = infer_grid_info(
        data, n_context_points=int(n_context_points), bounds=bounds
    )
    coords = make_coords_from_grid(grid_info)  # (n_grid, n_features)

    if int(coords.shape[1]) != n_features:
        msg = (
            "Grid inference produced incompatible feature dimension. "
            f"Got coords.shape[1]={int(coords.shape[1])}, expected {n_features}."
        )
        raise ValueError(msg)

    n_grid = int(coords.shape[0])
    n_ctx = int(n_context_points)

    # --- enforce exact n_context_points
    if n_grid > n_ctx:
        # deterministic-ish subsample (seeded)
        key, k = jax.random.split(key)
        idx = jax.random.choice(k, n_grid, shape=(n_ctx,), replace=False)
        coords = coords[idx]
    elif n_grid < n_ctx:
        # pad with uniform samples inside the same bounds as the grid
        min_val = jnp.asarray(grid_info["min_val"], dtype=coords.dtype)
        max_val = jnp.asarray(grid_info["max_val"], dtype=coords.dtype)
        key, k = jax.random.split(key)
        extra = jax.random.uniform(
            k,
            shape=(n_ctx - n_grid, n_features),
            minval=min_val,
            maxval=max_val,
            dtype=coords.dtype,
        )
        coords = jnp.concatenate([coords, extra], axis=0)

    context_x = coords.reshape(n_ctx, *feature_shape)

    # y is not needed for FSP prior on f(C), but keep API shape consistent
    key, ky = jax.random.split(key)
    context_y = _sample_uniform_like(data=all_y, n_points=n_ctx, key=ky)

    return context_x, context_y


def make_coords_from_grid(grid_info: GridInfo) -> Array:
    """Generate coordinate points from grid info.

    Args:
        grid_info: Grid metadata dictionary created by this module.

    Returns:
        Array of shape (n_points_total, n_dims) containing grid coordinates.

    Raises:
        TypeError: If grid_info fields have unexpected types.
    """
    min_val = grid_info["min_val"]
    max_val = grid_info["max_val"]
    shape = grid_info["shape"]

    if (
        not isinstance(min_val, list)
        or not isinstance(max_val, list)
        or not isinstance(shape, tuple)
    ):
        msg = (
            "grid_info must contain min_val/max_val as list[float] "
            "and shape as tuple[int, ...]."
        )
        raise TypeError(msg)

    linspaces = list(starmap(jnp.linspace, zip(min_val, max_val, shape, strict=False)))
    meshgrid = jnp.meshgrid(*linspaces, indexing="ij")
    coords = jnp.stack([m.reshape(-1) for m in meshgrid], axis=-1)
    return coords


def infer_grid_info(
    data: Data | Iterable,
    *,
    n_context_points: int,
    bounds: tuple[float, float] | None = None,
) -> GridInfo:
    """Infer grid info from data or explicit bounds.

    Args:
        data: A single batch (dict) or an iterable of batches.
        n_context_points: Target total number of grid points (approximate).
        bounds: If provided, use bounds and a heuristic resolution.

    Returns:
        Grid metadata dictionary.
    """
    if bounds is not None:
        if is_data_loader(data):
            it = iter(data)
            transform = validate_data_transform(next(it))
            batch = transform(next(iter(data)))
            data_shape = batch["input"].shape[1:]
        else:
            data_shape = data["input"].shape[1:]
        return _make_grid_from_data_shape(data_shape, bounds=bounds)

    if is_data_loader(data):
        return _make_grid_from_loader(data, n_points=n_context_points)

    # Explicit data array/batch
    transform = validate_data_transform(data)
    batch = transform(data)
    x = jnp.asarray(batch["input"])

    n_dims = int(x.shape[-1])
    points_per_dim = int(np.power(n_context_points, 1.0 / max(n_dims, 1)))
    points_per_dim = max(points_per_dim, 2)

    min_v = jnp.min(x, axis=0)
    max_v = jnp.max(x, axis=0)
    range_v = max_v - min_v

    min_val = (min_v - 0.1 * range_v).tolist()
    max_val = (max_v + 0.1 * range_v).tolist()

    shape = tuple([points_per_dim] * n_dims)
    spacing = _compute_grid_spacing(min_val, max_val, shape)

    return _make_grid_info(
        min_val=min_val, max_val=max_val, shape=shape, spacing=spacing
    )


# -----------------------------------------------------------------------------
# Random Fourier Features (RFF) helpers
# -----------------------------------------------------------------------------


def _rff_kernel(
    input_shape: tuple[int, ...],
    *,
    lengthscale: float,
    variance: float,
    key: KeyType,
    n_features: int = 100,
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Create an RFF approximation for an RBF kernel.

    Args:
        input_shape: Expected input shape without batch dimension. Must be 1D.
        lengthscale: RBF lengthscale.
        variance: RBF variance.
        key: JAX PRNG key.
        n_features: Number of random Fourier features.

    Returns:
        Tuple (feature_map, kernel_diag) where:
        - feature_map maps x -> phi(x) with shape (N, n_features)
        - kernel_diag returns the diagonal of K(x, x) as a vector

    Raises:
        ValueError: If input_shape is not 1D.
    """
    if len(input_shape) != 1:
        msg = "Only 1D input supported for now."
        raise ValueError(msg)

    dim = int(input_shape[0])

    k1, k2 = jax.random.split(key)
    omega = jax.random.normal(k1, (n_features, dim)) / lengthscale
    bias = jax.random.uniform(k2, (n_features,), minval=0.0, maxval=2.0 * np.pi)

    scale = jnp.sqrt(2.0 * variance / float(n_features))

    def feature_map(x: Array) -> Array:
        """Compute random Fourier features.

        Args:
            x: Input array of shape (N, dim).

        Returns:
            Feature matrix of shape (N, n_features).
        """
        proj = jnp.dot(x, omega.T) + bias
        return scale * jnp.cos(proj)

    def kernel_diag(x: Array) -> Array:
        """Return the diagonal of the RBF kernel at x.

        Args:
            x: Input array of shape (N, dim).

        Returns:
            Vector of shape (N,) containing k(x_i, x_i) for each row.
        """
        return jnp.full((x.shape[0],), variance)

    return feature_map, kernel_diag


def sample_gp_batch_1d(
    *,
    n_samples: int,
    n_points: int,
    lengthscale: float,
    variance: float,
    key: KeyType,
) -> Array:
    """Sample from a 1D GP using RFF.

    Args:
        n_samples: Number of functions to sample.
        n_points: Number of points per function.
        lengthscale: RBF lengthscale.
        variance: RBF variance.
        key: JAX PRNG key.

    Returns:
        Array of shape (n_samples, n_points).
    """
    grid = jnp.linspace(-3.0, 3.0, n_points).reshape(-1, 1)
    feat_map, _ = _rff_kernel(
        (1,), lengthscale=lengthscale, variance=variance, key=key, n_features=500
    )
    feats = feat_map(grid)  # (N, n_feat)

    w = jax.random.normal(key, (500, n_samples))  # (n_feat, S)
    f = jnp.dot(feats, w)  # (N, S)
    return f.T  # (S, N)


def sample_gp_batch_2d(
    *,
    n_samples: int,
    grid_info: GridInfo,
    lengthscale: float,
    variance: float,
    key: KeyType,
) -> Array:
    """Sample from a 2D GP using RFF.

    Args:
        n_samples: Number of functions to sample.
        grid_info: Grid metadata.
        lengthscale: RBF lengthscale.
        variance: RBF variance.
        key: JAX PRNG key.

    Returns:
        Array of shape (n_samples, n_grid_points_total).
    """
    coords = make_coords_from_grid(grid_info)
    dim = int(coords.shape[-1])

    feat_map, _ = _rff_kernel(
        (dim,),
        lengthscale=lengthscale,
        variance=variance,
        key=key,
        n_features=1000,
    )
    feats = feat_map(coords)
    w = jax.random.normal(key, (1000, n_samples))
    f = jnp.dot(feats, w)
    return f.T


# -----------------------------------------------------------------------------
# Context point selection
# -----------------------------------------------------------------------------


def _generate_low_discrepancy_sequence(
    *,
    n_dims: Int,
    n_points: Int,
    sequence_type: str = "sobol",
    seed: Int | None = None,
) -> Array:
    """Generate a low-discrepancy sequence in [0, 1]^n_dims.

    Args:
        n_dims: Number of dimensions.
        n_points: Number of points to generate.
        sequence_type: Sequence type ("sobol", "halton", "latin_hypercube").
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_points, n_dims) with values in [0, 1].

    Raises:
        ValueError: If sequence_type is not supported.
    """
    seq_type = sequence_type.lower()

    if seq_type == "sobol":
        m = int(np.ceil(np.log2(max(1, int(n_points)))))
        sampler = qmc.Sobol(d=int(n_dims), scramble=True, seed=seed)
        points_full = sampler.random_base2(m)
        points = points_full[: int(n_points)]
    elif seq_type == "halton":
        sampler = qmc.Halton(d=int(n_dims), scramble=True, seed=seed)
        points = sampler.random(int(n_points))
    elif seq_type == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=int(n_dims), seed=seed)
        points = sampler.random(int(n_points))
    else:
        msg = (
            f"Unknown sequence type: {sequence_type}. "
            "Choose from 'sobol', 'halton', 'latin_hypercube'."
        )
        raise ValueError(msg)

    return jnp.array(points, dtype=jnp.float32)


def _sample_uniform_like(
    data: Array,
    n_points: Int,
    key: KeyType,
) -> Array:
    """Sample points uniformly in the axis-aligned bounding box of data.

    Args:
        data: Array of shape (batch, ...).
        n_points: Number of points to sample.
        key: JAX PRNG key.

    Returns:
        Array of shape (n_points, ...) matching the non-batch shape of data.
    """
    batch_size = int(data.shape[0])
    feature_shape = data.shape[1:]

    data_flat = data.reshape(batch_size, -1)
    n_features = int(data_flat.shape[1])

    feat_min = jnp.min(data_flat, axis=0, keepdims=True)
    feat_max = jnp.max(data_flat, axis=0, keepdims=True)

    samples_flat = jax.random.uniform(
        key=key,
        shape=(int(n_points), n_features),
        minval=feat_min,
        maxval=feat_max,
    )

    return samples_flat.reshape(int(n_points), *feature_shape)


def _fit_pca_on_data(
    *,
    data: Array,
    pca_variance_threshold: Float,
) -> tuple[Array, Array, Array, PCA, tuple[int, ...]]:
    """Fit PCA on data and return transformed scores and metadata.

    Args:
        data: Array of shape (batch, ...).
        pca_variance_threshold: Variance threshold (0-1) used by sklearn PCA.

    Returns:
        Tuple of:
        - scores: PCA scores (batch, n_components)
        - feat_mean: Feature mean (1, n_features)
        - feat_std: Feature std (1, n_features)
        - pca: Fitted PCA object
        - original_shape: Original feature shape (without batch)
    """
    batch_size = int(data.shape[0])
    original_shape = tuple(int(x) for x in data.shape[1:])

    data_flat = data.reshape(batch_size, -1)
    data_np = np.array(data_flat)

    feat_mean = data_np.mean(axis=0, keepdims=True)
    feat_std = data_np.std(axis=0, keepdims=True)
    data_std = (data_np - feat_mean) / (feat_std + 1e-8)

    pca = PCA(n_components=float(pca_variance_threshold))
    scores = pca.fit_transform(data_std)

    return (
        jnp.array(scores, dtype=jnp.float32),
        jnp.array(feat_mean, dtype=jnp.float32),
        jnp.array(feat_std + 1e-8, dtype=jnp.float32),
        pca,
        original_shape,
    )


def _sample_from_pca(
    *,
    scores: Array,
    feat_mean: Array,
    feat_std: Array,
    pca: PCA,
    n_points: Int,
    sequence_type: str,
    seed: Int | None,
    original_shape: tuple[int, ...],
    jitter_scale: Float = 1e-8,
) -> Array:
    """Sample new points in PCA space using a low-discrepancy sequence.

    Args:
        scores: PCA scores from training data.
        feat_mean: Feature mean used for standardization.
        feat_std: Feature std used for standardization.
        pca: Fitted PCA object.
        n_points: Number of points to sample.
        sequence_type: Low-discrepancy sequence type.
        seed: Random seed.
        original_shape: Original feature shape (without batch).
        jitter_scale: Jitter scale relative to feature std.

    Returns:
        Array of sampled points with shape (n_points, *original_shape).
    """
    n_dims = int(pca.n_components_)
    ld_unit = _generate_low_discrepancy_sequence(
        n_dims=n_dims,
        n_points=int(n_points),
        sequence_type=sequence_type,
        seed=seed,
    )

    scores_np = np.array(scores)
    scores_min = scores_np.min(axis=0)
    scores_max = scores_np.max(axis=0)
    sampled_scores = np.array(ld_unit) * (scores_max - scores_min) + scores_min

    sampled_std = pca.inverse_transform(sampled_scores)
    sampled = sampled_std * np.array(feat_std) + np.array(feat_mean)

    sampled_jax = jax.device_put(sampled)

    if float(jitter_scale) > 0.0:
        jitter_seed = int(seed + 9999) if seed is not None else 42
        jitter_key = jax.random.PRNGKey(jitter_seed)
        jitter = (
            float(jitter_scale)
            * feat_std
            * jax.random.normal(
                jitter_key, shape=sampled_jax.shape, dtype=sampled_jax.dtype
            )
        )
        sampled_jax = sampled_jax + jitter

    n_features = int(np.prod(original_shape))
    return sampled_jax.reshape(int(n_points), n_features).reshape(
        int(n_points), *original_shape
    )


def _random_context_points(
    *,
    data: Data | Iterable,
    n_context_points: Int,
    key: KeyType,
) -> tuple[InputArray, InputArray]:
    """Generate random context points uniformly in data bounding box.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to generate.
        key: JAX PRNG key.

    Returns:
        Tuple of (context_x, context_y).
    """
    if is_data_loader(data):
        all_x, all_y = _load_all_data_from_dataloader(data)
    else:
        transform = validate_data_transform(data)
        batch_data = transform(data)
        all_x = jnp.asarray(batch_data["input"])
        all_y = jnp.asarray(batch_data["target"])

    key_x, key_y = jax.random.split(key)
    context_x = _sample_uniform_like(data=all_x, n_points=n_context_points, key=key_x)
    context_y = _sample_uniform_like(data=all_y, n_points=n_context_points, key=key_y)
    return context_x, context_y


def _pca_context_points(
    *,
    data: Data | Iterable,
    n_context_points: Int,
    sequence_type: str = "sobol",
    pca_variance_threshold: Float = 0.95,
    seed: Int | None = None,
    jitter_scale: Float = 1e-3,
) -> tuple[InputArray, InputArray]:
    """Generate context points using PCA and low-discrepancy sequences.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to generate.
        sequence_type: Low-discrepancy sequence type.
        pca_variance_threshold: Variance threshold for PCA.
        seed: Random seed.
        jitter_scale: Jitter scale for PCA sampling.

    Returns:
        Tuple of (context_x, context_y).
    """
    if is_data_loader(data):
        all_x, all_y = _load_all_data_from_dataloader(data)
    else:
        transform = validate_data_transform(data)
        batch_data = transform(data)
        all_x = jnp.asarray(batch_data["input"])
        all_y = jnp.asarray(batch_data["target"])

    x_scores, x_mean, x_std, pca_x, x_shape = _fit_pca_on_data(
        data=all_x,
        pca_variance_threshold=pca_variance_threshold,
    )

    context_x = _sample_from_pca(
        scores=x_scores,
        feat_mean=x_mean,
        feat_std=x_std,
        pca=pca_x,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
        original_shape=x_shape,
        jitter_scale=jitter_scale,
    )

    # y: keep it simple — uniform within y bounding box
    rng = np.random.default_rng(seed)
    y_seed = int(rng.integers(0, 2**31 - 1))
    key_y = jax.random.PRNGKey(y_seed)
    context_y = _sample_uniform_like(data=all_y, n_points=n_context_points, key=key_y)

    return context_x, context_y


def _gp_context_points(
    *,
    data: Data | Iterable,
    n_context_points: int,
    seed: int | None = None,
) -> tuple[InputArray, InputArray]:
    """Generate context points by a GP-inspired heuristic.

    Currently implemented as PCA + Sobol for robust space-filling.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to generate.
        seed: Random seed.

    Returns:
        Tuple of (context_x, context_y).
    """
    return _pca_context_points(
        data=data,
        n_context_points=n_context_points,
        sequence_type="sobol",
        seed=seed,
    )


def get_context_points(
    *,
    data: Data | Iterable,
    n_points: int,
    method: str = "random",
    pca_variance_threshold: Float = 0.95,
    seed: Int | None = None,
    jitter_scale: Float = 1e-3,
) -> tuple[InputArray, InputArray]:
    """Public helper: get context points without classes.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_points: Number of context points to generate.
        method: One of "random", "sobol", "halton", "latin_hypercube", "pca", "gp".
        pca_variance_threshold: Variance threshold for PCA-based methods.
        seed: Random seed.
        jitter_scale: Jitter scale for PCA-based methods.

    Returns:
        Tuple of (context_x, context_y).
    """
    return select_context_points(
        data=data,
        method=method,
        n_context_points=n_points,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
        jitter_scale=jitter_scale,
    )


def select_context_points(
    *,
    data: Data | Iterable,
    method: str,
    n_context_points: Int = 50,
    pca_variance_threshold: Float = 0.95,
    seed: Int | None = None,
    jitter_scale: Float = 1e-3,
    **kwargs: Kwargs,
) -> tuple[InputArray, InputArray]:
    """Select context points from data using the specified method.

    Args:
        data: Single batch (dict) or iterable of batches.
        method: Selection method ("random", "sobol", "halton",
            "latin_hypercube", "pca", "gp").
        n_context_points: Number of context points to generate.
        pca_variance_threshold: Variance threshold for PCA-based methods.
        seed: Random seed for reproducibility.
        jitter_scale: Jitter scale for PCA methods.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (context_x, context_y).

    Raises:
        ValueError: If method is not supported.
    """
    key = jax.random.PRNGKey(int(seed)) if seed is not None else jax.random.PRNGKey(0)
    method_lower = method.lower()
    bounds = kwargs.get("bounds", (-3.0, 3.0))

    if method_lower == "random":
        return _random_context_points(
            data=data, n_context_points=n_context_points, key=key
        )

    if method_lower in {"grid", "mesh", "meshgrid"}:
        return _grid_context_points(
            data=data,
            n_context_points=n_context_points,
            seed=seed,
            bounds=bounds,
        )

    if method_lower in {"sobol", "pca_sobol", "pca"}:
        return _pca_context_points(
            data=data,
            n_context_points=n_context_points,
            sequence_type="sobol",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    if method_lower in {"halton", "pca_halton"}:
        return _pca_context_points(
            data=data,
            n_context_points=n_context_points,
            sequence_type="halton",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    if method_lower in {"latin_hypercube", "pca_lhs"}:
        return _pca_context_points(
            data=data,
            n_context_points=n_context_points,
            sequence_type="latin_hypercube",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    if method_lower == "gp":
        return _gp_context_points(
            data=data, n_context_points=int(n_context_points), seed=seed
        )

    msg = (
        f"Unknown context selection method: {method}. "
        "Choose from 'random', 'sobol', 'halton', 'latin_hypercube', 'pca', 'gp', "
        "'grid', 'mesh', 'meshgrid'."
    )
    raise ValueError(msg)
