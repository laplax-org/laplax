import equinox as eqx
import hypothesis as hp
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from hypothesis import given
from hypothesis import strategies as st

from laplax.util.flatten import create_partial_pytree_flattener


# ----- Hypothesis Composite Functions -----
@st.composite
def pytree_params(draw):
    key = jax.random.PRNGKey(2)
    depth = draw(st.integers(min_value=1, max_value=4))
    branching_factor = draw(st.integers(min_value=1, max_value=4))
    num_array_dims = draw(st.integers(min_value=2, max_value=5))

    return key, depth, branching_factor, num_array_dims

# ----- Utility Functions -----


def create_kernel_size(kernel_size, dim):
    """Creates a tuple of length dim filled with kernel_size."""
    return (kernel_size,) * dim


def get_conv_params(dim):
    """Get kernel size and strides for a given dimension."""
    kernel_size = create_kernel_size(3, dim)
    strides = create_kernel_size(1, dim)
    return kernel_size, strides


def create_random_pytree(key, depth, branching_factor, num_array_dims):
    """Create random pytree where each leaf is a multidimensional array.
    
    Parameters:
        key: jax.random.PRNGKey - random key.
        depth (int): Depth of the tree.
        branching_factor (int): Number of children per non-leaf node.
        num_array_dims (int): Number of dimensions for arrays.
        axis (int): concatenation axis of flatten/unflatten operation.
        axis_shape (int): shape of the axis dimension. default is 5 for no reason.
    
    Returns:
        Random Pytree with JAX arrays as leaves.
    """
    key_global, key = jax.random.split(key)
    shapes = list(jax.random.randint(key_global, shape=(num_array_dims,), minval=1, maxval=6))

    def create_array(key):
        '''Create a random array with shape (num_array_dims,)'''
        leaf_shape = tuple(shapes[i] for i in range(num_array_dims))
        return jax.random.uniform(key, shape=shapes)

    def create_tree(key, current_depth):
        if current_depth == 0:
            return create_array(key)
        keys = jax.random.split(key, branching_factor)
        return {f'node_{i}': create_tree(keys[i], current_depth - 1)
                for i in range(branching_factor)}

    return create_tree(key, depth)

# ----- Model Creation Helpers -----


def create_cnn_nnx(num_layers, features, dim=1, key=None):
    """Create a CNN using nnx."""
    kernel_size, strides = get_conv_params(dim)

    layers = []
    for i in range(num_layers - 1):
        conv = nnx.Conv(in_features=features[i], out_features=features[i + 1],
                        kernel_size=kernel_size, strides=strides, rngs=key)
        layers.append(lambda x, conv=conv: conv(x))
        layers.append(lambda x: nnx.relu(x))
    return nnx.Sequential(*layers)


def create_cnn_linen(x, num_layers, features, dim=1):
    """Create a CNN using linen."""
    kernel_size, strides = get_conv_params(dim)

    for i in range(num_layers):
        x = nn.Conv(features=features, kernel_size=kernel_size, strides=strides)(x)
        x = nn.relu(x)
    return x


def create_mlp_eqx(features):
    """Create an MLP using equinox."""
    key = jax.random.split(jax.random.PRNGKey(0), len(features))
    model = LinearNN(key, features)
    return model


# ----- Model Classes -----
class LinearNN(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, keys, hidden):
        self.layers = [
            eqx.nn.Linear(hidden[i], hidden[i + 1], key=keys[i]) for i in range(len(hidden) - 1)
        ]
        self.extra_bias = jax.random.uniform(keys[0], (10,))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias


class Block(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, use_running_average=True, rngs=rngs)

    def __call__(self, x: jax.Array):
        return nnx.relu(self.bn(self.linear(x)))


class MLP(nnx.Module):
    def __init__(self, num_layers: int, dim: int, *, rngs: nnx.Rngs):
        self.blocks = [Block(dim, rngs=rngs) for _ in range(num_layers)]

    def __call__(self, x: jax.Array):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nnx.Module):
    def __init__(self, dim: int, kernel_size: tuple = (3, 3), *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(dim, dim, kernel_size=kernel_size, rngs=rngs)
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, use_running_average=True, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.conv(x)
        x = self.linear(x)
        return nnx.relu(self.bn(x))


class CNN(nnx.Module):
    def __init__(self, num_layers: int, dim: int, kernel_size: int, *, rngs: nnx.Rngs):
        kernel_size = create_kernel_size(kernel_size, dim)
        self.blocks = [ConvBlock(dim, kernel_size, rngs=rngs) for _ in range(num_layers)]

    def __call__(self, x: jax.Array):
        for block in self.blocks:
            x = block(x)
        return x


class CNN_nnx(nnx.Module):
    def __init__(self, num_layers, features, dim=1, rngs=None):
        self.key = rngs
        self.model = create_cnn_nnx(num_layers, features, dim, self.key)

    def __call__(self, x):
        return self.model(x)


class CNN_linen(nn.Module):
    num_layers: int
    features: int
    dim: int = 1

    @nn.compact
    def __call__(self, x):
        return create_cnn_linen(x, self.num_layers, self.features, self.dim)

# ----- Testing Utilities -----


def round_trip(params):
    """Test round-trip conversion of PyTree flattening and unflattening."""
    flatten, unflatten = create_partial_pytree_flattener(params)
    flattened = flatten(params)
    reconstructed = unflatten(flattened)

    # Check shapes match
    shapes_match = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: a.shape == b.shape, params, reconstructed)
    )
    assert shapes_match, "Shape mismatch in reconstructed PyTree"

    # Check values match
    values_match = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-5, atol=1e-5),
            params,
            reconstructed
        )
    )
    assert values_match, "Values in reconstructed PyTree don't match original"


def assert_correct_shapes(params, dim):
    """Assert that flattened parameters have correct shape."""
    flatten, unflatten = create_partial_pytree_flattener(params)
    flat = flatten(params)
    assert flat.shape[-1] == dim, f"Expected last dimension to be {dim}, got {flat.shape[-1]}"
    assert flat.ndim == 2, f"Expected 2D array, got {flat.ndim}D"


def create_input_shape(dim, batch_size, features, size=4):
    """Create appropriate input shape for testing."""
    if dim == 1:
        return (batch_size, size, features)
    if dim == 2:
        return (batch_size, size, size, features)
    if dim == 3:
        return (batch_size, size, size, size, features)
    raise ValueError(f"Invalid dimension: {dim}")

# ----- Tests -----


@given(
    num_layers=st.sampled_from([1, 2, 32]),
    features=st.sampled_from([1, 16, 32]),
    dim=st.sampled_from([1, 2, 3]),
    batch_size=st.sampled_from([1, 2, 32]),
)
@hp.settings(deadline=None)
def test_cnn_linen(num_layers, features, dim, batch_size):
    """Test PyTree flattening with linen CNN models."""
    key = jax.random.PRNGKey(0)
    input_shape = create_input_shape(dim, batch_size, features)

    x = jnp.ones(input_shape)
    model = CNN_linen(num_layers=num_layers, features=features, dim=dim)
    params = model.init(key, x)
    round_trip(params)


@given(
    num_layers=st.sampled_from([1, 2, 3]),
    dim=st.sampled_from([1, 2, 3]),
    kernel_size=st.sampled_from([1, 3, 5, 7]),
)
@hp.settings(deadline=None)
def test_block(num_layers, dim, kernel_size):
    """Test PyTree flattening with CNN blocks."""
    key = jax.random.PRNGKey(0)
    model = CNN(num_layers=num_layers, dim=dim, kernel_size=kernel_size, rngs=nnx.Rngs(2))
    graph, params = nnx.split(model)
    round_trip(params)


@given(
        pytree_params(),
)
@hp.settings(deadline=None)
def test_partial_pytree_flattener_with_random_pytree(params):
    """Test PyTree flattening with random PyTree structures."""
    key, depth, branching_factor, num_array_dims = params
    pytree = create_random_pytree(key, depth, branching_factor, num_array_dims)
    round_trip(pytree)
    # assert_correct_shapes(pytree, array_shape)


def test_mlp_eqx():
    model = create_mlp_eqx([1, 16, 32])
    params, static = eqx.partition(model, eqx.is_array)

    round_trip(params)
    assert_correct_shapes(params, 32)
