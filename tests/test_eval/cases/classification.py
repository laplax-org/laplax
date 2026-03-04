"""Test cases for classifciation."""

from functools import partial

import equinox as eqx
import jax
import pytest_cases
from flax import linen as nn
from flax import nnx
from jax import numpy as jnp

from laplax.types import Any, Callable


def generate_data(key, input_shape, target_shape):
    return {
        "input": jax.random.normal(key, input_shape),
        "target": jax.nn.one_hot(
            jax.random.randint(key, (target_shape[0],), 0, maxval=target_shape[1]),
            num_classes=target_shape[1],
        ),
    }


class BaseClassificationTask:
    def __init__(
        self,
        in_channels: tuple,
        conv_features: int,
        conv_kernel_size: int,
        avg_pool_shape: int,
        avg_pool_strides: int,
        linear_in: int,
        out_channels: int,
        seed: int,
        framework: str,
    ):
        self.in_channels = in_channels
        self.conv_features = conv_features
        self.conv_kernel_size = conv_kernel_size
        self.avg_pool_shape = avg_pool_shape
        self.avg_pool_strides = avg_pool_strides
        self.linear_in = linear_in
        self.out_channels = out_channels
        self.seed = seed
        self.framework = framework
        self.loss_fn_type = "mse"
        self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def get_model_fn(self) -> Callable:
        raise NotImplementedError

    def get_parameters(self) -> Any:
        return self.params

    def get_data_batch(self, batch_size: int) -> dict:
        key = jax.random.key(self.seed)
        return generate_data(
            key, (batch_size, *self.in_channels), (batch_size, self.out_channels)
        )


class LinenClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="linen")

    def _initialize(self):
        class CNN(nn.Module):
            in_channels: tuple
            conv_features: int
            conv_kernel_size: int
            avg_pool_shape: int
            avg_pool_strides: int
            out_channels: int

            def setup(self):
                self.conv1 = nn.Conv(
                    features=self.conv_features,
                    kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
                )
                self.linear1 = nn.Dense(features=self.out_channels)

            def __call__(self, x):
                # Ensure x has 4 dimensions (batch_size, height, width, channels)
                batch_dim = True
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)
                    batch_dim = False

                x = nn.relu(self.conv1(x))
                x = nn.avg_pool(
                    x,
                    window_shape=(self.avg_pool_shape, self.avg_pool_shape),
                    strides=(self.avg_pool_strides, self.avg_pool_strides),
                )

                x = x.reshape((
                    x.shape[0],
                    -1,
                ))  # Shape: (batch_size, flattened_features)

                x = nn.relu(self.linear1(x))

                if not batch_dim:
                    return jnp.squeeze(x)
                return x

        rng_key = jax.random.PRNGKey(self.seed)
        data = generate_data(rng_key, (1, *self.in_channels), (1, self.out_channels))
        self.model = CNN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_features=self.conv_features,
            conv_kernel_size=self.conv_kernel_size,
            avg_pool_shape=self.avg_pool_shape,
            avg_pool_strides=self.avg_pool_strides,
        )
        self.params = self.model.init(rng_key, data["input"])

    def get_model_fn(self):
        def model_fn(params, input):
            return self.model.apply(params, input)

        return model_fn


class NNXClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="nnx")

    def _initialize(self):
        class CNN(nnx.Module):
            def __init__(
                self,
                rngs: nnx.Rngs,
                in_channels: tuple,
                conv_features: int,
                conv_kernel_size: int,
                avg_pool_shape: int,
                avg_pool_strides: int,
                linear_in: int,
                out_channels: int,
            ):
                self.conv1 = nnx.Conv(
                    in_channels[2],
                    conv_features,
                    kernel_size=(conv_kernel_size, conv_kernel_size),
                    rngs=rngs,
                )
                self.avg_pool = partial(
                    nnx.avg_pool,
                    window_shape=(avg_pool_shape, avg_pool_shape),
                    strides=(avg_pool_strides, avg_pool_strides),
                )
                self.linear1 = nnx.Linear(linear_in, out_channels, rngs=rngs)

            def __call__(self, x):
                batch_dim = True
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)
                    batch_dim = False

                x = self.avg_pool(nnx.relu(self.conv1(x)))
                x = x.reshape(x.shape[0], -1)  # flatten
                x = nnx.relu(self.linear1(x))

                if not batch_dim:
                    return jnp.squeeze(x)
                return x

        rngs = nnx.Rngs(self.seed)

        self.model = CNN(
            rngs=rngs,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_features=self.conv_features,
            conv_kernel_size=self.conv_kernel_size,
            avg_pool_shape=self.avg_pool_shape,
            avg_pool_strides=self.avg_pool_strides,
            linear_in=self.linear_in,
        )
        _, self.params, _ = nnx.split(self.model, nnx.Param, ...)

    def get_model_fn(self):
        def model_fn(params, input):
            _, _, rest = nnx.split(self.model, nnx.Param, ...)
            nnx.update(
                self.model, nnx.GraphState.merge(params, rest)
            )  # Load the model from parameters
            return self.model(input)

        return model_fn


class EquinoxClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="equinox")

    def _initialize(self):
        class CNN(eqx.Module):
            conv1: eqx.nn.Conv2d
            avg_pool: eqx.nn.AvgPool2d
            linear1: eqx.nn.Linear

            def __init__(
                self,
                key,
                in_channels: tuple,
                conv_features: int,
                conv_kernel_size: int,
                avg_pool_shape: int,
                avg_pool_strides: int,
                linear_in: int,
                out_channels: int,
            ):
                key1, key2 = jax.random.split(key, 2)
                self.conv1 = eqx.nn.Conv2d(
                    in_channels[2],
                    conv_features,
                    kernel_size=conv_kernel_size,
                    padding=((1, 1), (1, 1)),
                    key=key1,
                )
                self.avg_pool = eqx.nn.AvgPool2d(
                    kernel_size=avg_pool_shape, stride=avg_pool_strides
                )
                self.linear1 = eqx.nn.Linear(linear_in, out_channels, key=key2)

            def __call__(self, x):
                x = nn.relu(self.conv1(x))
                x = self.avg_pool(x)
                x = jnp.ravel(x)
                x = nn.relu(self.linear1(x))
                return x

        key = jax.random.key(self.seed)

        self.model = CNN(
            key=key,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_features=self.conv_features,
            conv_kernel_size=self.conv_kernel_size,
            avg_pool_shape=self.avg_pool_shape,
            avg_pool_strides=self.avg_pool_strides,
            linear_in=self.linear_in,
        )
        self.params, self.static = eqx.partition(self.model, eqx.is_array)

    def get_model_fn(self):
        def model_fn(params, input):
            new_model = eqx.combine(params, self.static)
            input = jnp.transpose(input, (2, 0, 1))
            return new_model(input)

        return model_fn


def create_task(
    task_class,
    in_channels,
    conv_features,
    conv_kernel_size,
    avg_pool_shape,
    avg_pool_strides,
    out_channels,
    linear_in,
    seed,
):
    """Factory function to create regression tasks.

    Returns:
        The initialized task.
    """
    return task_class(
        in_channels=in_channels,
        conv_features=conv_features,
        conv_kernel_size=conv_kernel_size,
        avg_pool_shape=avg_pool_shape,
        avg_pool_strides=avg_pool_strides,
        linear_in=linear_in,
        out_channels=out_channels,
        seed=seed,
    )


@pytest_cases.parametrize(
    "task_class",
    [LinenClassificationTask, NNXClassificationTask, EquinoxClassificationTask],
)
@pytest_cases.parametrize(
    "in_channels, conv_features, conv_kernel_size, avg_pool_shape, "
    "avg_pool_strides, linear_in, out_channels",
    [
        # Configuration 1
        ((8, 8, 1), 2, 3, 2, 2, 32, 10),
        # Configuration 2
        ((11, 5, 3), 2, 3, 2, 2, 20, 2),
    ],
)
def case_classification(
    task_class,
    in_channels: tuple,
    conv_features: int,
    conv_kernel_size: int,
    avg_pool_shape: int,
    avg_pool_strides: int,
    linear_in: int,
    out_channels: int,
):
    """Test classification tasks with multiple frameworks and parameter combinations.

    Args:
        task_class: The classification task class to test.
        in_channels (tuple): Input channel configuration.
        conv_features (int): Number of convolutional features.
        conv_kernel_size (int): Convolution kernel size.
        avg_pool_shape (int): Average pooling shape.
        avg_pool_strides (int): Strides for average pooling.
        linear_in (int): Input dimension for the linear layer.
        out_channels (int): Number of output channels.

    Returns:
        The instantiated classification task object.
    """
    seed = 42
    return task_class(
        in_channels,
        conv_features,
        conv_kernel_size,
        avg_pool_shape,
        avg_pool_strides,
        linear_in,
        out_channels,
        seed,
    )
