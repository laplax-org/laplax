from typing import Any, Callable, List, Tuple

import equinox as eqx
import jax
import pytest_cases
from flax import linen as nn
from flax import nnx
from jax import numpy as jnp

def generate_data(key, input_shape, target_shape):
    return {
        "input": jax.random.normal(key, input_shape),
        "target": jax.random.normal(key, target_shape),
    }


class BaseClassificationTask:
    def __init__(
        self,
        in_channels: int,
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
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_model_fn(self) -> Callable:
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_parameters(self) -> Any:
        return self.params

    def get_data_batch(self, batch_size: int) -> dict:
        key = jax.random.key(self.seed)
        return generate_data(
            key, (batch_size, self.in_channels), (batch_size, self.out_channels)
        )


class LinenClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="linen")

    def _initialize(self):
        class CNN(nn.Module):
            in_channels: int
            conv_features: int
            conv_kernel_size: int
            avg_pool_shape: int
            avg_pool_strides: int
            out_channels: int

            def setup(self):
                self.conv1 = nn.Conv(features=self.conv_features, kernel_size=(self.conv_kernel_size, self.conv_kernel_size))
                self.linear1 = nn.Dense(features=self.out_channels)

            def __call__(self, x):
                # Ensure x has 4 dimensions (batch_size, height, width, channels)
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)

                x = nn.relu(self.conv1(x))
                x = nn.avg_pool(x, window_shape=self.avg_pool_shape, strides=self.avg_pool_strides)

                x = x.reshape((x.shape[0], -1))  # Shape: (batch_size, flattened_features)

                x = nn.relu(self.linear1(x))

                return x

        rng_key = jax.random.PRNGKey(self.seed)
        data = generate_data(rng_key, (1, self.in_channels), (1, self.out_channels))
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