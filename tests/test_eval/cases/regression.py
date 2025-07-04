from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import pytest_cases
from flax import linen as nn
from flax import nnx


def generate_data(key, input_shape, target_shape):
    return {
        "input": jax.random.normal(key, input_shape),
        "target": jax.random.normal(key, target_shape),
    }


class BaseRegressionTask:
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        seed: int,
        framework: str,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
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
        key = jax.random.PRNGKey(self.seed)
        return generate_data(
            key, (batch_size, self.in_channels), (batch_size, self.out_channels)
        )


class LinenRegressionTask(BaseRegressionTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="linen")

    def _initialize(self):
        class MLP(nn.Module):
            hidden_channels: int
            out_channels: int

            def setup(self):
                self.linear1 = nn.Dense(self.hidden_channels)
                self.linear2 = nn.Dense(self.out_channels)

            def __call__(self, x):
                x = self.linear1(x)
                x = nn.tanh(x)
                x = self.linear2(x)
                return x

        rng_key = jax.random.PRNGKey(self.seed)
        data = generate_data(rng_key, (1, self.in_channels), (1, self.out_channels))
        self.model = MLP(
            hidden_channels=self.hidden_channels, out_channels=self.out_channels
        )
        self.params = self.model.init(rng_key, data["input"])

    def get_model_fn(self):
        def model_fn(input, params):
            return self.model.apply(params, input)

        return model_fn


class NNXRegressionTask(BaseRegressionTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="nnx")

    def _initialize(self):
        class MLP(nnx.Module):
            def __init__(self, rngs, in_channels, hidden_channels, out_channels):
                self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
                self.tanh = nnx.tanh
                self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

            def __call__(self, x):
                x = self.linear1(x)
                x = self.tanh(x)
                x = self.linear2(x)
                return x

        rngs = nnx.Rngs(self.seed)
        self.model = MLP(
            rngs=rngs,
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )

        self.graph_def, self.params = nnx.split(self.model)

    def get_model_fn(self):
        def model_fn(input, params):
            return nnx.call((self.graph_def, params))(input)[0]

        return model_fn


class EquinoxRegressionTask(BaseRegressionTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="equinox")

    def _initialize(self):
        class MLP(eqx.Module):
            layers: list[eqx.nn.Linear]
            activation: Callable = eqx.static_field()

            def __init__(self, in_channels, hidden_channels, out_channels, keys):
                self.layers = [
                    eqx.nn.Linear(in_channels, hidden_channels, key=keys[0]),
                    eqx.nn.Linear(hidden_channels, out_channels, key=keys[1]),
                ]
                self.activation = jax.nn.tanh

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = self.activation(layer(x))
                return self.layers[-1](x)

        rng_key = jax.random.key(self.seed)
        keys = jax.random.split(rng_key, 2)
        self.model = MLP(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            keys=keys,
        )
        self.params, self.static = eqx.partition(self.model, eqx.is_array)

    def get_model_fn(self):
        def model_fn(input, params):
            new_model = eqx.combine(params, self.static)
            return new_model(input)

        return model_fn


def create_task(task_class, in_channels, hidden_channels, out_channels, seed):
    """Factory function to create regression tasks.

    Returns:
        The initialized task.
    """
    return task_class(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        seed=seed,
    )


@pytest_cases.parametrize(
    "task_class", [LinenRegressionTask, NNXRegressionTask, EquinoxRegressionTask]
)
@pytest_cases.parametrize("in_channels", [10, 20])
@pytest_cases.parametrize("hidden_channels", [10])
@pytest_cases.parametrize("out_channels", [1])
def case_regression(
    task_class, in_channels: int, hidden_channels: int, out_channels: int
):
    """Test regression tasks with multiple frameworks and parameter combinations.

    Returns:
        The initialized task.
    """
    seed = 42
    return task_class(in_channels, hidden_channels, out_channels, seed)
