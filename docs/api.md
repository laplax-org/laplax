# API Reference

Welcome to the Laplax API reference.

## Package Overview

The `laplax` package contains a [high-level API](./reference/main_api.md), that is designed to be used out-of-the box, and a modular low-level API, which provides exposes all essential building blocks for the high-level API and can be used for fast experimentations. The low-level API is organized into the following modules:

- [`laplax.curv`](./reference/curv.md): Tools for computing and approximating curvature information
- [`laplax.eval`](./reference/eval.md): Evaluation metrics and utilities for assessing predictive uncertainty
- [`laplax.util`](./reference/util.md): Various utilities for working with PyTrees, DataLoaders, and other common utilities.

## Main design decisions

### Model function signature

`laplax` operates by taking an arbitrary `model_fn` with (key-word) signature `model_fn(input, params)`. This allows for a wide range of JAX-based neural network libraries to be used. For [`flax.nnx`](https://flax.readthedocs.io/en/latest/nnx_basics.html) and [`equinox`](https://equinox.readthedocs.io/en/latest/index.html), this would look like:


=== "flax.nnx"
    ```python
    from flax import nnx

    model = ...

    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]
    ```

=== "equinox"
    ```python
    from equinox import filter_jit

    model = ...

    params, static = eqx.partition(model, eqx.is_array)

    def model_fn(input, params):
        model = eqx.combine(params, static)
        return model(input)
    ```
