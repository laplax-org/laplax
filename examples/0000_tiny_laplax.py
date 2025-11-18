# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tiny illustration of Laplace approximations

# %% [markdown]
# This script is a super tiny illustration of a Laplace approximation - one where
# curvature approximation is tractable and can be easy visualised

# %%
import jax.numpy as jnp
from jax.nn import relu
from plotting import plot_figure_1

from laplax import laplace

# You need optimized parameters,
best_params = {"theta1": jnp.array(1.6546547), "theta2": jnp.array(1.0420421)}


def model_fn(input, params):
    return relu(params["theta1"] * input - 1) * params["theta2"]


data = {  # and training data.
    "input": jnp.array([1.0, -1.0]).reshape(2, 1),
    "target": jnp.array([1.0, -1.0]).reshape(2, 1),
}

# Then apply laplax
posterior_fn, _ = laplace(
    model_fn,
    best_params,
    data,
    loss_fn="mse",
    curv_type="full",
)
curv = posterior_fn({"prior_prec": 0.2}).state["scale"]

# to get figure 1.
plot_figure_1(best_params, curv, save_fig=False)

# %%
