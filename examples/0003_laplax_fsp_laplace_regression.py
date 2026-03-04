# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Function-Space Prior (FSP) Laplace for regression
#
# This tutorial mirrors the standard *weight-space* Laplace regression example, but
# uses a **function-space** Gaussian Process (GP) prior.
#
# ## Key idea
#
# Pick a set of **context points** $C = \{c_m\}_{m=1}^M$ in input space and a GP
# prior on the function values at those points:
#
# $$
# f(C) \sim \mathcal{N}\big(m(C), K(C, C)\big).
# $$
#
# This induces an RKHS energy (quadratic form):
#
# $$
# \|f\|_{K}^{2} := (f(C) - m(C))^\top K(C, C)^{-1} (f(C) - m(C)).
# $$
#
# We train a MAP model by minimizing **negative log-likelihood + RKHS energy**:
#
# $$
# \mathcal{L}(\theta) = -\log p(\mathcal{D}\mid\theta)
#  + \tfrac{1}{2}\,\tau\,\|f_\theta\|_{K}^{2},
# $$
#
# where $\tau$ is the prior precision (regularizer strength).
#
# After MAP training, we build a (low-rank) Laplace approximation around the MAP
# parameters with `create_fsp_posterior`, and push the resulting parameter uncertainty
# forward to predictive mean/variance using the same pushforward API as in the
# standard Laplace example.

# %%
from __future__ import annotations

from functools import partial

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flax import nnx
from helper import DataLoader, DictLoader, get_sinusoid_example, kernel_from_gpjax
from plotting import plot_regression_with_uncertainty
from tqdm.auto import tqdm

from laplax.curv.fsp import create_fsp_posterior
from laplax.enums import LossFn, LowRankMethod
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_setup,
    set_lin_pushforward,
)
from laplax.util.context_points import select_context_points
from laplax.util.objective import (
    add_ll_rho,
    create_fsp_objective_from_chol,
    create_loss_nll,
    create_loss_reg_from_chol,
    fsp_wrapper,
    split_params_ll_rho,
)
from laplax.util.tree import to_dtype

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Dataset: truncated sine regression
#
# We use the same truncated sine dataset as in the standard regression tutorial.

# %%
key = jr.PRNGKey(0)

num_train = 200
num_valid = 50
num_test = 800

X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
    num_train_data=num_train,
    num_valid_data=num_valid,
    num_test_data=num_test,
    sigma_noise=0.1,
    sinus_factor=2 * jnp.pi,
    intervals=[(-1.5, -1.0), (1.0, 1.5)],
    test_interval=(-3.0, 3.0),
    rng_key=key,
)

# %%
dataset_size = int(X_train.shape[0])
batch_size = 32
train_loader = DictLoader(
    DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
)

# %% [markdown]
# ## Model and `model_fn` / `params` split
#
# As in the standard Laplace tutorial, we split the `flax.nnx` model into
# `model_fn(*, input, params)` and a `params` PyTree.

# %%
NNXList = nnx.List


class MLP(nnx.Module):
    def __init__(self, width: int, depth: int, rngs: nnx.Rngs):
        self.inp = nnx.Linear(1, width, rngs=rngs)
        self.hid = NNXList([
            nnx.Linear(width, width, rngs=rngs) for _ in range(depth - 1)
        ])
        self.out = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.tanh(self.inp(x))
        for layer in self.hid:
            x = nnx.tanh(layer(x))
        return self.out(x)


model = MLP(width=64, depth=3, rngs=nnx.Rngs(0))
graph_def, base_params = nnx.split(model)


def base_model_fn(*, input, params):
    out = nnx.call((graph_def, params))(input)
    return out[0] if isinstance(out, tuple) else out


# %% [markdown]
# ## Choose context points + GP prior
#
# We select context points **once** (for MAP training) using a low-discrepancy
# procedure based on the data distribution, and define a GP prior via a kernel.
#
# Here we use a periodic kernel (good inductive bias for a sine).

# %%
n_context_map = 500
C, _ = select_context_points(
    data=train_loader, method="grid", n_context_points=n_context_map
)

gpjax_kernel = gpx.kernels.Periodic(lengthscale=1.0, variance=1.0, period=1.0)
KCC = kernel_from_gpjax(C, C, gpjax_kernel)

jitter = 1e-4
L_C = jnp.linalg.cholesky(KCC + jitter * jnp.eye(KCC.shape[0], dtype=KCC.dtype))

prior_mean_context = jnp.zeros((C.shape[0], 1), dtype=jnp.float64)

# %% [markdown]
# ## MAP training with FSP objective
#
# We wrap the base model with `fsp_wrapper` to add a learnable noise scale
# (internally stored as `ll_rho`), and build:
#
# - a Gaussian negative log-likelihood loss, and
# - an RKHS (quadratic) regularizer from the Cholesky of $K(C,C)$.
#
# The combined objective is created via `create_fsp_objective_from_chol`.

# %%
sigma_min = 1e-3
prior_prec = 1.0  # τ in the derivation above

fsp_model = fsp_wrapper(base_model_fn, sigma_min=sigma_min)
params = add_ll_rho(base_params, init_ll_rho=0.0)

loss_nll_fn = create_loss_nll(model_fn=fsp_model, dataset_size=dataset_size)
loss_reg_fn = create_loss_reg_from_chol(
    model_fn=fsp_model,
    prior_mean=prior_mean_context,
    prior_cov_chol=L_C,
    has_batch_dim=True,
    normalize="mean",
)
fsp_objective = create_fsp_objective_from_chol(
    model_fn=fsp_model,
    dataset_size=dataset_size,
    prior_mean=prior_mean_context,
    prior_cov_chol=L_C,
    has_batch_dim=True,
    normalize="mean",
    regularizer_scale=0.5 * prior_prec,
)

# %%
steps = 15_000
lr = 1e-3
grad_clip = 1.0

tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
opt_state = tx.init(params)


@jax.jit
def train_step(params, opt_state, key):
    key, k = jr.split(key)
    idx = jr.randint(k, (batch_size,), 0, dataset_size)
    xb = X_train[idx]
    yb = y_train[idx]
    batch = {"input": xb, "target": yb}
    context = {"context": C}

    def loss_fn(p):
        sigma = jnp.clip(fsp_model.sigma(p), a_min=sigma_min)
        nll = loss_nll_fn(batch, p, sigma)
        reg = loss_reg_fn(context, p)
        loss = fsp_objective(batch, context, p, sigma)
        return loss, {"loss": loss, "nll": nll, "rkhs": reg, "sigma": sigma}

    (_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, aux


key = jr.PRNGKey(123)
hist = []

for _ in tqdm(range(steps), desc="Training FSP-MAP (NLL + RKHS)"):
    params, opt_state, key, aux = train_step(params, opt_state, key)
    hist.append(aux["loss"])

hist = jnp.asarray(hist)
print("Final MAP sigma:", float(fsp_model.sigma(params)))

# %% [markdown]
# ### Quick visualization of the MAP fit

# %%
X_grid = jnp.linspace(-3.0, 3.0, 500).reshape(-1, 1)
y_map = fsp_model(input=X_grid, params=params).reshape(-1)

plot_regression_with_uncertainty(
    X_train=X_train,
    y_train=y_train,
    X_pred=X_grid,
    y_pred=y_map,
    title="FSP MAP training (Periodic GP prior)",
)
plt.ylim(-5, 5)
plt.xlim(-3, 3)
plt.show()

# %% [markdown]
# ## FSP-Laplace posterior + predictive
#
# Now we create the FSP Laplace posterior around the MAP solution.
#
# - We remove the noise parameter (`ll_rho`) from the parameter PyTree.\n# - We call
# `create_fsp_posterior(...)` to build a low-rank posterior function.\n# - We push it
# forward with the usual linearized predictive machinery.\n\n# For regression with a
# learned noise level $\sigma$, the GGN scaling factor is\n# $1/\sigma^2$.

# %%
base_params_map, _ll_rho = split_params_ll_rho(params)
base_params_map = to_dtype(base_params_map, dtype=jnp.float64)

sigma_map = float(fsp_model.sigma(params))
print("MAP sigma:", sigma_map)


def kernel_fn(x, y):
    return kernel_from_gpjax(x, y, gpjax_kernel)


fsp_posterior_fn = create_fsp_posterior(
    model_fn=base_model_fn,
    params=base_params_map,
    data=train_loader,
    loss_fn=LossFn.MSE,
    key=jr.PRNGKey(0),
    kernel_fn=kernel_fn,
    low_rank_method=LowRankMethod.LANCZOS,
    jitter=1e-4,
    ggn_factor=1.0 / (sigma_map**2),
    n_context_points=120,
    context_selection="grid",
    n_chunks=4,
    max_rank=60,
)

set_prob_predictive = partial(
    set_lin_pushforward,
    model_fn=base_model_fn,
    mean_params=base_params_map,
    posterior_fn=fsp_posterior_fn,
    pushforward_fns=[lin_setup, lin_pred_mean, lin_pred_var, lin_pred_std],
)

prior_arguments = {"prior_prec": float(prior_prec)}
prob_predictive = set_prob_predictive(prior_arguments=prior_arguments)

pred = jax.vmap(prob_predictive)(X_grid)
y_mean = pred["pred_mean"][:, 0]
y_var = pred["pred_var"][:, 0]
y_std = jnp.sqrt(jnp.maximum(y_var, 0.0))

plot_regression_with_uncertainty(
    X_train=X_train,
    y_train=y_train,
    X_pred=X_grid,
    y_pred=y_mean,
    y_std=y_std,
    title="FSP-Laplace predictive (linearized pushforward)",
)
plt.ylim(-5, 5)
plt.xlim(-3, 3)
plt.show()
