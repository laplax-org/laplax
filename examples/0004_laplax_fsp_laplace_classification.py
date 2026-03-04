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
# # Function-Space Prior (FSP) Laplace for classification (two moons)
#
# This tutorial follows the same structure as the regression example:
#
# 1. Train a MAP model, but with a **function-space prior** implemented as an RKHS
#    energy term on the logits at *context points*.
# 2. Build an FSP-Laplace posterior around the MAP parameters with
#    `create_fsp_posterior`.
# 3. Push forward the weight uncertainty to get predictive mean/variance of logits,
#    and turn this into probabilities and predictive entropy.

# %%
from __future__ import annotations

from functools import partial

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from helper import (
    DataLoader,
    DictLoader,
    bce_logits_full_data_approx,
    entropy_bernoulli,
    get_two_moons_example,
    kernel_from_gpjax,
    sigmoid_gaussian_moment,
)
from plotting import plot_two_moons_fsp
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
from laplax.util.objective import compute_rkhs_energy_from_chol
from laplax.util.tree import to_dtype

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Dataset: two moons

# %%
X, y, X_np, y_np = get_two_moons_example(num_data=300, noise=0.15, seed=0)
dataset_size = int(X.shape[0])

xmin, ymin = np.min(X_np, axis=0) - 1.0
xmax, ymax = np.max(X_np, axis=0) + 1.0

batch_size = 64
train_loader = DictLoader(DataLoader(X, y, batch_size=batch_size, shuffle=True))

# %% [markdown]
# ## Model and MAP training objective (CE + RKHS)
#
# We train a binary classifier that outputs a single logit $z(x)$.
#
# The FSP-MAP objective is:
#
# $$
# \mathcal{L}(\theta) = \text{NLL}_\text{BCE}(\theta) +
# \tfrac{1}{2}\tau\,
# \|z_\theta\|_K^2,
# $$
#
# where the RKHS energy is computed on logits at context points $C$.


# %%
class MLP2D(nnx.Module):
    def __init__(self, width: int, depth: int, rngs: nnx.Rngs):
        self.inp = nnx.Linear(2, width, rngs=rngs)
        self.hid = nnx.List([
            nnx.Linear(width, width, rngs=rngs) for _ in range(depth - 1)
        ])
        self.out = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.tanh(self.inp(x))
        for layer in self.hid:
            x = nnx.tanh(layer(x))
        return self.out(x)


model = MLP2D(width=64, depth=3, rngs=nnx.Rngs(0))
graph_def, base_params = nnx.split(model)


def base_model_fn(*, input, params):
    out = nnx.call((graph_def, params))(input)
    return out[0] if isinstance(out, tuple) else out


# %% [markdown]
# ### Context points + GP prior kernel

# %%
n_context = 25 * 25
C, _ = select_context_points(
    data=train_loader, method="pca", n_context_points=n_context
)

gpjax_kernel = gpx.kernels.Matern52(lengthscale=0.6, variance=1.0)
KCC = kernel_from_gpjax(C, C, gpjax_kernel)

jitter = 1e-4
L_C = jnp.linalg.cholesky(KCC + jitter * jnp.eye(KCC.shape[0], dtype=KCC.dtype))

prior_mean_context = jnp.zeros((C.shape[0], 1), dtype=jnp.float64)

# %%
prior_prec_fsp = 1.0
lr = 1e-3
grad_clip = 1.0
steps = 20_000

tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
opt_state = tx.init(base_params)


@jax.jit
def train_step(params, opt_state, key):
    key, k = jr.split(key)
    idx = jr.randint(k, (batch_size,), 0, dataset_size)
    xb = X[idx]
    yb = y[idx]

    def loss_fn(p):
        logits_b = base_model_fn(input=xb, params=p)
        nll = bce_logits_full_data_approx(yb, logits_b, dataset_size=dataset_size)

        logits_context = base_model_fn(input=C, params=p)
        energy = compute_rkhs_energy_from_chol(
            f_hat=logits_context,
            prior_mean=prior_mean_context,
            prior_cov_chol=L_C,
            normalize="mean",
        )
        loss = nll + 0.5 * prior_prec_fsp * energy
        return loss, {"loss": loss, "nll": nll, "rkhs": energy}

    (_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, aux


key = jr.PRNGKey(42)
for _ in tqdm(range(steps), desc="Training FSP-MAP (BCE + RKHS)"):
    base_params, opt_state, key, aux = train_step(base_params, opt_state, key)

# %% [markdown]
# ## FSP-Laplace posterior + predictive on a 2D grid
#
# We compute the Laplace posterior in weight space induced by the function-space prior.
# For binary cross entropy we set `loss_fn=LossFn.BINARY_CROSS_ENTROPY` and use
# `ggn_factor=1.0`.

# %%
base_params_map = to_dtype(base_params, dtype=jnp.float64)


def kernel_fn(x, y):
    return kernel_from_gpjax(x, y, gpjax_kernel)


fsp_posterior_fn = create_fsp_posterior(
    model_fn=base_model_fn,
    params=base_params_map,
    data=train_loader,
    loss_fn=LossFn.BINARY_CROSS_ENTROPY,
    key=jr.PRNGKey(0),
    kernel_fn=kernel_fn,
    low_rank_method=LowRankMethod.LANCZOS,
    jitter=1e-4,
    ggn_factor=1.0,
    n_context_points=int(n_context),
    context_selection="pca",
    n_chunks=4,
    max_rank=80,
)

set_prob_predictive = partial(
    set_lin_pushforward,
    model_fn=base_model_fn,
    mean_params=base_params_map,
    posterior_fn=fsp_posterior_fn,
    pushforward_fns=[lin_setup, lin_pred_mean, lin_pred_var, lin_pred_std],
)
prob_predictive = set_prob_predictive(prior_arguments={"prior_prec": 1e-2})

# %%
grid_res = 140
gx = np.linspace(xmin, xmax, grid_res)
gy = np.linspace(ymin, ymax, grid_res)
XX, YY = np.meshgrid(gx, gy, indexing="xy")
Xg = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
Xg_j = jnp.asarray(Xg, dtype=jnp.float64)

pred = jax.vmap(prob_predictive)(Xg_j)
logits_mean = pred["pred_mean"][:, 0]
logits_var = pred["pred_var"][:, 0]

p = sigmoid_gaussian_moment(logits_mean, logits_var)
H = entropy_bernoulli(p)

p_np = np.asarray(p).reshape(grid_res, grid_res)
H_np = np.asarray(H).reshape(grid_res, grid_res)

fig = plot_two_moons_fsp(
    X_np=X_np,
    y_np=y_np,
    XX=XX,
    YY=YY,
    prob_grid=p_np,
    entropy_grid=H_np,
    x_bounds=(xmin, xmax),
    y_bounds=(ymin, ymax),
    title_prob="FSP-Laplace p(y=1|x)",
    title_entropy="FSP-Laplace predictive entropy",
)
plt.show()
