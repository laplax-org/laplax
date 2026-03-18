# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Active learning for 2D classification

# %% [markdown]
# In this tutorial, we showcase active learning using the posterior uncertainty on a
# two-dimensional classification task with three classes.
# The tutorial relies on concepts from the tutorial for active learning on regression.


# %%
from functools import partial

import jax
import optax
from flax import nnx
from helper import DataLoader, Model, split, train_model
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from optax.losses import softmax_cross_entropy_with_integer_labels
from plotting import (
    plot_datapoints,
    plot_decision_boundaries,
    plot_next_point,
    plot_prediction,
    show_animation_classification,
)
from tqdm import tqdm

from laplax.api import calibration, estimate_curvature
from laplax.curv import create_ggn_mv, create_posterior_fn
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_setup,
    set_lin_pushforward,
)

seed = 2392386
key = random.key(seed)
init_data_key, cali_data_key, passive_data_key, sampling_key = random.split(key, 4)


# %% [markdown]
# First, we define the ground truth decision boundary function.


# %%
@jax.jit
def true_function(point):
    def f1(x):
        return 1.9 * x**3 - 1.5 * x**2 + 0.5

    def f2(x):
        return -1.5 * x**2 + 2 * x + 0.2

    x, y = point[0], point[1]
    return jnp.where(y >= f2(x), 2, jnp.where(y >= f1(x), 1, 0))


# %% [markdown]
# We generate some initial datapoints and visualize them.

# %%
n_initial_datapoints = 20


def generate_dataset(n_points, key):
    key1, key2 = random.split(key)
    xs = random.uniform(key1, shape=n_points, minval=0, maxval=1)
    ys = random.uniform(key2, shape=n_points, minval=0, maxval=1)
    datapoints = jnp.stack((xs, ys)).mT
    labels = jax.vmap(true_function)(datapoints)
    return DataLoader(datapoints, labels, batch_size=10)


class_dataloader = generate_dataset(n_initial_datapoints, init_data_key)
plt.figure(figsize=(5, 5))
plot_decision_boundaries()
plot_datapoints(class_dataloader)
plt.show()


# %% [markdown]
# As our model, we reuse the model from the other active learning tutorial,
# a small fully connected network with four layers.
# Here, we have 2 input features, 3 output logits and use cross entropy loss.
#
# We train the model on a small starting batch of datapoints.


# %%
@nnx.jit
def train_step(model, optimizer, batch, labels):
    def loss_fn(model):
        logits = model(batch)
        return softmax_cross_entropy_with_integer_labels(logits, labels).sum()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


start_model = Model(
    in_channels=2, hidden_channels=32, out_channels=3, rngs=nnx.Rngs(seed)
)

params = nnx.state(start_model)
total_params = sum(p.size for p in jax.tree.leaves(params))
print(f"Total number of parameters: {total_params}")

lr = 1e-3
n_initial_epochs = n_initial_datapoints * 50
class_optimizer = nnx.Optimizer(start_model, optax.adam(lr))
class_model = train_model(
    start_model,
    class_optimizer,
    class_dataloader,
    train_step,
    n_epochs=n_initial_epochs,
)

# %% [markdown]
# We visualize the trained model's predictions as background color in the data plane.

# %%
xv, yv = jnp.meshgrid(jnp.linspace(0, 1, 100), jnp.linspace(0, 1, 100))
gridpoints = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
true_labels = jax.vmap(true_function)(gridpoints)
logits = jax.vmap(class_model)(gridpoints)
preds = logits.argmax(axis=-1)

plot_decision_boundaries()
plot_datapoints(class_dataloader)
plot_prediction(preds)


# %% [markdown]
# The existent data is fit well, indicating that the training has worked.
# The true decision boundary is not recovered however, simply because the model hasn't
# seen enough data yet.
# We are going to continue learning with more actively chosen data.

# %% [markdown]
# In this example, we use the first rule for maximal total information gain, as shown in
# the active learning tutorial for regression.
# The maximum of the total information gain is at the same location as the maximum of
# the model's predicted standard deviation, as the constants and logarithm in the
# formula do not change the location of the maximum.
# Therefore, it is sufficient here to calculate the posterior uncertainty using laplax.
# As in the other tutorial, we calibrate the prior precision to get meaningful
# uncertainty estimates. This time however, we calibrate on a calibration dataset.
# This is because on the training dataset, the model makes no errors and hence,
# the calibrated prior precision diverges to large values.
# This leads to bad results during active learning.
# Of course, this introduces a dependency on more datapoints compared to
# passive learning, which does not align well with the goal to make learning more
# data-efficient. We argue however that a validation set is anyway needed to optimize
# other hyperparameters in a realistic setting, and that the prior precision is just
# another hyperparameter to be fitted using this validation set.


# %%
def construct_prob_predictive(data, model):
    dataset = {"input": data.X, "target": data.y}
    model_fn, params = split(model)
    ggn_mv = create_ggn_mv(
        model_fn,
        params,
        dataset,
        loss_fn="cross_entropy",
    )
    posterior_fn = create_posterior_fn(
        curv_type="full",
        mv=ggn_mv,
        layout=params,
    )
    prob_predictive = partial(
        set_lin_pushforward,
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        pushforward_fns=[
            lin_setup,
            lin_pred_mean,
            lin_pred_std,
        ],
    )
    return dataset, model_fn, params, ggn_mv, posterior_fn, prob_predictive


def calibrate_prior_precision(data, model, grid_params):
    """Calibrate the prior precision.

    Args:
        data: dataloader to use for calibration
        model: nnx.Module
        grid_params: dict of parameters for grid search

    Returns:
        Calibrated prior precision.
    """
    dataset, model_fn, params, ggn_mv, posterior_fn, _ = construct_prob_predictive(
        data, model
    )
    curv_estimate = estimate_curvature(
        curv_type="full",
        mv=ggn_mv,
        layout=params,
    )
    return calibration(
        posterior_fn=posterior_fn,
        model_fn=model_fn,
        params=params,
        data=dataset,
        loss_fn="CROSS_ENTROPY",
        predictive_type="MC_BRIDGE",
        curv_estimate=curv_estimate,
        curv_type="full",
        calibration_objective="ECE",
        calibration_method="GRID_SEARCH",
        **grid_params,
    )[0]


grid_params = {
    "log_prior_prec_min": -2.0,
    "log_prior_prec_max": 3.0,
    "grid_size": 100,
}

n_calibration_datapoints = 30
cali_dataloader = generate_dataset(n_calibration_datapoints, cali_data_key)

prior_args = calibrate_prior_precision(cali_dataloader, class_model, grid_params)


print("Prior precision: ", prior_args["prior_prec"])


# %%
def compute_uncertainty(prob_predictive, prior_args):
    prob_predictive = prob_predictive(prior_arguments=prior_args)
    pred = jax.vmap(prob_predictive)(gridpoints)
    return pred["pred_std"]


prob_predictive = construct_prob_predictive(class_dataloader, class_model)[-1]
uncertainties = compute_uncertainty(prob_predictive, prior_args)
uncertainty = uncertainties[jnp.arange(10000), preds]


# %% [markdown]
# We calculate the uncertainty on a regular grid within data space,
# and find its maximum on the grid.
# This is going to be the best next datapoint location.
# We visualize the uncertainty as the alpha-value of the prediction colors,
# with stronger color corresponding to larger uncertainty.
#
# 'get_next_point_sampled' is an alternative rule to find the next datapoint,
# which samples from the data plane by interpreting the uncertainty as logits to
# a categorical distribution. This way, random points with high uncertainty are chosen,
# which prevents the active learning loop from sampling in the same region repeatedly.
# Feel free to try out both methods in the active learning loop and see the difference!
#


# %%
def get_next_point(uncertainty):
    return gridpoints[jnp.argmax(uncertainty)]


def get_next_point_sampled(key, uncertainty):
    return gridpoints[jax.random.categorical(key, uncertainty)]


next_point = get_next_point(uncertainty)

plot_decision_boundaries()
plot_datapoints(class_dataloader)
plot_prediction(preds, uncertainty)
plot_next_point(next_point)


# %%
def evaluate(model):
    logits = jax.vmap(model)(gridpoints)
    preds = logits.argmax(axis=-1)
    return preds


def accuracy(model):
    preds = evaluate(model)
    acc = jnp.mean(preds == true_labels)
    return acc


# %% [markdown]
# We see that the uncertainty is large where the model thinks the
# decision boundary lies, and low elsewhere.
# This means the active learning loop, which we are going to implement next,
# is going to sample in these areas,
# confirming or adapting the found decision boundary.

# %%
learning_rounds = 50
epochs_per_learning_round = 35
plot_data = []
sampling_keys = jax.random.split(sampling_key, learning_rounds)
accuracies = []

for i, _key in tqdm(enumerate(sampling_keys)):
    print(f"Active learning round {i + 1}")
    # 1) Sample new datapoint
    next_target = true_function(next_point)
    class_dataloader = class_dataloader.add(next_point, jnp.atleast_1d(next_target))

    # 2) Continue training
    class_model = train_model(
        class_model,
        class_optimizer,
        class_dataloader,
        train_step,
        n_epochs=epochs_per_learning_round,
    )
    grid_preds = jnp.argmax(class_model(gridpoints), axis=-1)

    # 3) Compute uncertainty
    prob_predictive = construct_prob_predictive(class_dataloader, class_model)[-1]
    uncertainties = compute_uncertainty(prob_predictive, prior_args)
    uncertainty = uncertainties[jnp.arange(10000), grid_preds]

    # 4) Find next datapoint location
    # next_point = get_next_point_sampled(_key, uncertainty)
    next_point = get_next_point(uncertainty)

    # Evaluation
    accuracies.append(accuracy(class_model))

    # Plotting
    data_preds = jnp.argmax(class_model(class_dataloader.X), axis=-1)
    plot_data.append((
        grid_preds,
        class_dataloader,
        uncertainty,
        next_point,
    ))
    print("-----------------------")

# %%
show_animation_classification(plot_data)

# %% [markdown]
# We see that the datapoints are concentrated around the true decision boundary,
# therefore increasing the gained information compared to sampling datapoints randomly
# from the plane.

# %% [markdown]
# ### Comparison against passive learning

# %% [markdown]
# As in the active learning example for regression, we compare our actively trained
# model against one that is trained as usual, with a fixed dataset.

# %%
n_passive_datapoints = n_initial_datapoints + learning_rounds
key1, key2 = random.split(passive_data_key)
xs = random.uniform(key1, shape=n_passive_datapoints, minval=0, maxval=1)
ys = random.uniform(key2, shape=n_passive_datapoints, minval=0, maxval=1)
datapoints = jnp.stack((xs, ys)).mT
labels = jax.vmap(true_function)(datapoints)
passive_class_dl = DataLoader(datapoints, labels, batch_size=10)

passive_class_model = Model(
    in_channels=2, hidden_channels=32, out_channels=3, rngs=nnx.Rngs(seed)
)
passive_class_optimizer = nnx.Optimizer(passive_class_model, optax.adam(lr))

n_epochs_passive = n_initial_epochs + learning_rounds * epochs_per_learning_round
passive_class_model = train_model(
    passive_class_model,
    passive_class_optimizer,
    passive_class_dl,
    train_step,
    n_epochs=n_epochs_passive,
)

passive_logits = jax.vmap(passive_class_model)(gridpoints)
passive_preds = passive_logits.argmax(axis=-1)

plot_decision_boundaries()
plot_datapoints(passive_class_dl)
plot_prediction(passive_preds)
plt.show()

# %% [markdown]
# ### Evaluation

# %%
accuracies = jnp.array(accuracies)

passive_preds = evaluate(passive_class_model)
passive_acc = accuracy(passive_class_model)

active_preds = evaluate(class_model)
active_acc = accuracies[-1]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.suptitle("Misclassification regions of")
ax1.set_title("active model")
plot_prediction(active_preds == true_labels, ax=ax1)
ax2.set_title("passive model")
plot_prediction(passive_preds == true_labels, ax=ax2)
plt.show()

# %% [markdown]
# The passively trained model's prediction boundaries are not well-aligned with
# the ground truth, simply because there are fewer datapoints close to the
# ground truth boundaries, compared to the actively learned model. Datapoints far from
# the decision boundary contribute little to no information to the model.
# This results in a higher overall accuracy for the actively learned model.

# %%
print(f"Accuracy of actively trained model: {accuracies[-1]:.3f}")
print(f"Accuracy of passively trained model: {passive_acc:.3f}")

plt.plot(accuracies * 100, label="Active learning")
plt.hlines(
    y=passive_acc * 100,
    xmin=0,
    xmax=learning_rounds,
    linestyles="dashed",
    label="Passive baseline",
)
plt.xlabel("Active learning rounds")
plt.ylabel("Accuracy [%]")
plt.legend()
plt.show()

# %% [markdown]
# Here, the accuracy is plotted as a function of completed active learning rounds.
# The dashed line represents the accuracy of the passively trained model, with the same
# number of datapoints as the active one after the last iteration.
# One can see that the accuracy of the actively trained model is larger at the end,
# and that the point where the actively trained model surpasses the passively trained
# model is pretty early, meaning that less data is needed to achieve the same
# model quality.

# %% [markdown]
# This concludes the active learning tutorial for classification,
# where we have seen that active learning improves data efficiency
# by ensuring that datapoints are chosen closer to the decision boundary
# to maximize their informativeness about the true function.
