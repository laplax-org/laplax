"""example of the marginal likelihood with four different models trained on the same sinoid dataset.
The models have different levels of complexity, where one is two simple one too complex and two are a good fit for the task.

The goal is to show the tradeoff between model complexity and performance and how the marginal likelihood
can give a good indication, if the model complexity fits the complexity of the task.
For each of the four models, the model gets plotted, the marginal likelihood gets calculated with three different
approaches and the loss gets calculated. On one hand this shows how the marginal likelihood functions can be used,
on the other hand it tries to show that the concept works and that it can sucessfully penalize overly complex
or simple models."""

from laplax.eval.marginal_likelihood import calculate_marginal_likelihood, marg_lik_with_hessian, calculate_marginal_likelihood_diagonal
from helper import get_sinusoid_examples_spaced
import matplotlib.pyplot as plt
from laplax.curv.ggn import create_ggn_mv
from laplax.util.flatten import create_pytree_flattener, wrap_function
import jax.numpy as jnp
import optax
from jax import random
from flax import nnx
from laplax.curv.cov import create_posterior_fn
import jax
import pickle


def create_model(rngs):
    model = model_class(rngs)
    return model


# generate training data
n_data = 150
sigma_noise = 0.2
batch_size = 32
rng_key = random.key(711)
loss_fn = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2)  # noqa: E731

X_train, y_train, train_loader, X_test = get_sinusoid_examples_spaced(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))

plt.scatter(X_train, jnp.sin(X_train), color='green', label='sin(x)')
plt.scatter(X_train, y_train)


# Too Simple: Cannot capture sine properly
class MLP_Simple(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 2, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(2, 1, rngs=rngs)
        self.learning_rate = 1e-2
        self.epochs = 1

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# Reasonable Complexity: Captures sine wave well
class MLP_Reasonable(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 10, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(10, 1, rngs=rngs)
        self.learning_rate = 1e-2
        self.epochs = 100

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# Good fit but complex: Better fit but also using a lot more parameters
class MLP_Good_Fit(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 50, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(50, 1, rngs=rngs)
        self.learning_rate = 6e-3
        self.epochs = 150

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# extremly overfitting: too complex
class MLP_Overfit(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 25, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(25, 50, rngs=rngs)
        self.linear3 = nnx.Linear(50, 25, rngs=rngs)
        self.linear4 = nnx.Linear(25, 1, rngs=rngs)
        self.learning_rate = 9e-4
        self.epochs = 500

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        return x


models = [("simple", MLP_Simple), ("reasonable", MLP_Reasonable), ("good_fit", MLP_Good_Fit), ("overfit", MLP_Overfit)]

for name, model_class in models:

    # Mean squared error loss function
    def mse_loss(model, X, y):
        pred_y = model(X)
        return jax.numpy.mean((y - pred_y) ** 2)

    # Update function using Optax
    def update(X, y, model, optimizer):
        loss_fn = lambda model: mse_loss(model, X, y)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

    # Training loop
    # Assuming `train_loader` is a generator that yields (X, y) batches
    def train_model(train_loader, rng_key):
        model = create_model(rng_key)
        n_epochs = model.epochs
        optimizer = nnx.Optimizer(model, optax.adamw(model.learning_rate))
        for i in range(n_epochs):
            for X, y in train_loader:
                update(X, y, model, optimizer)
                # You can log the loss here if you want
            if i % 10 == 0:
                pred = model(X_test)
                y_test = jnp.sin(X_test)
                test_loss = loss_fn(pred, y_test).item()
                pred_train = model(X_train)
                train_loss = loss_fn(pred_train, y_train).item()
                print("\n")
                print(f"epoch: {str(i)}")
                print("train loss:", str(train_loss))
                print("test loss:", str(test_loss))

        return model

    rng_key = nnx.Rngs(0)
    model = train_model(train_loader, rng_key)
    _, params, rest = nnx.split(model, nnx.Param, ...)

    data = {"input": X_train, "target": y_train}


    def model_fn(input, params):
        nnx.update(model, nnx.GraphState.merge(params, rest))  # Load the model from parameters\n",
        return model(input)

    full_fn = lambda p, data: loss_fn(model_fn(data["input"], p), data["target"])


    ggn_mv = create_ggn_mv(
        model_fn,
        params,
        data,
        loss_fn="mse",
        num_curv_samples=150,
        num_total_samples=75,
    )


    # Create Posterior
    posterior_fn = create_posterior_fn(
        "full",
        mv=ggn_mv,
        layout=params,
    )

    posterior_state = posterior_fn({"prior_prec": 1})

    pred = model(X_test)
    y_test = jnp.sin(X_test)
    test_loss = loss_fn(pred, y_test).item()


    pred_train = model(X_train)
    train_loss = loss_fn(pred_train, y_train).item()

    marginal_likelihood_full = calculate_marginal_likelihood(posterior_state, params, full_fn, data)
    marginal_likelihood_diagonal = calculate_marginal_likelihood_diagonal(posterior_state, params, full_fn, data)
    marginal_likelihood_hessian = marg_lik_with_hessian(params, full_fn, data)

    flatten, _ = create_pytree_flattener(params)
    num_params = len(flatten(params))
    print(f"{name} ({num_params} parameters)")
    print("likelihood_full:", marginal_likelihood_full)
    print("likelihood_diagonal:", marginal_likelihood_diagonal)
    print("likelihood_hessian:", marginal_likelihood_hessian)
    print("train loss:", str(train_loss))
    print("test loss:", str(test_loss))

    plt.scatter(X_train, y_train, color="green")
    plt.scatter(X_test, y_test, color='red')
    pred = pred.primal
    plt.scatter(X_test, pred, color="blue")
    plt.show()
