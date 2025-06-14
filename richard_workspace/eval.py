import jax
import jaxtyping
import jax.numpy as jnp

from flax import nnx
from typing import *
from richard_workspace.dataloading import *

def eval_model(model, _testloader, seeds):
    per_seed_acc = []
    for seed in seeds:
        testloader = permute(_testloader, seed)
        y_preds_test = []
        for x_batch, y_batch in testloader:
            y_preds_test.append(accuracy(model, x_batch, y_batch))
        y_preds_test = jnp.stack(y_preds_test)
        accuracy_value = jnp.mean(y_preds_test)
        per_seed_acc.append(accuracy_value)
    return per_seed_acc

def flat(leaves : List[jaxtyping.Array]):
    return jnp.concat([arr.reshape(-1) for arr in leaves])

def accuracy(model, x, y):
    log_y_pred = jax.vmap(model)(x)
    y_pred = jnp.argmax(log_y_pred, axis=-1)
    y_true = jnp.argmax(y, axis=-1)
    return sum(y_pred == y_true).astype(int) / len(y)

def d2name(kwds):
    return "--".join([key + ':' + str(value) for key, value in kwds.items()])