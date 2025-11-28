import jax
import jax.numpy as jnp

from laplax.util.flatten import full_flatten


class FisherCase:

    def __init__(self, fn, data, params, loss):
        self.fn = fn
        self.data = data
        self.params = params
        self.loss = loss
        self.fisher_manual = self.fisher_manual()

    def fisher_manual(self):
        def df_dparams(input, params):
            jac = jax.jacfwd(self.fn, argnums=1)
            return jnp.atleast_2d(full_flatten(jac(input, params)))
        jacs = [df_dparams(x, self.params) for x in self.data["input"]]

        dLoss_df = jax.jacfwd(self.loss, argnums=0)
        grads = [
            jnp.atleast_2d(dLoss_df(self.fn(x, self.params), y))
            for x, y in zip(self.data["input"], self.data["target"], strict=True)
        ]
        
        fisher_manual = jnp.mean(jnp.array([
                jac.T @ grad @ grad.T @ jac 
                for jac, grad in zip(jacs, grads, strict=True)
            ]),axis=0,)
        return fisher_manual