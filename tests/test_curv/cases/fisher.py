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
        o = self.fn(self.data["input"][0], self.params).shape[0]
        def df_dparams(input, params):
            jac = jax.jacfwd(self.fn, argnums=1)
            return jnp.reshape(full_flatten(jac(input, params)), shape=(o,-1))
        jacs = [df_dparams(x, self.params) for x in self.data["input"]]
        dLoss_df = jax.grad(self.loss, argnums=0)
        grads = [
            dLoss_df(self.fn(x, self.params), y)[:,None]
            for x, y in zip(self.data["input"], self.data["target"], strict=True)
        ]
        
        fisher_manual = jnp.mean(jnp.array([
                jac.T @ 
                grad @ 
                grad.T @ 
                jac 
                for jac, grad in zip(jacs, grads, strict=True)
            ]),axis=0,)
        return fisher_manual