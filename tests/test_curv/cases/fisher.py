import jax
import jax.numpy as jnp

from laplax.util.flatten import full_flatten


class FisherCase:

    def __init__(self, n, o, i, l, p, fn, data, params, loss):
        self.n = n
        self.o = o
        self.i = i
        self.l = l
        self.p = p
        print(self.o)
        def safe_fn(input, params):
            assert input.shape == (n,i) or input.shape == (1,i)
            assert params.shape == (p,)
            output = fn(input, params)
            print(f"Assert: {output.shape}")
            assert output.shape == (input.shape[0], o)
            return output

        self.fn = safe_fn
        self.data = data
        assert data["input"].shape == (n,i)
        assert data["target"].shape == (n,l)
        self.params = params
        assert params.shape == (p,)
        def safe_loss(fn, y):
            assert fn.shape == (n,o) or fn.shape == (1,o)
            assert y.shape == (n,l) or y.shape == (1,l)
            output = loss(fn, y)
            assert output.shape == (fn.shape[0],)
            return output
        self.loss = safe_loss
        self.fisher_manual = self.fisher_manual()

    def fisher_manual(self):
        def df_dparams(input, params):
            jac = jax.jacfwd(self.fn, argnums=1)
            return jnp.reshape(full_flatten(jac(input, params)), shape=(self.o, self.p))
        jacs = [df_dparams(jnp.atleast_2d(x), self.params) for x in self.data["input"]]
        print("jacs[0].shape =?= (o,p): ", jacs[0].shape)

        dLoss_df = jax.jacfwd(self.loss, argnums=0)
        grads = [
            dLoss_df(self.fn(jnp.atleast_2d(x), self.params), jnp.atleast_2d(y))
            for x, y in zip(self.data["input"], self.data["target"], strict=True)
        ]
        print("grads[0].shape =?= (1,o): ", grads[0].shape)
        
        fisher_manual = jnp.mean(jnp.array([
                jac.T @ 
                grad @ 
                grad.T @ 
                jac 
                for jac, grad in zip(jacs, grads, strict=True)
            ]),axis=0,)
        return fisher_manual

    def construct_fisher(self, fisher_mv):
        # Construct full matrix via mvp with one-hot vectors
        fisher_rows = [full_flatten(fisher_mv(row)) for row in jnp.eye(len(self.params))]
        return jnp.stack(fisher_rows)
