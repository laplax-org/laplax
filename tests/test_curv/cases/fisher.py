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
        self.handle_batches = False

        def optionally_batched_fn(input, params):
            assert params.shape == (p,)
            if not self.handle_batches:
                assert input.shape == (i,)
                output = fn(input, params)
                assert output.shape == (o,)
            else:
                assert input.shape == (n, i)
                output = jax.vmap(lambda x: fn(x, params))(input)
                assert output.shape == (n, o)
            return output

        self.fn = optionally_batched_fn
        self.data = data
        assert data["input"].shape == (n, i)
        assert data["target"].shape == (n, l)
        self.params = params
        assert params.shape == (p,)

        self.loss = loss
        self.fisher_manual = self.fisher_manual()

    def fisher_manual(self):
        def df_dparams(input, params):
            jac = jax.jacfwd(self.fn, argnums=1)
            return jnp.reshape(full_flatten(jac(input, params)), shape=(self.o, self.p))

        jacs = [df_dparams(x, self.params) for x in self.data["input"]]

        dLoss_df = jax.grad(self.loss, argnums=0)
        grads = [
            dLoss_df(self.fn(x, self.params), y)[:, None]
            for x, y in zip(self.data["input"], self.data["target"], strict=True)
        ]

        fisher_manual = jnp.mean(
            jnp.array([
                jac.T @ grad @ grad.T @ jac
                for jac, grad in zip(jacs, grads, strict=True)
            ]),
            axis=0,
        )
        return fisher_manual

    def construct_fisher(self, fisher_mv):
        # Construct full matrix via mvp with one-hot vectors
        eye = jnp.eye(len(self.params))
        fisher_rows = [full_flatten(fisher_mv(row)) for row in eye]
        return jnp.stack(fisher_rows)
