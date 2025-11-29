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
            if self.handle_batches == False:
                assert input.shape == (i,)
                output = fn(input, params)
                assert output.shape == (o,)
            else:
                assert input.shape == (n,i)
                output = jax.vmap(fn)(input)
                assert output.shape == (n,o)
            return output
            

        self.fn = optionally_batched_fn
        self.data = data
        assert data["input"].shape == (n,i)
        assert data["target"].shape == (n,l)
        self.params = params
        assert params.shape == (p,)
        print("o: ", self.o)
        def optionally_batched_loss(fn, y):
            if self.handle_batches == False:
                assert fn.shape == (o,)
                assert y.shape == (l,)
                output = loss(fn, y)
                #print(output)
                #assert output.shape == (1,)
            else:
                assert fn.shape == (n,o)
                assert y.shape == (n,l)
                output = jax.vmap(loss)(fn, y)
                #assert output.shape == (n,)
            return output

        self.loss = optionally_batched_loss
        self.fisher_manual = self.fisher_manual()

    def fisher_manual(self):
        def df_dparams(input, params):
            jac = jax.jacfwd(self.fn, argnums=1)
            return jnp.reshape(full_flatten(jac(input, params)), shape=(self.o, self.p))
        jacs = [df_dparams(x, self.params) for x in self.data["input"]]
        print("jacs[0].shape =?= (o,p): ", jacs[0].shape)

        dLoss_df = jax.grad(self.loss, argnums=0)
        grads = [
            dLoss_df(self.fn(x, self.params), y)[:,None]
            for x, y in zip(self.data["input"], self.data["target"], strict=True)
        ]
        print("grads[0].shape =?= (o,1): ", grads[0].shape)
        
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
