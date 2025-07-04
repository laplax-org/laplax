import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from temp.kfac import kfac_blocks
from richard_workspace.dataloading import collect, minimnist
import optax

import matplotlib.pyplot as plt

class nnx_mlp(nnx.Module):
    def __init__(self, in_dim, mid_dim, out_dim, key):
        self.linear1 = nnx.Linear(in_dim, mid_dim, rngs=key)
        self.linear2 = nnx.Linear(mid_dim, mid_dim, rngs=key)
        self.linear3 = nnx.Linear(mid_dim, out_dim, rngs=key)

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x

class TestKFACBlocks(unittest.TestCase):
    def test_emp_fisher_equiv(self):
        """
            This tests for equivalence between the empirical fisher
            and KFAC-expand-empirical for a single training example as per
            Test-Case 1 of F. Dangel's KFAC tutorial (page 49ff).

            I'm still unsure why there isn't exact correlation between the two,
            i suspect precision errors since I'm working in different scales.
        """

        model = nnx_mlp(in_dim=28*28, mid_dim=10, out_dim=10, key=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001))
        trainloader, testloader, *_ = minimnist(batch_size=64, random_state=0)

        
        def ce_loss(model, x, y):
            logits = model(x)
            loss = -(y * jax.nn.log_softmax(logits)).mean()
            return loss
        
        @nnx.jit
        def step(model, optimizer, x, y):
            loss, grads = nnx.value_and_grad(ce_loss)(model, x, y)
            optimizer.update(grads)
            return loss

        # train model for 30 epochs
        for epoch in range(30):
            for x_batch, y_batch in trainloader:
                loss = step(model, optimizer, x_batch, y_batch)

        # has to be one sample, otherwise equivalance wont hold
        x, y = collect(trainloader, maxsamples=1)
        fisher_grads = jax.tree.leaves(
            nnx.grad(ce_loss)(model, x, y)
        )
        flat_fisher_grads = jnp.concat(jax.tree.map(
            lambda arr: arr.reshape(-1), fisher_grads
        ))
        FIM = jnp.outer(flat_fisher_grads, flat_fisher_grads)
        As, Bs = kfac_blocks(model, x, y)

        idx = 0
        _, axes = plt.subplots(nrows=len(As), ncols=2)

        for j ,a, b in zip(range(len(As)), As, Bs):
            
            block = jnp.kron(a, b)
            i = block.shape[0]
            true_block = FIM[idx : idx + i, idx : idx + i]

            # Compute correlation coefficient between flattened blocks
            block_flat = block.flatten()
            ggn_flat = true_block.flatten()
            corr = jnp.corrcoef(block_flat, ggn_flat)[0, 1]
            assert jnp.isclose(corr, 1.0, atol=1e-2), f"Block correlation {corr} not close to 1"
            idx += i

            axes[j][0].imshow(block[:20, :20])
            axes[j][1].imshow(true_block[:20, :20])
        plt.show()

"""
import matplotlib.pyplot as plt
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(ggn_block[:20, :20])
ax1.set_title('GGN Block')
ax2.imshow(block[:20, :20])
ax2.set_title('Empirical Fisher Block')
plt.show()

"""
if __name__ == '__main__':
    unittest.main()


