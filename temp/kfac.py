import jax
import jax.numpy as jnp; import jax.random as jr;
from flax import nnx
from temp.intergrad import intergrad

"""
    KFAC from scratch.
    
    1) Can I construct the KFAC matrix?
        1.1) get gradients and activations got single example
        1.2) calculate the block matrices
        1.3) yield back mv-mult function.
            
    2) Make sanity checks. Visualize block-diagonal matrix approximation.
"""


if __name__ == '__main__':
    pass