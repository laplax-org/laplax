# Design Philosophy of `laplax`
The development of `laplax` is guided by the following principles:

- **Minimal Dependencies:** The package only depends on [`jax`](https://github.com/google/jax), ensuring compatibility and ease of integration.

- **Matrix-Free Linear Algebra:** The core of our implementation revolves around efficient matrix-vector products. By passing around Python `Callable` objects, we maintain a loose coupling between components, allowing for easy interaction with various other packages, including linear operator libraries in [`jax`](https://github.com/google/jax).

- **Performance and Practicality:** We prioritize a performant and minimal implementation that serves practical needs. The package offers a simple API for basic use cases while primarily serving as a reference implementation for researchers to compare new methods or iterate quickly over experiments.

- **PyTree-Centric Structure:** Internally, the package is structured around PyTrees. This design choice allows us to defer materialization until necessary, optimizing performance and memory usage.

## Roadmap and Contributions
We're developing this package in public. The roadmap and feature priorities are discussed in the [Issues](https://github.com/laplax-org/laplax/issues) of the repository. If you're interested in contributing or want to see what's planned for the future, please check them out.
