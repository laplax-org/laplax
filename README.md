<p align="center">
  <img src="./docs/images/laplax_logo.svg" width="541" height= "auto" alt="Laplax Logo"/>
</p>

---
[![Python
3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/release/python-3110/)
[![Test](https://github.com/laplax-org/laplax/actions/workflows/test.yaml/badge.svg)](https://github.com/laplax-org/laplax/actions/workflows/test.yaml) [![Docs](https://github.com/laplax-org/laplax/actions/workflows/docs.yaml/badge.svg)](https://github.com/laplax-org/laplax/actions/workflows/docs.yaml) [![Coverage Status](https://coveralls.io/repos/github/laplax-org/laplax/badge.svg?branch=laplax_api)](https://coveralls.io/github/laplax-org/laplax?branch=laplax_api)

## What is `laplax`?
The `laplax` package aims to provide a performant, minimal, and practical implementation of Laplace approximation techniques in [`jax`](https://github.com/google/jax). This package is designed to support a wide range of scientific libraries, initially focusing on compatibility with popular neural network libraries such as [`equinox`](https://github.com/patrick-kidger/equinox), [`flax.linen`](https://github.com/google/flax/tree/main/flax/linen), and [`flax.nnx`](https://github.com/google/flax/tree/main/flax/nnx). Our goal is to create a flexible tool for both practical applications and research, enabling rapid iteration and comparison of new approaches.

## Installation
Use `pip install laplax`.

## Documentation
Available at [https://laplax-org.github.io/laplax/](https://laplax-org.github.io/laplax/).

## Design Philosophy
The development of `laplax` is guided by the following principles:

- **Minimal Dependencies:** The package only depends on [`jax`](https://github.com/google/jax), ensuring compatibility and ease of integration.

- **Matrix-Vector Product Focus:** The core of our implementation revolves around efficient matrix-vector products. By passing around callables, we maintain a loose coupling between components, allowing for easy interaction with various other packages, including linear operator libraries in [`jax`](https://github.com/google/jax).

- **Performance and Practicality:** We prioritize a performant and minimal implementation that serves practical needs. The package offers a simple API for basic use cases while primarily serving as a reference implementation for researchers to compare new methods or iterate quickly over experiments.

- **PyTree-Centric Structure:** Internally, the package is structured around PyTrees. This design choice allows us to defer materialization until necessary, optimizing performance and memory usage.

## Roadmap and Contributions
We're developing this package in public, and discussions about the roadmap and feature priorities are structured in the [Issues](https://github.com/bmucsanyi/laplax/issues) section. If you're interested in contributing or want to see what's planned for the future, please check them out.
