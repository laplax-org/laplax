# Utility Module

The `util` module provides utility functions for the `laplax` package, that might be useful for working with JAX-based libraries in general. Here we list its submodules with a short description.

## `flatten.py`
This module provides utilities for working with JAX PyTrees, primarily focused on flattening them into 1D or 2D arrays and unflattening arrays back into PyTrees. It includes functions for creating flattener/unflattener pairs and for wrapping other functions to automatically handle these data structure transformations.

## `mv.py`
This module provides matrix-free operations, specifically for handling matrix-vector products (MVPs). It offers functions to compute the diagonal, or densify both an explicit matrix or an abstract linear operator defined by a function.

## `loader.py`
This module offers tools for processing data in batches from data loaders. It includes various reduction functions (like sum, concatenation, and online mean) to aggregate results across batches. The central `DataLoaderMV` class allows matrix-vector product functions to be applied over entire datasets by iterating through a data loader, making it possible to compute dataset-wide quantities like diagonals or dense matrices from batch-level operations.

## `ops.py`
This module contains general-purpose adaptive operations. Its main feature is `precompute_list`, a function that can optionally precompute and cache the results of applying a function to a sequence of items instead of recomputing them each time.

## `tree.py`
This module provides a comprehensive suite of utility functions for manipulating JAX PyTrees. It includes element-wise arithmetic and statistical operations, functions for creating structured PyTrees (e.g., of zeros, ones, or random numbers), utilities for linear algebra operations like matrix-vector products with PyTrees, and tools for creating basis vectors and identity-like structures within a PyTree format.
