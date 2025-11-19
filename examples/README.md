# Laplax Examples

This directory contains examples demonstrating the usage of laplax for Laplace approximations.

## Core Examples (from original laplax)

These examples demonstrate the basic functionality of laplax:

- **`0000_tiny_laplax.ipynb`**: Minimal introduction to laplax
- **`0001_laplax_for_regression.ipynb`**: Regression with Laplace approximation
- **`0002_laplax_on_mnist.ipynb`**: MNIST classification with Laplace approximation
- **`ex_classification.py`**: CIFAR-10 classification example
- **`ex_regression.py`**: Regression example
- **`ex_helper.py`**: Helper functions for examples

## FSP (Function-Space Prior) Examples

These examples demonstrate FSP Laplace approximations, which use function-space priors with GP kernels:

### Regression
- **`fsp_sin_regression.py`**: FSP Laplace on truncated sine regression
  - Demonstrates uncertainty quantification in extrapolation regions
  - Uses RBF kernel for function-space prior
  - Shows how uncertainty increases away from training data

### Classification
- **`fsp_two_moons_classification.ipynb`**: Interactive notebook for FSP vs Standard Laplace on two moons
  - Compares FSP Laplace with standard parameter-space Laplace
  - Uses zero-noise two moons dataset for clean visualization
  - Shows S-shaped decision boundaries and uncertainty estimates

## Running the Examples

### Python Scripts
```bash
python examples/fsp_sin_regression.py
python examples/ex_classification.py
python examples/ex_regression.py
```

### Jupyter Notebooks
```bash
jupyter notebook examples/
```

Then open and run the desired `.ipynb` file.

## Requirements

Make sure you have laplax installed with all dependencies:
```bash
pip install -e .
pip install flax optax matplotlib scikit-learn jupyter
```
