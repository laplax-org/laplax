# Curvature Module

## Curvatures

Currently supported curvature-vector products are:

- **GGN-mv (Generalized Gauss-Newton):**

    $$
    v \mapsto \sum_{n=1}^{N} \mathcal{J}_\theta^\top(f_{\theta^*}(x_n)) \nabla^2_{f_{\theta^*}(x_n),f_{\theta^*}(x_n)} \ell(f_\theta(x_n), y_n) \mathcal{J}_\theta(f_{\theta^*})\, v
    $$

- **Hessian-mv (Hessian):**
    $$
    v \mapsto \sum_{n=1}^{N} \nabla_{\theta \theta}^2 \ell(f_\theta(x_n), y_n)\,v
    $$

## Curvature estimators/approximations

For both curvature-vector products, the following methods are supported for approximating and transforming them into a weight space covariance matrix-vector product:

- `CurvApprox.FULL` denses the curvature-vector product into a full matrix. The posterior function is then given by

    $$
    (\tau, \mathcal{C}) \mapsto \left[ v \mapsto \left(\textbf{Curv}(\mathcal{C}) + \tau I \right)^{-1} v \right].
    $$

- `CurvApprox.DIAGONAL` approximates the curvature using only its diagonal, obtained by evaluating the curvature-vector product with standard basis vectors from both sides. This leads to:

    $$
    (\tau, \mathcal{C}) \mapsto \left[ v \mapsto \left(\text{diag}(\textbf{Curv}(\mathcal{C}) + \tau I \right)^{-1}v  \right].
    $$

- **Low-Rank** employs either a custom Lanczos routine (`CurvApprox.LANCZOS`) or a variant of the LOBPCG algorithm (`CurvApprox.LOBPCG`). These methods approximate the top eigenvectors $U$ and eigenvalues $S$ of the curvature via matrix-vector products. The posterior is then given by a low-rank plus scaled diagonal:

    $$
    (\tau, \mathcal{C}) \mapsto \left[ v \mapsto \left(\big[U S U^\top\big](\mathcal{C}) + \tau I \right)^{-1} v \right].
    $$

## Main computational scaffold

This pipeline is controlled via the following three functions:

- `laplax.curv.estimate_curvature`: Estimates the curvature based on the provided type and curvature-vector-product.

- `laplax.curv.set_posterior_fn`: Takes an estimated curvature and returns a function that maps `prior_arguments` to the posterior.

- `laplax.curv.create_posterior_fn`: Combines the `estimate_curvature` and `set_posterior_fn`.

### laplax.curv.estimate_curvature
::: laplax.curv.estimate_curvature

### laplax.curv.set_posterior_fn
::: laplax.curv.set_posterior_fn

### laplax.curv.create_posterior_fn
::: laplax.curv.create_posterior_fn
