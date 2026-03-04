## FSP-Laplace in `laplax`: practical workflow and examples

In addition to the standard weight-space Laplace approximation, `laplax` supports **FSP-Laplace** (*Function-Space Priors for Laplace*). The key difference is that the prior is specified **in function space** (as a GP), and it is enforced during training via an **RKHS regulariser** evaluated at **context points**.

We provide two examples in the same “notebook-style” format as the standard Laplace regression tutorial:

- **Regression:** `examples/0003_laplax_fsp_laplace_regression.py` (and `.ipynb`)
- **Classification (two moons):** `examples/0004_laplax_fsp_laplace_classification.py` (and `.ipynb`)

These notebooks focus on the core `laplax` steps; plotting and general utility code is kept in helper/plotting modules.

---

### 1) Function-space prior: GP, context set, and Gram matrix

FSP-Laplace starts by placing a GP prior on the latent function
$$
f \sim \mathcal{GP}(m, k).
$$

You choose
- a kernel function $k(\cdot,\cdot)$ (e.g. Matérn / periodic),
- context points $X_C = \{x_j^{(c)}\}_{j=1}^M$,
- optionally a prior mean function $m(\cdot)$ (often $m \equiv 0$).

On the context set we form the Gram matrix
$$
K_{CC} := k(X_C, X_C) \in \mathbb{R}^{M\times M},
$$
and for numerical stability we compute a Cholesky factor with jitter:
$$
L_C L_C^\top = K_{CC} + \epsilon I.
$$

---

### 2) FSP training: likelihood + RKHS energy

To make the function prior operational, FSP-Laplace augments training with an RKHS penalty.
With
- $f(C) := f_\theta(X_C)$ and
- $m(C) := m(X_C)$,

the RKHS energy (for a scalar output) is approximated by
$$
\| f_\theta - m \|_{\mathcal{H}}^2 \;\approx\; (f(C) - m(C))^\top K_{CC}^{-1} (f(C) - m(C)).
$$

The resulting FSP objective (up to scaling/constants) is
$$
\mathcal{L}_{\text{FSP}}(\theta)
=
-\sum_{n=1}^N \log p(y_n \mid f_\theta(x_n))
+
\frac{1}{2}\,\tau \, \| f_\theta - m \|_{\mathcal{H}}^2,
$$
where $\tau$ is the **function-space prior precision**.

**Implementation in `laplax`:**
- the RKHS energy is computed stably via Cholesky solves using `prior_cov_chol=L_C`,
- likelihood terms are typically scaled to the full dataset size,
- a combined objective adds the weighted RKHS penalty to the likelihood.

Intuitively, context points “tether” the network to the GP prior even away from the training inputs, shaping extrapolation behaviour.

---

### 3) FSP-Laplace posterior: Laplace around the FSP optimum

After training, FSP-Laplace builds a Laplace approximation around $\theta^*$, but with curvature that reflects the RKHS term evaluated via the context set.
In `laplax`, the main entry point is:

- `laplax.curv.fsp.create_fsp_posterior(...)`

Conceptually, we again obtain a Gaussian approximation in weight space,
$$
p(\theta\mid \mathcal{D}) \approx \mathcal{N}(\theta^*, \Sigma),
$$
where $\Sigma$ is represented **matrix-free** via MVPs and can optionally use low-rank structure.

**Regression note (Gaussian likelihood):**  
If the observation noise variance is $\sigma^2$, the data curvature is typically scaled by
$$
\text{GGN factor} \;\propto\; \frac{1}{\sigma^2}.
$$
In the regression example this is passed explicitly via a parameter such as `ggn_factor=1.0/(sigma_map**2)`.

---

### 4) Pushforward: linearised predictive uncertainty

As in standard Laplace, we push weight-space uncertainty to output space, often via linearisation:
$$
f_{\theta^{\text{lin}}}(x)
=
f_{\theta^*}(x) + J_{\theta^*}(x)\,(\theta-\theta^*).
$$

This yields an approximate Gaussian predictive in output space
$$
f(x) \sim \mathcal{N}\Bigl(\mu(x), \; J_{\theta^*}(x)\,\Sigma\,J_{\theta^*}(x)^\top \Bigr).
$$

In `laplax`, this is done via:

- `laplax.eval.pushforward.set_lin_pushforward(...)`

**Classification:**  
The approximation is analytic in **logit space**, but the predictive probability $p(y\mid x)$ typically requires an approximation (e.g. sigmoid-/softmax-moment or Monte Carlo).

---

### 5) What the examples demonstrate

#### FSP-Laplace regression (`examples/0003_laplax_fsp_laplace_regression.py`)
- dataset: **truncated sinusoid** (training on sub-intervals + extrapolation region)
- kernel: e.g. **periodic** (encodes periodic structure)
- context points: cover the region of interest (including extrapolation)
- pipeline:
  1) FSP training (likelihood + RKHS)
  2) `create_fsp_posterior(...)`
  3) `set_lin_pushforward(...)`
  4) evaluate predictive mean and uncertainty on a grid

#### FSP-Laplace classification (`examples/0004_laplax_fsp_laplace_classification.py`)
- dataset: **two moons**
- kernel: e.g. **Matérn 5/2** (smooth decision boundaries)
- context points: 2D grid over the region of interest
- pipeline:
  1) FSP training (BCE + RKHS on logits)
  2) `create_fsp_posterior(..., loss_fn=BINARY_CROSS_ENTROPY, ...)`
  3) `set_lin_pushforward(...)`
  4) compute $p(y=1\mid x)$ and e.g. **predictive entropy** as an uncertainty diagnostic

---

### Practical tips

- Since FSP is using the RKHS regularizer, we recommend not training additionally with any $L^2$ regularization such as implicitly by AdamW.
- Context points are a modelling tool: place them where you care about extrapolation or boundary behaviour.
- Be explicit about **scaling**:
  - dataset-size scaling of likelihood terms,
  - the RKHS regulariser weight (often $\tfrac{1}{2}\tau$),
  - and regression curvature scaling via $1/\sigma^2$ when modelling observation noise.
- For large models, prefer **low-rank** methods (Lanczos/Lobpcg); increasing $M=|C|$ strengthens prior conditioning but increases cost.
