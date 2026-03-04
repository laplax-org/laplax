# Turning neural networks Bayesian

## Probabilistic perspective on supervised learning

Given labelled training data $\mathcal{D} = \{(x_n, y_n)\}_{n=1}^N$, loss function $\ell$ and regularizer $\Omega$, the parameters $\theta$ of a neural network $f_\theta$ are typically obtained by minimising the regularised empirical risk $\mathcal{L}(\mathcal{D}, f_\theta)$.
From a probabilistic perspective, this procedure corresponds to finding a maximum a posteriori (MAP) estimate of the weights under a likelihood and prior.
Formally, both views lead to the following optimisation problem:

$$
\begin{aligned}
\theta^* &= \operatorname*{arg\,min}_{\theta} \mathcal{L} (\theta; \mathcal{D}) \\
         &= \operatorname*{arg\,min}_{\theta} \underbrace{\sum_{n=1}^{N} \ell(f_\theta(x_n), y_n) + \Omega(\theta)}_{\mathcal{L}(\mathcal{D}, f_\theta)} \\
         &= \operatorname*{arg\,max}_{\theta} \sum_{n=1}^{N} \log p(y_n \vert f_\theta(x_n)) + \log p(\theta)\, .
\end{aligned}
$$

The weight-space uncertainty is then described by the posterior distribution given the training data:

$$
p(\theta \vert \mathcal{D}) = \frac{ p(\mathcal{D} \vert \theta)\, p(\theta)}{\int p(\mathcal{D} \vert \theta)\,p(\theta)\,d\theta}\; .
$$

However, for deep neural networks, the integral in the denominator is usually intractable. The Laplace approximation circumvents this by utilising a Gaussian distribution to approximate the posterior.
To this end, we apply a second-order Taylor approximation to the negative log-posterior loss $\mathcal{L}$ around the MAP estimate $\theta^*$, which yields

$$
\mathcal{L}(\mathcal{D}, f_\theta) \approx \mathcal{L}(\mathcal{D}, f_{\theta^*}) + \nabla_\theta \mathcal{L}(\mathcal{D}, f_{\theta^*})^\top (\theta - \theta^*) + \frac{1}{2} (\theta - \theta^*)^\top\nabla^2_{\theta \theta} \mathcal{L}(\mathcal{D}, f_{\theta^*}) (\theta - \theta^*),
$$

where the first-order term vanishes due to the assumed local optimality of $\theta^*$. Negation and exponentiation yield

$$
p(\theta \vert \mathcal{D}) \approx \mathcal{N}\Bigl(\theta^*, \mathbf{H}(\mathcal{D}, f_{\theta^*})^{-1}\Bigr)
$$

with $\mathbf{H} = \nabla^2_{\theta \theta} \mathcal{L}(\mathcal{D}, f_{\theta^*})$ being the posterior precision.


## Linearised Laplace approximation

To obtain predictive uncertainty estimates, the weight space uncertainty is pushed forward into the neural network's output space. This is either done via sampling a set of $S$ weights from the approximate posterior and using these in the neural network forward pass to obtain $S$ predictions, or by **linearising** the network around the MAP estimate as

$$
f_{\theta^{\text{lin}}}(\cdot, \theta) = f_{\theta^*}(\cdot, \theta^*) + \mathcal{J}_{\theta^*}(\cdot)(\theta - \theta^*)
$$

and using the linear closure of Gaussian distributions[@immer_improving_2021], yielding **closed-form** output-space uncertainty.
[^1]: For classification, the logit-space uncertainty is analytic, but the predictive distribution has to be approximated, e.g., through Monte Carlo sampling and averaging the softmax probabilities.

The linearised approach is guaranteed to yield positive-definite weight-space covariance matrices for a strictly convex regulariser $\Omega$ at any weight configuration $\theta$, not just at MAP estimates (that are hard to obtain exactly in deep learning settings).
Usually, further approximations are needed to reduce the computational and memory requirements of the curvature. These are discussed in [Curvature](reference/curv.md).

## Marginal log-likelihood

An important Bayesian tool for model selection is the marginal log-likelihood given by

$$
\log p(\mathcal{D} \vert \mathcal{M}) \approx \log p(\mathcal{D}, \theta^* \vert \mathcal{M}) - \frac{1}{2} \log \left\vert \frac{1}{2\pi} \mathbf{H}(\mathcal{D}, f_{\theta^*}) \right\vert.
$$

This term is often used for the selection of the model hyperparameters $\mathcal{M}$ via maximization
[@immer_scalable_2021], since it represents an analytic trade-off between **complexity** and **expressivity** -- the so-called Occam's razor [@rasmussen2000occam]. Tractability and scalability depend on the structure of the estimated $\mathbf{H}(\mathcal{D}, f_{\theta^*})$, but compared to the predictive uncertainty above (cf. [Weight posterior](reference/curv.md)), no inversion is needed.



# FSP-Laplace: Function-Space Priors for the Laplace Approximation in Bayesian Deep Learning

In some applications you may want to impose a prior not only on weights $\theta$, but directly on the *function* $f_\theta$. **FSP-Laplace** does exactly this by placing a **Gaussian process (GP) prior in function space** and deriving a Laplace approximation that reflects this inductive bias. This makes it possible to encode interpretable structure—e.g. smoothness, periodicity, or boundary behavior—while retaining the practical scalability of Laplace methods [@cinquin_fsp_laplace_2024].

A key conceptual point is that in infinite-dimensional function spaces one cannot write ordinary Lebesgue densities. FSP-Laplace therefore recasts training as finding a “weak mode” of the posterior measure under a GP prior restricted to the neural-network function class [@cinquin_fsp_laplace_2024].

## Function-space prior

Choose a GP prior on the latent function:
$$
f \sim \mathcal{GP}(m, k),
$$
with mean function $m(\cdot)$ and kernel $k(\cdot,\cdot)$. The kernel is where you encode your prior belief (RBF/Matérn for smoothness, periodic kernels for periodic structure, etc.) [@cinquin_fsp_laplace_2024].

## RKHS regulariser (required for FSP training)

To make the function prior operational during training, FSP-Laplace adds an RKHS penalty induced by the GP kernel. In practice, this is implemented via a finite set of **context points** $X_C = \{x^{(c)}_j\}_{j=1}^{M}$, where the model is “tethered” to the prior even outside the training inputs [@cinquin_fsp_laplace_2024].

Let

* $f(C) := f_\theta(X_C) \in \mathbb{R}^{M\times D}$ be the network outputs at context points,
* $m(C) := m(X_C) \in \mathbb{R}^{M\times D}$,
* $K(C, C) := k(X_C, X_C) \in \mathbb{R}^{M\times M}$.

Then the **squared RKHS norm** is approximated by the quadratic form
$$
\| f_\theta - m \|_{\mathcal{H}}^2
\approx\sum_{d=1}^{D}
\left(f_{d}(C) - m_{d}(C)\right)^\top
K_{CC}^{-1}
\left(f_{d}(C) - m_{d}(C)\right),
$$
where we sum over output dimensions $d$ if outputs are treated as conditionally independent given the same kernel.

With a likelihood term $\log p(y_n \mid f_\theta(x_n))$, the **FSP objective** becomes:
$$
\mathcal{L}_{\text{FSP}}(\theta) = -\sum_{n=1}^{N}\log p\left(y_n \mid f_\theta(x_n)\right)
+
\frac{1}{2}\| f_\theta - m \|_{\mathcal{H}}^2.
$$
This is the training objective you need for FSP-Laplace to “condition well” on the chosen function prior [@cinquin_fsp_laplace_2024].

> **Implementation note (laplax):** this is exactly what `laplax.util.objective.compute_rkhs_norm(...)` computes via a stable Cholesky solve (with jitter), and `n_gaussian_log_posterior_objective(...)` combines it with a Gaussian log-likelihood (scaled to the full dataset size).

## Why the posterior has a low-rank structure

After training, FSP-Laplace applies a Laplace approximation around the obtained solution. The crucial computational observation is that the function-space prior enters through the context set $X_C$. Under linearisation, the curvature contribution from the RKHS term depends on Jacobians evaluated at $X_C$, which induces a **structured (often effectively low-rank) update** in weight space whose “rank” is controlled by the number of context points (and outputs). This is what enables scalable matrix-free implementations [@cinquin_fsp_laplace_2024].

> **Connection to `laplax.curv.cov.Posterior`:** the `Posterior` dataclass exposes `cov_mv`/`scale_mv` for matrix-vector products and optionally stores `low_rank_terms` and `rank` of the posterior. This matches the way FSP-Laplace can be represented and applied without ever materialising a full dense covariance.

## Context points in laplax

Because context points are central to both *conditioning* and *scalability*, laplax provides utilities in `laplax.util.context_points.py` to generate them from data using:

* **space-filling low-discrepancy sequences** (Sobol/Halton/LHS),
* **PCA-based space filling** (fit PCA on the data, sample in score space, then inverse-transform),
* **random bounding-box sampling**.

These strategies are helpful when you want the prior to constrain behavior:

* slightly **outside** the empirical training distribution,
* on **boundaries** (e.g. PDE domains),
* or in **regions of interest** where extrapolation matters.

