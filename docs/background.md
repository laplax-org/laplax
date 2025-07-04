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
p(\theta \vert \mathcal{D}) \approx \mathcal{N}\Bigl(\theta^*, \mathbf{H}(\mathcal{D}, f_{\theta^*})^{-\frac{1}{2}}\Bigr)
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
