# Evaluation Module

The evaluation module provides functionality for propagating weight-space uncertainty to output-space predictions, calibrating probabilistic predictions on datasets, and evaluating various uncertainty metrics on datasets as well as the marginal log-likelihood.

## Pushforward Methods

Currently supported pushforward methods are:

- **Pushforward.LINEAR:**
  Propagates the weight-space covariance $\mathbf{H}^{-1}$ through the network using Jacobian-vector products $\mathcal{J}_{\theta}(f(x_n; \theta))$ to obtain a Gaussian predictive distribution in output space:

    $$
    \mathcal{N}\bigg(
        f(x_n; \theta^*),\
        \mathcal{J}_\theta(f(x_n; \theta^*))\ \mathbf{H}^{-1}\ \mathcal{J}_\theta(f(x_n; \theta^*))^\top
    \bigg)
    $$

- **Pushforward.NONLINEAR:**
  Draws samples from the weight-space posterior and passes them through the neural network to form an ensemble of predictions in output space. Empirical statistics (mean, variance, etc.) are then computed from this ensemble:

  $$
  f(x_n, \theta_s), \quad \theta_s \sim \mathcal{N}(\theta^*,\ \mathbf{H}^{-1})
  $$


## Predictives

For the classification case, the following predictives for pushing the uncertainty from the weight space to the output space are supported:

- `MC_BRIDGE`: Draw $z_s\sim\mathcal{N}\big(\mu,\Sigma\big)$, compute $p_s=\operatorname{softmax}(z_s)$ for $s=1\ldots S$, and form $\frac1S\sum_s p_s$.

- `LAPLACE_BRIDGE`: Transforms the Gaussian over logits into a Dirichlet by moment‑matching (“bridge”), yielding closed‑form Dirichlet parameters and thus an analytic predictive mean. The Laplace Bridge predictive approximates the true predictive as follows:
    \begin{equation}
        \mathbf{\hat p} := \frac{\frac{1}{\mathbf{\tilde\sigma}^2}\left(1-\frac{2}{C}+\frac{e^{\mathbf{\tilde\mu}}}{C^2}\sum_{c=1}^Ce^{-\tilde\mu_c}\right)}{\sum_{c=1}^C\frac{1}{\tilde\sigma^2_c}\left(1-\frac{2}{C}+\frac{e^{\mathbf{\tilde\mu}}}{C^2}\sum_{c'=1}^Ce^{-\tilde\mu_{c'}}\right)}
    \end{equation}
    where
    \begin{equation}
        \mathbf{\tilde\mu}^2 := \sqrt{\frac{\sqrt{C/2}} {\sum_{c=1}^C\sigma^2_c}}\mathbf\mu,\; \mathbf{\tilde\sigma^2} := \frac{\sqrt{C/2}} {\sum_{c=1}^C\sigma^2_c}\mathbf\sigma^2.
    \end{equation}

- `MEAN_FIELD_0_PREDICTIVE`:  A zeroth‐order mean‐field (probit‐style) approximation.

    $$
    \mathbb{E}[\operatorname{softmax}_i(\mathbf{z})]\approx\operatorname{softmax}_i\left(\frac{\mathbf{\mu}}{\sqrt{1+\lambda_0\,\operatorname{diag}(\mathbf{\Sigma}})}\right)
    $$

    which rescales each mean logit by its variance.

- `MEAN_FIELD_1_PREDICTIVE`: A first‐order pairwise approximation: for each $i$ we approximate $\Pr(z_i>z_j)$ under the bivariate Gaussian of $(z_i,z_j)$ and then normalize:

    $$
    \mathbb{E}[\operatorname{softmax}_i(\mathbf{z})]\approx \frac{1}{1 + \sum_{i \neq k} \exp \left( -\frac{(\mu_k - \mu_i)}{\sqrt{1 + \lambda_0 (\Sigma_{kk} + \Sigma_{ii})}} \right)}.
    $$

- `MEAN_FIELD_2_PREDICTIVE`:  A second‐order correction that incorporates full covariance: uses all bivariate variances and covariances in the exponentiated difference integrals:

    $$
    \mathbb{E}[\operatorname{softmax}_i(\mathbf{z})]\approx \frac{1}{1 + \sum_{i \neq k} \exp \left( -\frac{(\mu_k - \mu_i)}{\sqrt{1 + \lambda_0 (\Sigma_{kk} + \Sigma_{ii} - 2\Sigma_{ik})}} \right)}.
    $$

Each method trades off cost versus fidelity: sampling is asymptotically exact but can be slow; the mean‑field approximations incur only $O(C^2)$ or $O(C)$ work; and the Laplace bridge often gives the best calibrated probabilities when variances are large.


## Calibration

The module `laplax.eval.calibrate` provides functionality for calibrating the `prior_prec` parameter of posterior covariance using a grid search.


## Metrics

The submodule [`laplax.eval.metrics`](./eval/metrics.md) supports simple metrics for evaluating the uncertainty of the predictions, which support both classification and regression. The main evaluation functions can be simply extended to support other forms of evaluations.


## Main computational scaffold

The main pipeline functionality operates by mapping a list of functions over the predictions. This is done into rounds. The first round computes uncertainty related summaries such as mean, standard deviation, variance, covariance and/or samples; and the second round apply metrics to these predictions. Both lists of functions are applied iteratively making the results of previous functions available to the next functions, which allows for efficient shortcuts (e.g. computing the variance as the diagonal of the previously computed covariance matrix). To allow a flexible interface for operating on such results we provide the `apply_fns` and `transfer_entry` functions. These can be used in combination with `evaluate_on_dataset` and `evaluate_on_generator` to evaluate the uncertainty of the predictions on a dataset or generator, such as a DataLoader.
