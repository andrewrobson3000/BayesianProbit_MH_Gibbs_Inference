# Bayesian Inference for a Probit Model: Implementation and Comparison of MCMC Techniques

## Abstract

A probit model is a statistical model used to analyze binary outcome data. In this report, I investigate Bayesian inference for a probit model applied to a simulated dataset with known parameters. I began by visualizing the posterior density of the coefficients and implemented two Markov Chain Monte Carlo (MCMC) sampling algorithms: Metropolis-Hastings and Gibbs sampling. Finally, I evaluated their performance using metrics such as acceptance rate and autocorrelation.

## Methodology

### Data Generation

I generated a synthetic dataset (size: $n$, explanatory variables: $p$) according to a probit model:

$$
Y_i \sim \text{Bernoulli}(\Phi(X_i^T \beta))
$$

where $Y_i$ represents the binary outcome (0 or 1) for the $i$-th observation, $X_i$ is the $i$-th row of the explanatory variable matrix ($n \times p$), $\beta$ is the coefficient vector ($p \times 1$), and $\Phi$ is the standard normal cumulative distribution function (CDF). The true parameters used for simulation were denoted by $\beta_{true}$.

### Prior Specification

I adopted a Gaussian prior distribution on the coefficients, centered at zero with a covariance matrix $B$:

$$
\pi(\beta) \propto N(\beta \mid 0, B)
$$

This choice encodes a degree of regularization and a prior belief that the coefficients are likely centered around zero.

### Posterior Exploration

To understand the posterior distribution, I computed the log-posterior density function:

$$
\log \pi(\beta \mid Y, X) \propto \log \pi(Y \mid X, \beta) + \log \pi(\beta)
$$

where $\pi(Y \mid X, \beta)$ represents the data likelihood given the model and parameters. Due to the complexity of calculating the posterior directly, I visualized it on a grid to explore different parameter combinations.

## Metropolis-Hastings

This MCMC method iteratively proposes new parameter values, assesses their fit to the data and prior, and uses an acceptance probability to determine whether to include them in the sampled chain. My implementation typically used a Gaussian proposal distribution with a small variance.

## Gibbs Sampling

This method relies on iteratively sampling from the conditional distributions of each variable given the others. I introduced a latent variable $Z_i$ for each observation, distributed normally with mean $X_i^T \beta$ and variance 1:

$$
Z_i \sim N(X_i^T \beta, 1)
$$

Leveraging conjugacy relationships, I derived tractable conditional distributions for Gibbs sampling:

- Sample $\beta$ given $Z$ and $Y$:

$$
\beta \sim N((B^{-1} + X^T X)^{-1} X^T Z, (B^{-1} + X^T X)^{-1})
$$

- Sample $Z_i$ given $\beta$ and $Y$:

$$
Z_i \sim \text{TruncatedNormal}(\text{bounds based on } Y_i; \text{mean}=X_i^T \beta, \text{variance}=1)
$$

## Results

### Posterior Visualization

The visualization of the posterior density confirmed a region of high probability containing the "true" parameters. This indicates the model's ability to recover the underlying data-generating process.

### Metropolis-Hastings

The MH sampler achieved an acceptance rate of approximately 15%. While some exploration occurred, a higher acceptance rate might suggest better efficiency.

### Gibbs Sampling

The Gibbs sampler exhibited lower autocorrelation than Metropolis-Hastings, implying it moved through the posterior distribution more effectively.

## Discussion

The Gibbs sampler's superior performance in this analysis highlights the power of exploiting conjugacy relationships. By introducing the latent variable $Z$, I was able to derive convenient conditional distributions that led to efficient sampling.

The Metropolis-Hastings algorithm's low acceptance rate suggests potential for improvement. Refining the proposal distribution, possibly by adjusting its variance, could lead to better exploration of the posterior space.

## Future Work

- Explore the use of Hamiltonian Monte Carlo, which can be more efficient in high-dimensional problems, potentially handling larger numbers of explanatory variables.
- Investigate the effects of strongly informative priors and weakly informative priors on the posterior distribution and parameter estimates.

## Code Elaboration

Let me walk you through how my code works to implement this Bayesian probit analysis:

### Data Generation

`n, p = 50, 2`: I set up my simulated dataset with 50 observations and 2 explanatory variables.

`beta_true = np.array([2, -1]).reshape(-1, 1)`: I established the "true" coefficients that I wanted to see if my model could recover. Here, one variable has a positive influence on the outcome, and the other has a negative influence.

Lines involving `X` and `Y`: I built the explanatory variable matrix (`X`) and used the probit model with the true coefficients to simulate the binary outcome variable (`Y`).

### Prior Specification

`def log_prior(beta) ...`: I defined my prior belief about the coefficients (`beta`). I used a Gaussian prior centered at zero, which means I assumed the coefficients were likely to be near zero. The covariance matrix `B` allowed me to adjust how strongly I held this belief.

### Posterior Calculation

`def full_log_posterior(beta) ...`: I created a function to calculate the log of the posterior density. This is where the magic happens – I combined the likelihood of my data given the model and parameters (prob_1, prob_0) with my prior belief (`log_prior`).

### Metropolis-Hastings Implementation

`def metropolis_hastings() ...`: I implemented the MH algorithm. Here's the gist of how it works in my code:

- I propose new candidate values for the parameters.
- I calculate how well these new parameters fit the data and my prior.
- Based on an acceptance probability, I decide whether to include the new candidate values in the chain that represents the posterior.

### Gibbs Sampling Implementation

`def sample_z_given_beta(beta) ...` and `def sample_beta_given_z(z)`: I implemented functions to sample the latent variable `Z` and the coefficients (`beta`) according to their conditional distributions. The use of `truncnorm` was key to correctly modeling the relationship between `Z` and my binary outcome `Y`.

## Key Takeaways

- The choice of MCMC sampler (Metropolis-Hastings vs. Gibbs) significantly impacts the efficiency of exploring the posterior distribution in this probit model.
- Gibbs sampling demonstrated superior performance, suggesting it may be the preferred choice in similar scenarios.
- The ability to visualize the posterior distribution and confirm that the true parameters lie within a high-density region validates the model's ability to fit the data.
