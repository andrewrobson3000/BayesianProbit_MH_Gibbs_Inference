## Bayesian Probit Regression with Gibbs Sampling (and Metropolis-Hastings)

This repository implements Bayesian inference for a probit regression model. It employs both Gibbs sampling and Metropolis-Hastings to explore the posterior distribution. The code leverages latent variables to make Gibbs sampling efficient.

**Key Files**

* **Bayesian Probit Regression:** Contains the core implementation of:
    * Data generation functions
    * Prior specification
    * Posterior calculation
    * Metropolis-Hastings implementation
    * Gibbs sampler implementation

**Prerequisites**

* Python 3.x
* NumPy
* SciPy
* Matplotlib 
* Pandas
* Seaborn
* Statsmodels 

**Usage**

1. Clone the repository: `git clone https://github.com/andrewrobson3000/BayesianProbit_MH_Gibbs_Inference.git`
2. Install dependencies: `pip install -r requirements.txt`

**Probit Model and Bayesian Inference**

* **Probit Regression:** The probit regression model is a statistical method designed to analyze relationships between explanatory variables and a binary outcome (e.g., success/failure, yes/no). It assumes a latent (unobserved) continuous variable that determines the binary outcome.

* **Bayesian Approach:** Bayesian inference provides a framework for incorporating prior knowledge about the model's parameters and quantifying uncertainty in our estimates.  

* **Gibbs Sampling:** Direct sampling from the posterior distribution in a probit model is often difficult. By introducing latent variables, we can use Gibbs sampling, an MCMC technique, to efficiently sample from the posterior. Gibbs sampling iteratively draws samples from the conditional distributions of the parameters and the latent variables, simplifying the sampling process.

**Feedback and Contributions**

Feel free to submit issues for bug reports or feature requests. Pull requests for improvements and extensions are welcome!

