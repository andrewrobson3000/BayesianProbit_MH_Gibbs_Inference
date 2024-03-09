#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, multivariate_normal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


# In[18]:


# Generate synthetic dataset
n, p = 50, 2  # Sample size and number of covariates
beta_true = np.array([2, -1]).reshape(-1, 1)  # True beta values
X = np.hstack((np.random.normal(1, 1, (n, 1)), np.random.normal(1.5, 1, (n, 1))))  # Explanatory variables
Y = (np.random.normal(loc=X.dot(beta_true).flatten(), scale=1) >= 0).astype(int)  # Binary observations


# In[14]:


def compute_posterior_on_grid(grid_x, grid_y):
    grid_density = np.zeros(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            beta = np.array([[grid_x[i, j]], [grid_y[i, j]]])
            grid_density[i, j] = np.exp(full_log_posterior(beta))
    return grid_density

# Define grid
beta1_range = np.linspace(0.5, 3.5, 100)
beta2_range = np.linspace(-2, 0, 100)
grid_x, grid_y = np.meshgrid(beta1_range, beta2_range)

# Compute posterior on grid
grid_density = compute_posterior_on_grid(grid_x, grid_y)


# In[15]:


# Plot
plt.figure(figsize=(8, 6))
plt.contourf(grid_x, grid_y, grid_density, levels=50, cmap='viridis')
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title('Posterior Density')
plt.colorbar(label='Density')
plt.show()


# In[16]:


def log_prior(beta):
    B = np.array([[3, 0], [0, 3]])  # Prior variance
    mean = np.zeros(beta.shape[0])
    return multivariate_normal.logpdf(beta.flatten(), mean=mean, cov=B)

def full_log_posterior(beta):
    prob_1 = stats.norm.logcdf(X[Y == 1].dot(beta))
    prob_0 = stats.norm.logcdf(-X[Y == 0].dot(beta))
    return np.sum(prob_1) + np.sum(prob_0) + log_prior(beta)


# In[5]:


def metropolis_hastings(n_iterations=10000, init_beta=np.array([0, 0])):
    chain = np.zeros((n_iterations, 2))
    chain[0] = init_beta
    n_accepts = 0
    current_log_posterior = full_log_posterior(init_beta.reshape(-1, 1))
    
    for t in range(1, n_iterations):
        proposal = chain[t-1] + np.random.normal(0, 1, 2)
        proposal_log_posterior = full_log_posterior(proposal.reshape(-1, 1))
        accept_probability = min(1, np.exp(proposal_log_posterior - current_log_posterior))
        
        if np.random.rand() < accept_probability:
            chain[t] = proposal
            current_log_posterior = proposal_log_posterior
            n_accepts += 1
        else:
            chain[t] = chain[t-1]
    
    acceptance_rate = n_accepts / n_iterations
    print(f"Acceptance rate: {acceptance_rate * 100:.2f}%")
    return chain

mh_chain = metropolis_hastings()


# In[6]:


def sample_z_given_beta(beta):
    z = np.zeros(n)
    for i in range(n):
        mean = X[i, :].dot(beta)
        a, b = (0, np.inf) if Y[i] == 1 else (-np.inf, 0)
        z[i] = truncnorm.rvs((a - mean) / 1, (b - mean) / 1, loc=mean, scale=1)
    return z

def sample_beta_given_z(z):
    B_inv = np.linalg.inv(np.array([[3, 0], [0, 3]]))
    variance = np.linalg.inv(B_inv + X.T.dot(X))
    mean = variance.dot(X.T.dot(z))
    return np.random.multivariate_normal(mean, variance)

def gibbs_sampler(n_iterations=10000):
    beta_samples = np.zeros((n_iterations, 2))
    current_beta = np.array([0, 0])  # Initial beta values
    
    for t in range(n_iterations):
        z = sample_z_given_beta(current_beta)
        current_beta = sample_beta_given_z(z)
        beta_samples[t, :] = current_beta
    
    return beta_samples

gibbs_chain = gibbs_sampler()


# In[7]:


# MH Sampler Path
plt.figure(figsize=(10, 6))
plt.plot(mh_chain[:1000, 0], mh_chain[:1000, 1], marker='o', color='darkblue', markersize=3, linestyle='-', alpha=0.5)
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title('Metropolis-Hastings Sampler Chain (First 1000 Steps)')
plt.show()

# Gibbs Sampler Path
plt.figure(figsize=(10, 6))
plt.plot(gibbs_chain[:1000, 0], gibbs_chain[:1000, 1], 'o', color='darkred', markersize=3, linestyle='-', alpha=0.5)
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title('Gibbs Sampler Path (First 1000 Iterations)')
plt.show()


# In[8]:


# MH Sampler's KDE
MHchain_df = pd.DataFrame(mh_chain, columns=["X1", "X2"])
sns.kdeplot(data=MHchain_df, x="X1", y="X2", fill=True)
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title("Kernel Density Estimate of MH Sampler's Posterior Distribution")
plt.show()

# Gibbs Sampler's KDE
Gibbschain_df = pd.DataFrame(gibbs_chain, columns=["X1", "X2"])
sns.kdeplot(data=Gibbschain_df, x="X1", y="X2", fill=True)
plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title("Kernel Density Estimate of Gibbs Sampler's Posterior Distribution")
plt.show()


# In[9]:


acf_gibbs = acf(gibbs_chain[:, 0], nlags=50, fft=True)
acf_mh = acf(mh_chain[:, 0], nlags=50, fft=True)
acf_df = pd.DataFrame({'Lag': np.arange(len(acf_gibbs)), 'Gibbs': acf_gibbs, 'MH': acf_mh})
acf_df_melted = acf_df.melt(id_vars=['Lag'], var_name='Sampler', value_name='ACF')

plt.figure(figsize=(10, 6))
sns.lineplot(data=acf_df_melted, x='Lag', y='ACF', hue='Sampler', marker='o')
plt.axhline(0, color='grey', lw=1, linestyle='--')
plt.title('Autocorrelation Function (ACF) Comparison')
plt.ylabel('ACF')
plt.xlabel('Lag')
plt.show()


# In[11]:


burn_in = 1000
mh_samples_post_burnin = mh_chain[burn_in:, :]
gibbs_samples_post_burnin = gibbs_chain[burn_in:, :]

# MH Sampler Beta1 Posterior
sns.histplot(mh_samples_post_burnin[:, 0], kde=True)
plt.xlabel(r'$\beta_1$')
plt.title('Posterior Distribution of Beta1 (MH Sampler)')
plt.show()

# Gibbs Sampler Beta1 Posterior
sns.histplot(gibbs_samples_post_burnin[:, 0], kde=True)
plt.xlabel(r'$\beta_1$')
plt.title('Posterior Distribution of Beta1 (Gibbs Sampler)')
plt.show()

