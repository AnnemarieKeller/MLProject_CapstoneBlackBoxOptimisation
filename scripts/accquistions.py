from scipy import stats
import numpy as np


def acquisition_ucb(mu, sigma, iteration, beta0=10):
    beta = beta0 * np.exp(-iteration / 13) + 1.0
    return mu + beta * sigma

def acquisition_pi(mu, sigma, y_max, eta=0.01):
    z = (mu - y_max - eta) / (sigma + 1e-12)
    return stats.norm.cdf(z)

def acquisition_ei(mu, sigma, y_max, xi=0.01):
    with np.errstate(divide='ignore'):
        z = (mu - y_max - xi) / (sigma + 1e-12)
        ei = (mu - y_max - xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    ei[sigma == 0.0] = 0.0
    return ei


def acquisition_thompson(mu, sigma):
    """
    Thompson Sampling: sample from the Gaussian posterior
    """
    return np.random.normal(mu, sigma)

def acquisition_knowledge_gradient(mu, sigma, y_max, xi=0.01):
    """
    Approximate Knowledge Gradient: expected improvement using one-step lookahead
    For simplicity, we can use EI as a proxy
    """
    return acquisition_ei(mu, sigma, y_max, xi)

def acquisition_entropy_search(mu, sigma, y_max, n_samples=1000):
    """
    Max-value Entropy Search (MES) approximation
    Reduces uncertainty about global maximum
    This is a simplified approximation using log probabilities
    """
    samples = np.random.normal(mu, sigma, size=n_samples)
    max_samples = np.max(samples, axis=0)
    entropy = -np.mean(np.log(stats.norm.pdf(max_samples)))
    return entropy * np.ones_like(mu)  # same shape as mu

def acquisition_portfolio(mu, sigma, y_max, iteration):
    """
    Simple portfolio: average of UCB, EI, and Thompson
    """
    ucb = acquisition_ucb(mu, sigma, iteration)
    ei = acquisition_ei(mu, sigma, y_max)
    ts = acquisition_thompson(mu, sigma)
    portfolio = (ucb + ei + ts) / 3
    return portfolio
