from scipy import stats
import numpy as np


def acquisition_ucb_Kappa(mu, sigma, iteration, kappa):
    return mu + kappa * sigma

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
    if np.isscalar(ei):
        if sigma == 0.0:
            ei = 0.0
    else:
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



def select_acquisition(acq_name, mu, sigma=None, iteration=None, y_max=None, model_type="GP", **kwargs):
    """
    Select acquisition function dynamically.
    
    For SVR or other deterministic models, sigma can be None, and acquisition defaults to exploitation.
    
    Parameters:
        acq_name: "UCB", "EI", "PI" (ignored for SVR)
        mu: predicted mean / prediction
        sigma: predicted std (optional for SVR)
        iteration: for UCB
        y_max: for EI/PI
        model_type: "GP" or "SVR"
    """
    if model_type.upper() == "SVR" or sigma is None:
        # Deterministic model: exploitation only
        return mu
    
    # GP case
    acq_name = acq_name.upper()
    if acq_name == "UCB":
        if iteration is None:
            raise ValueError("iteration must be provided for UCB")
        return acquisition_ucb(mu, sigma, iteration, beta0=kwargs.get("beta0", 10))
    elif acq_name == "EI":
        if y_max is None:
            raise ValueError("y_max must be provided for EI")
        return acquisition_ei(mu, sigma, y_max, xi=kwargs.get("xi", 0.01))
    elif acq_name == "PI":
        if y_max is None:
            raise ValueError("y_max must be provided for PI")
        return acquisition_pi(mu, sigma, y_max, eta=kwargs.get("eta", 0.01))
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")
def modified_acquisition(X, gp, acq_name="UCB", iteration=None, y_max=None, model_type="GP", boost_middle=True, middle_bounds=(0.3,0.7), boost_factor=2.0, **kwargs):
    """
    Compute acquisition values with optional middle-region boosting.

    Parameters:
        X: candidate points, shape (n_points, n_dims)
        gp: trained GP model
        acq_name: which acquisition function to use ("UCB", "EI", "PI", "THOMPSON", "KG", "ENTROPY", "PORTFOLIO")
        iteration: required for UCB
        y_max: required for EI/PI/KG
        model_type: "GP" or "SVR"
        boost_middle: whether to boost middle points
        middle_bounds: tuple (low, high) defining the middle region in all dimensions
        boost_factor: multiplier for acquisition values in the middle
        **kwargs: extra args for acquisition functions (xi, eta, beta0, n_samples)
    """
    # Compute the standard acquisition values using your existing wrapper
    acq = select_acquisition(acq_name, mu=gp.predict(X, return_std=False)[0],
                             sigma=gp.predict(X, return_std=True)[1] if model_type=="GP" else None,
                             iteration=iteration,
                             y_max=y_max,
                             model_type=model_type,
                             **kwargs)

    if boost_middle:
        low, high = middle_bounds
        # Create mask for points where all dimensions are inside the middle box
        middle_mask = np.all((X >= low) & (X <= high), axis=1)
        acq[middle_mask] *= boost_factor

    return acq
