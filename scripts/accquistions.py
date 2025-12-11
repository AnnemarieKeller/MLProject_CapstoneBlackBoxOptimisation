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
def log_all_acquisitions(mu_all, sigma_all, y_max, gp_health, iteration):
    """
    Evaluate and log all acquisition function values for a batch of candidates.

    Parameters
    ----------
    mu_all : array-like
        Predicted means for each candidate (shape: n_candidates)
    sigma_all : array-like
        Predicted standard deviations for each candidate (shape: n_candidates)
    y_max : float
        Current best observed output
    gp_health : float
        Current GP health (used to scale acquisitions)
    iteration : int
        Current iteration number (used for time-dependent acquisitions like UCB)

    Returns
    -------
    acq_values : dict
        Dictionary mapping acquisition name -> np.ndarray of acquisition values
    """
    import numpy as np

    acq_values = {}
    for acq_name in ["UCB", "EI", "PI", "THOMPSON", "KG", "ENTROPY", "PORTFOLIO"]:
        vals = []
        gamma = 0.01  # EI and PI parameter
        eta = 0.01    # PI parameter
        kappa_ucb = 3.0 * np.exp(-iteration / max(1, iteration)) + 0.1  # avoid div by zero
        gamma_thompson = 1.0

        for mu, sigma in zip(mu_all, sigma_all):
            mu, sigma = mu.item(), sigma.item()
            if acq_name == "EI":
                xi = 0.01
                v = acquisition_ei(mu, sigma, y_max, xi=xi)
                params = {"xi": xi}
            elif acq_name == "PI":
                v = acquisition_pi(mu, sigma, y_max, eta)
            elif acq_name == "UCB":
                v = acquisition_ucb_Kappa(mu, sigma, iteration, kappa=kappa_ucb)
            elif acq_name == "THOMPSON":
                v = acquisition_thompson(mu, sigma)
            elif acq_name == "KG":
                v = acquisition_knowledge_gradient(mu, sigma, y_max)
            elif acq_name == "ENTROPY":
                v = acquisition_entropy_search(mu, sigma, y_max)
            elif acq_name == "PORTFOLIO":
                v = acquisition_portfolio(mu, sigma, y_max, iteration=iteration)
            else:
                v = 0.0

            vals.append(v * gp_health)

        acq_values[acq_name] = np.array(vals)

    return acq_values
def evaluate_acquisition(acq_name, mu, sigma, y_max, iteration, total_iterations):
    """
    Evaluate a single acquisition function for one candidate.

    Parameters
    ----------
    acq_name : str
        Acquisition function name ("EI", "PI", "UCB", "THOMPSON", etc.)
    mu : float
        GP predicted mean for the candidate
    sigma : float
        GP predicted standard deviation for the candidate
    y_max : float
        Current best observed output
    iteration : int
        Current iteration number
    total_iterations : int
        Total number of iterations

    Returns
    -------
    val : float
        Acquisition function value
    params : dict
        Dictionary of parameters used in the acquisition
    """
    params = {}

    if acq_name.upper() == "EI":
        xi = 0.01
        val = acquisition_ei(mu, sigma, y_max, xi=xi)
        params = {"xi": xi}

    elif acq_name.upper() == "PI":
        eta = 0.01
        val = acquisition_pi(mu, sigma, y_max, eta)
        params = {"eta": eta}

    elif acq_name.upper() == "UCB":
        kappa = 3.0 * np.exp(-iteration / max(1, total_iterations)) + 0.1
        val = acquisition_ucb_Kappa(mu, sigma, iteration=iteration, kappa=kappa)
        params = {"kappa": kappa}

    elif acq_name.upper() == "THOMPSON":
        gamma = 1.0
        val = acquisition_thompson(mu, sigma)
        params = {"mu": mu}

    elif acq_name.upper() == "KG":
        val = acquisition_knowledge_gradient(mu, sigma, y_max)

    elif acq_name.upper() == "ENTROPY":
        val = acquisition_entropy_search(mu, sigma, y_max)

    elif acq_name.upper() == "PORTFOLIO":
        val = acquisition_portfolio(mu, sigma, y_max, iteration=iteration)

    else:
        val = 0.0

    return val, params
# ------------------------------
# Dynamic acquisition parameter generator
def get_dynamic_params(acq_name, iteration, total_iterations):
    t = iteration / total_iterations  # normalized iteration (0 -> 1)
    
    if acq_name == "UCB":
        kappa_max, kappa_min = 3.0, 1.0
        kappa = kappa_max - (kappa_max - kappa_min) * t
        return {"kappa": round(kappa, 2)}
    
    elif acq_name in ["EI", "PI"]:
        xi_max, xi_min = 0.1, 0.01
        xi = xi_max - (xi_max - xi_min) * t
        return {"xi": round(xi, 3)}
    
    elif acq_name == "THOMPSON":
        samples_min, samples_max = 3, 10
        samples = int(samples_min + (samples_max - samples_min) * t)
        return {"samples": samples}
    
    elif acq_name == "PORTFOLIO":
        return {"weights": [0.5, 0.5]}
    
    else:
        return {}

# ------------------------------
# Strategy map including dynamic acquisition
function_strategy_map = {
    1: {"type": "dense_exploit", "acq": [{"name": "UCB"}]},
    2: {"type": "global_explore", "acq": [{"name": "UCB"}, {"name": "PORTFOLIO"}]},
    3: {"type": "refinement", "acq": [{"name": "UCB"}]},
    4: {"type": "dense_exploit", "acq": [{"name": "EI"}, {"name": "UCB"}]},
    5: {"type": "mixed", "acq": [{"name": "EI"}, {"name": "UCB"}]},
    6: {"type": "local_exploit", "acq": [{"name": "PI"}, {"name": "EI"}]},
    7: {"type": "explore_then_exploit", "acq": [{"name": "EI"}, {"name": "THOMPSON"}]},
    8: {"type": "refinement", "acq": [{"name": "UCB"}, {"name": "EI"}]}
}

# ------------------------------
# Candidate generation using strategy map
# def generate_candidates_by_strategy(strategy_type, gp, X_train_scaled, iteration, total_iterations, n_candidates):
#     """
#     Generate candidate points based on the given strategy type.
#     """
#     if strategy_type == "dense_exploit":
#         return generate_local_exploit_candidates(gp, X_train_scaled, n=n_candidates)
#     elif strategy_type == "global_explore":
#         return generate_uncertainty_candidates(gp, X_train_scaled, n=n_candidates)
#     elif strategy_type == "refinement":
#         return generate_refinement_candidates(gp, X_train_scaled, n=n_candidates)
#     elif strategy_type == "mixed":
#         return generate_dual_peak_candidates(gp, X_train_scaled, n=n_candidates)
#     elif strategy_type == "local_exploit":
#         return generate_local_exploit_candidates(gp, X_train_scaled, radius_scale=0.10, n=n_candidates)
#     elif strategy_type == "explore_then_exploit":
#         return generate_explore_then_exploit(gp, X_train_scaled, iteration, total_iterations, n=n_candidates)
#     else:
#         # fallback: random candidates
#         input_dim = X_train_scaled.shape[1]
#         return np.random.rand(n_candidates, input_dim)

# ------------------------------
# Wrapper to get acquisition parameters dynamically for the current function & iteration
def get_function_acq_params(func_id, iteration, total_iterations):
    acq_list = function_strategy_map[func_id]["acq"]
    acq_params = {}
    for acq in acq_list:
        name = acq["name"]
        acq_params[name] = get_dynamic_params(name, iteration, total_iterations)
    return acq_params

# ------------------------------
# Example usage in your BBO loop
# func_id: current function
# iteration: current iteration
# total_iterations: total planned iterations
# X_train_scaled, gp: current training data and GP model
# n_candidates = 500

# # Generate candidates
# strategy_type = function_strategy_map[func_id]["type"]
# X_candidates = generate_candidates_by_strategy(strategy_type, gp, X_train_scaled, iteration, total_iterations, n_candidates)

# # Get acquisition parameters for this iteration
# acquisition_params = get_function_acq_params(func_id, iteration, total_iterations)

# print("Candidates shape:", X_candidates.shape)
# print("Acquisition params:", acquisition_params)
def generate_local_exploit_candidates(gp, X_train, radius_scale=0.15, n=400):
    """
    Generate candidates around GP-predicted local maxima.
    Used for functions: 1, 4, 6 (peaked, exploit-local maxima).
    """
    mu, sigma = gp.predict(X_train, return_std=True)
    best_idx = np.argmax(mu)
    center = X_train[best_idx]

    dim = X_train.shape[1]
    noise = np.random.normal(0, radius_scale, size=(n, dim))
    cand = center + noise

    return np.clip(cand, 0.0, 1.0)


def generate_refinement_candidates(gp, X_train, n=350, noise=0.05):
    """
    Refine in narrow band around current best. Used for Functions 3, 8.
    """
    mu, sigma = gp.predict(X_train, return_std=True)
    best_idx = np.argmax(mu)
    center = X_train[best_idx]

    dim = X_train.shape[1]
    noise_arr = np.random.normal(0, noise, size=(n, dim))
    cand = center + noise_arr

    return np.clip(cand, 0, 1)


def generate_dual_peak_candidates(gp, X_train, n=500, noise=0.12):
    """
    Cluster GP predictions into two promising regions and sample both.
    Dual-Peak Candidate Generator (Function 5)
    """
    mu, sigma = gp.predict(X_train, return_std=True)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=5).fit(X_train, sample_weight=mu - mu.min() + 1e-6)

    centroids = kmeans.cluster_centers_
    dim = X_train.shape[1]

    cand = []
    for c in centroids:
        block = c + np.random.normal(0, noise, size=(n // 2, dim))
        cand.append(block)

    cand = np.vstack(cand)
    return np.clip(cand, 0, 1)


def generate_explore_then_exploit(gp, X_train, i, total_iters, n=500, explore_ratio=0.3, radius_scale=0.20):
    """
    Stage 1 → global explore, Stage 2 → exploit.
    Explore-Then-Exploit (Function 7)
    """
    if i < explore_ratio * total_iters:
        return generate_uncertainty_candidates(gp, X_train, n=n)
    else:
        return generate_local_exploit_candidates(gp, X_train, radius_scale=radius_scale, n=n)


def generate_uncertainty_candidates(gp, X_train, n=600):
    """
    Sample where uncertainty σ is largest. Function 2 and early 7
    """
    dim = X_train.shape[1]
    lhs = np.random.rand(n, dim)

    mu, sigma = gp.predict(lhs, return_std=True)
    idx = np.argsort(-sigma)[:min(300, n)]

    return lhs[idx]
# ------------------------------
# Optimized candidate generation parameters per function
function_candidate_params = {
    1: {"n_candidates": 400, "radius_scale": 0.15},  # dense_exploit, peaked local
    2: {"n_candidates": 600},                       # global_explore, higher candidates to cover space
    3: {"n_candidates": 350, "noise": 0.05},       # refinement, narrow around best
    4: {"n_candidates": 400, "radius_scale": 0.15},# dense_exploit, local maxima
    5: {"n_candidates": 500, "noise": 0.12},       # dual-peak, cover two promising regions
    6: {"n_candidates": 400, "radius_scale": 0.10},# local_exploit, small radius
    7: {"n_candidates": 500, "explore_ratio": 0.3, "radius_scale": 0.20}, # explore_then_exploit
    8: {"n_candidates": 350, "noise": 0.05}        # refinement, narrow around best
}

# ------------------------------
# Updated candidate generation using function-specific parameters
def generate_candidates_by_strategy(func_id,strat_type, gp, X_train_scaled, iteration, total_iterations):
    params = function_candidate_params.get(func_id, {})
    n_candidates = params.get("n_candidates", 500)
    
    strategy_type = strat_type
    
    if strategy_type == "dense_exploit":
        radius_scale = params.get("radius_scale", 0.15)
        return generate_local_exploit_candidates(gp, X_train_scaled, radius_scale=radius_scale, n=n_candidates)
    
    elif strategy_type == "global_explore":
        return generate_uncertainty_candidates(gp, X_train_scaled, n=n_candidates)
    
    elif strategy_type == "refinement":
        noise = params.get("noise", 0.05)
        return generate_refinement_candidates(gp, X_train_scaled, n=n_candidates, noise=noise)
    
    elif strategy_type == "mixed":
        noise = params.get("noise", 0.12)
        return generate_dual_peak_candidates(gp, X_train_scaled, n=n_candidates, noise=noise)
    
    elif strategy_type == "local_exploit":
        radius_scale = params.get("radius_scale", 0.10)
        return generate_local_exploit_candidates(gp, X_train_scaled, radius_scale=radius_scale, n=n_candidates)
    
    elif strategy_type == "explore_then_exploit":
        explore_ratio = params.get("explore_ratio", 0.3)
        radius_scale = params.get("radius_scale", 0.20)
        return generate_explore_then_exploit(gp, X_train_scaled, iteration, total_iterations,
                                             n=n_candidates, explore_ratio=explore_ratio, radius_scale=radius_scale)
    
    else:
        # fallback: random candidates
        input_dim = X_train_scaled.shape[1]
        return np.random.rand(n_candidates, input_dim)
