import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import qmc


def random_candidates(n_points, dim):
    return np.random.rand(n_points, dim)

def grid_candidates(resolution, dim):
    axes = [np.linspace(0, 1, resolution)] * dim
    mesh = np.meshgrid(*axes)
    return np.column_stack([m.ravel() for m in mesh])

def generate_candidates(input_dim, n_candidates=500, method="random"):

    if method =="lhs":
        sampler = qmc.LatinHypercube(d=input_dim)
        return  sampler.random(n_candidates)
    if method == "random":
        return np.random.rand(n_candidates, input_dim)
    elif method == "grid":
        # only works for small dim <= 3
        lin = [np.linspace(0,1,int(np.ceil(n_candidates**(1/input_dim)))) for _ in range(input_dim)]
        mesh = np.meshgrid(*lin)
        X = np.column_stack([m.ravel() for m in mesh])
        if X.shape[0] > n_candidates:
            X = X[:n_candidates] 
        return X
    elif method == "sobol":
        try:
            sampler = qmc.Sobol(d=input_dim, scramble=True)
            m = int(np.ceil(np.log2(n_candidates)))
            X= sampler.random_base2(m)[:n_candidates]
            return X
        except ImportError:
            print("Sobol requires scipy >= 1.7. Falling back to random.")
            return np.random.rand(n_candidates, input_dim)
    else:
        raise ValueError(f"Unknown candidate generation method: {method}")
    
def determine_candidate_generation_method(input_dim):

    # ---- Convert input_dim to integer safely ----
    if isinstance(input_dim, tuple):
        # e.g. (20, 8) -> 8
        if len(input_dim) == 1:
            input_dim = input_dim[0]
        else:
            input_dim = input_dim[-1]

    elif isinstance(input_dim, (list, np.ndarray)):
        if len(input_dim) == 1:
            input_dim = int(input_dim[0])
        else:
            raise ValueError(f"Ambiguous input_dim: {input_dim}")

    elif isinstance(input_dim, int):
        pass  # already good

    else:
        raise TypeError(f"input_dim must be int/tuple/list/array, got {type(input_dim)}")


    # ---- Validate ----
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive integer, got {input_dim}")

    # ---- Select method ----
    if input_dim <= 3:
       return "random"
    elif 4 <= input_dim <= 6:
        return "lhs"
    elif input_dim >= 7:
        return "sobol"
    else:
        raise ValueError(f"Unhandled input_dim: {input_dim}")

    method = ''
    if input_dim <= 3:
       return "random"
    elif 4 <= input_dim <= 6:
        return "lhs"      # Latin Hypercube Sampling
    elif input_dim >= 7:
        return "sobol" 
    else:
        raise ValueError(f"input_dim not mapped to candiate method generation: {input_dim}")
   


def choose_candidates_with_risk_filter(
    gp,               # trained GaussianProcessRegressor
    X_cand,           # array of candidate inputs, shape (n_cand, d)
    current_best,     # scalar: current best observed output (y_best)
    margin=0.0,       # float: extra margin over current_best (optional)
    min_prob=0.6,     # float in [0,1]: minimum required probability to accept candidate
    n_draws=5000      # int: number of Monte‑Carlo draws to estimate probability
):
    """
    For each candidate in X_cand, estimate the probability that the *true* (noisy)
    output at that input exceeds (current_best + margin), by sampling from the GP predictive distribution.
    Returns the subset of candidates that pass the min_prob threshold,
    sorted by decreasing probability.
    """

    mu, sigma = gp.predict(X_cand, return_std=True)
    # Draw many samples per candidate from Normal(mu, sigma^2)
    # shape: (n_cand, n_draws)
    draws = np.random.normal(loc=mu[:, None],
                             scale=sigma[:, None],
                             size=(X_cand.shape[0], n_draws))

    threshold = current_best + margin
    # For each candidate, compute fraction of draws > threshold
    probs = (draws > threshold).mean(axis=1)

    # Filter candidates
    idx_pass = np.where(probs >= min_prob)[0]
    # Sort those by descending prob (so highest-chance candidates first)
    idx_sorted = idx_pass[np.argsort(-probs[idx_pass])]

    return X_cand[idx_sorted], probs[idx_sorted], mu[idx_sorted], sigma[idx_sorted]

   
def estimate_success_prob(gp, X_cand, current_best,
                          n_samples=5000, margin=0.0):
    """
    For each candidate in X_cand, draw many samples from the GP predictive
    distribution N(mu, sigma^2), and estimate the probability that the
    actual (noisy) output exceeds current_best + margin.
    Returns array of success probabilities.
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    # Draw many samples: shape (n_cand, n_draws)
    draws = np.random.normal(loc=mu[:, None],
                             scale=sigma[:, None],
                             size=(len(X_cand), n_samples))
    # Compare to threshold
    threshold = current_best + margin
    prob = (draws > threshold).mean(axis=1)
    return prob, mu, sigma
   
   
    # alternative sobol generation method
    # elif method == "sobol":
    #     # Hybrid approach for high dimension
    #     n_sobol = int(n_candidates * 0.6)
    #     n_local = n_candidates - n_sobol

    #     # Sobol points (must be base2)
    #     m = int(np.ceil(np.log2(n_sobol)))
    #     sampler = qmc.Sobol(d=input_dim, scramble=True)
    #     X_sobol = sampler.random_base2(m=m)
    #     if X_sobol.shape[0] > n_sobol:
    #         X_sobol = X_sobol[:n_sobol]

    #     # Local perturbations
    #     if initial_points is not None and len(initial_points) > 0:
    #         best_idx = np.random.choice(len(initial_points), n_local)
    #         best_points = initial_points[best_idx]
    #         sigma = 0.05
    #         X_local = best_points + np.random.normal(0, sigma, size=(n_local, input_dim))
    #         X_local = np.clip(X_local, 0, 1)
    #     else:
    #         X_local = np.random.rand(n_local, input_dim)

    #     return np.vstack([X_sobol, X_local])