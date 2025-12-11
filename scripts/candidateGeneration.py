import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import qmc
from scripts.accquistions import *
from scripts.analysis.noise import *


def local_candidates(center, radius, n_points, bounds=None):
    center = np.atleast_1d(center)  # works for scalar or array
    dim = len(center)

    # radius can be scalar to broadcast to all dims
    radius = np.atleast_1d(radius)
    if radius.size == 1:
        radius = np.ones(dim) * radius

    noise = (np.random.rand(n_points, dim) * 2 - 1) * radius
    X = center + noise

    if bounds is not None:
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        X = np.clip(X, lb, ub)

    return X



def random_candidates(n_points, dim):
    return np.random.rand(n_points, dim)

def grid_candidates(resolution, dim):
    axes = [np.linspace(0, 1, resolution)] * dim
    mesh = np.meshgrid(*axes)
    return np.column_stack([m.ravel() for m in mesh])

def generate_candidates(input_dim, n_candidates=500, method="random",
                        local_center=None, local_radius=0.05, bounds=None):

    # ---- Local (dimension-agnostic) ----
    if method == "local":
        if local_center is None:
            raise ValueError("local_center required for local sampling")
        return local_candidates(local_center, local_radius,
                                n_candidates, bounds=bounds)
 

    # ---- Global (dimension-aware via method) ----
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
        raise ValueError(f"Unhandled input_dim - Not Mapped to Candidate Method Generation: {input_dim}")


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


def analyze_for_candidate_generation(X_train, y_train, gp=None, local_radius_base=0.05):
    """
    Analyze the landscape and return parameters for candidate generation.
    
    Args:
        X_train: np.array of shape (n_points, dim) - evaluated inputs
        y_train: np.array of shape (n_points,) - corresponding outputs
        gp: optional trained GaussianProcessRegressor
        local_radius_base: base radius for local candidate generation (scaled by smoothness)
    
    Returns:
        dict containing:
            - current_best_x
            - current_best_y
            - min_prob (for global candidate filtering)
            - local_radius (for local candidate generation)
            - favor_global (True/False)
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train).flatten()

    
    # 1. Current best
    idx_best = np.argmax(y_train)
    current_best_x = X_train[idx_best]
    current_best_y = y_train[idx_best]
    
    # 2. Estimate noise / smoothness
    if gp is not None:
        # Use GP cross-validation residuals to estimate noise
        mu, sigma = gp.predict(X_train, return_std=True)
        residuals = np.abs(mu - y_train)
    else:
        # Simple estimate using differences between neighbors
        residuals = np.abs(np.diff(y_train))
        if residuals.size == 0:
            residuals = np.array([0.0])
    
    y_range = max(y_train.max() - y_train.min(), 1e-8)
    noise_level = np.mean(residuals) / y_range
    noise_level = np.clip(noise_level, 0.0, 1.0)
    
    # 3. Map noise_level to min_prob
    # More noise → lower min_prob to allow exploration
    min_prob = 0.7 - 0.4 * noise_level
    min_prob = np.clip(min_prob, 0.2, 0.7)
    
    # 4. Suggest local radius (smaller for noisy functions, larger for smooth)
    local_radius = local_radius_base * (1.0 - noise_level)  # scales down if noisy
    
    # 5. Decide if global search is likely useful
    # Heuristic: if many points are close to best and smooth → exploit locally
    # Otherwise, favor global exploration
    favor_global = True
    fdc = np.corrcoef(np.linalg.norm(X_train - current_best_x, axis=1), y_train)[0,1]
    if fdc is not None and fdc < -0.5:
        favor_global = False
    
    return {
        "current_best_x": current_best_x,
        "current_best_y": current_best_y,
        "min_prob": min_prob,
        "local_radius": local_radius,
        "favor_global": favor_global,
        "noise_level": noise_level
    }

   
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
def global_is_good(X_pass, mu, sigma, current_best_y,
                   min_global_pass=5,
                   EI_threshold=0.0):
    
    # A: enough candidates have P(improve) >= threshold
    cA = len(X_pass) >= min_global_pass

    # B: expected improvement exists somewhere globally
    ei = acquisition_ei(mu, sigma, current_best_y)
    cB = ei.max() > EI_threshold

    # global is good if either signal says “yes”
    return cA or cB
def propose_candiates(gp, n_candidates, X_train, y_train,method  ):
    idx_best = np.argmax(y_train)
    input_dim = X_train.shape[1]

# Current best output
    current_best_y = y_train[idx_best]

# Current best input
    current_best_x = X_train[idx_best]
# 1. Generate global candidates
    X_global = generate_candidates(input_dim, n_candidates, method)

# 2. Evaluate them
    prob, mu, sigma = estimate_success_prob(
        gp, X_global, current_best_y, n_samples=3000
    )
    ei = acquisition_ei(mu, sigma, current_best_y)
    noise_level, min_prob = estimate_noise_and_min_prob(X_train, y_train, gp=gp)


# 3. Decide
    if global_is_good(prob, mu, sigma, ei):
    # keep global exploration
        idx_pass = prob >= min_prob
        return X_global[idx_pass]
    else:
    # exploit locally
        X_local = generate_candidates(
        input_dim, n_candidates, method='local',
        local_center=current_best_x,
        local_radius=0.05
        )   
        return X_local
   
   
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

def propose_next_point_local(X_train, y_train, gp, acq_name="UCB",
                             local_center=None, local_radius=0.1,
                             n_candidates=50, iteration=0, model_type="GP",
                             y_scaler=None):
    """
    Propose the next point in a local region around local_center.

    Parameters:
        X_train, y_train: current training data
        gp: trained GP model
        acq_name: acquisition function to use ("UCB", "EI", "PI", etc.)
        local_center: center of the local search region
        local_radius: scalar radius around the center
        n_candidates: number of local candidates to generate
        iteration: current BO iteration (for UCB)
        model_type: "GP" or "SVR"
        y_scaler: optional scaler for y if data is scaled

    Returns:
        next_point_scaled: candidate (scaled) chosen to evaluate next
    """
    if local_center is None:
        raise ValueError("local_center must be provided for local candidate generation")
    
    input_dim = X_train.shape[1]

    # Generate local candidates
    X_candidates = generate_candidates(
        input_dim=input_dim,
        n_candidates=n_candidates,
        method="local",
        local_center=local_center,
        local_radius=local_radius
    )

    # Evaluate acquisition function for all candidates
    acq_values = []
    y_max = np.max(y_train)
    for x in X_candidates:
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        mu, sigma = mu.item(), sigma.item()
        acq_val = select_acquisition(acq_name, mu, sigma, iteration=iteration, y_max=y_max, model_type=model_type)
        acq_values.append(acq_val)

    acq_values = np.array(acq_values)

    # Pick the candidate with the highest acquisition value
    best_idx = np.argmax(acq_values)
    next_point_scaled = X_candidates[best_idx]

    return next_point_scaled

def propose_next_point_local_mult(
    X_train,
    y_train,
    gp,
    acq_name="UCB",
    local_centers=None,
    local_radius=0.1,
    n_candidates=50,
    y_max=None,
    threshold=None,
    scale_candidates=False,
    random_state=42,
    iter=0
):
    np.random.seed(random_state)
    n_dims = X_train.shape[1]

    # Default single center
    if local_centers is None:
        local_centers = np.array([X_train[-1]])

    # ---------------------------------------
    # 1. Handle scaling consistently
    # ---------------------------------------
    if scale_candidates:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        local_centers_scaled = scaler.transform(local_centers)
        center_data = local_centers_scaled
    else:
        scaler = None
        center_data = local_centers  # unscaled

    # ---------------------------------------
    # 2. Generate local candidates
    # ---------------------------------------
    candidates = []
    for center in center_data:
        perturb = (np.random.rand(n_candidates, n_dims) - 0.5) * 2 * local_radius
        candidates.append(center + perturb)
    candidates = np.vstack(candidates)

    # Clip only if scaled
    if scale_candidates:
        candidates = np.clip(candidates, 0.0, 1.0)

    # ---------------------------------------
    # 3. Predict μ, σ
    # ---------------------------------------
    mu, sigma = gp.predict(candidates, return_std=True)

    # Threshold filter
    if threshold is not None:
        mask = mu >= threshold
        if not mask.any():
            mask = np.ones_like(mu, bool)
        candidates = candidates[mask]
        mu = mu[mask]
        sigma = sigma[mask]

    # ---------------------------------------
    # 4. Acquisition
    # ---------------------------------------
    acq_values = select_acquisition(acq_name, mu, sigma, iter, y_max=y_max)

    best_idx = np.argmax(acq_values)
    next_point_scaled = candidates[best_idx]

    # ---------------------------------------
    # 5. Inverse scale only if scaled
    # ---------------------------------------
    if scale_candidates:
        next_point = scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten()
    else:
        next_point = next_point_scaled

    y_next = gp.predict(next_point.reshape(1, -1)).item()

    # IMPORTANT: return next_point, not candidates
    best_acq_name = acq_name  # or whichever was selected
#     # Compute the acquisition value
#     if acq_name.upper() == "UCB":
#         kappa = 3.0 * np.exp(-iteration / total_iterations) + 0.1
#         acq = acquisition_ucb_Kappa(mu, sigma, iteration=iteration, kappa=kappa)
#         acq_params = get_acq_params(acq_name, iteration, kappa=kappa)
#     elif acq_name.upper() == "EI":
#         acq = acquisition_ei(mu, sigma, y_max)
#         acq_params = get_acq_params(acq_name, eta=0.01)
# # etc...

   

    return next_point, y_next


import numpy as np
from sklearn.preprocessing import MinMaxScaler
def get_acq_params(acq_name, iteration=None, kappa=None, eta=None, gamma=None):
    """
    Return a dictionary of acquisition function parameters for logging.
    """
    acq_name = acq_name.upper()
    params = {}
    
    if acq_name == "UCB":
        params["kappa"] = kappa
    elif acq_name == "EI":
        params["eta"] = eta if eta is not None else 0.0
    elif acq_name == "PI":
        params["eta"] = eta if eta is not None else 0.0
    elif acq_name == "THOMPSON":
        params["gamma"] = gamma if gamma is not None else 0.0
    # Add any other acquisition types you use
    return params


def propose_next_point_local_boost(
    X_train,
    y_train,
    gp,
    acq_name="UCB",
    local_centers=None,      # array of shape (n_centers, n_dims)
    local_radius=0.1,
    n_candidates=50,
    y_max=None,
    threshold=None,          # only consider μ >= threshold
    scale_candidates=False,
    boost_top=True,          # boost acquisition near top peaks
    boost_radius=0.05,       # fraction of domain for boosting
    boost_factor=2.0,        # how much to multiply acq value
    random_state=42
):
    """
    Generate candidate points locally around one or multiple centers and select the best via acquisition.
    Supports optional boosting near the current top predictions.
    """
    np.random.seed(random_state)
    n_dims = X_train.shape[1]

    # Default: single center = last training point
    if local_centers is None:
        local_centers = np.array([X_train[-1]])

    # Scale candidates to [0,1] for uniform sampling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    local_centers_scaled = scaler.transform(local_centers) if scale_candidates else local_centers

    # Generate candidates around each center
    candidates = []
    for center in local_centers_scaled:
        perturb = (np.random.rand(n_candidates, n_dims) - 0.5) * 2 * local_radius
        candidates.append(center + perturb)
    candidates = np.vstack(candidates)

    # Clip candidates to [0,1]
    candidates = np.clip(candidates, 0.0, 1.0)

    # Predict μ and σ
    mu, sigma = gp.predict(candidates, return_std=True)

    # Apply threshold filter if needed
    if threshold is not None:
        mask = mu >= threshold
        if not np.any(mask):
            mask = np.ones_like(mu, dtype=bool)  # fallback if all below threshold
        candidates = candidates[mask]
        mu = mu[mask]
        sigma = sigma[mask]

    # Compute acquisition
    acq_values = select_acquisition(acq_name, mu, sigma, y_max=y_max, iteration=None, model_type="GP")

    # Boost candidates near current top peak
    if boost_top:
        # Find top point in training data
        top_idx = np.argmax(y_train)
        top_point = X_train[top_idx]
        top_point_scaled = scaler.transform(top_point.reshape(1, -1)).flatten() if scale_candidates else top_point

        # Create mask for candidates within boost_radius of top point
        distance = np.linalg.norm(candidates - top_point_scaled, axis=1)
        boost_mask = distance <= boost_radius
        acq_values[boost_mask] *= boost_factor

    # Pick best candidate
    best_idx = np.argmax(acq_values)
    next_point_scaled = candidates[best_idx]

    # Inverse scale if needed
    next_point = scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten() if scale_candidates else next_point_scaled
    y_next = gp.predict(next_point.reshape(1, -1)).item()

    return next_point, y_next
def score_candidates_with_health(gp, X_candidates, acq_name, y_train, iteration, gp_health_score):
    """
    Scores candidates based on acquisition function and GP health.
    Returns best candidate index.
    """
    mu, sigma = gp.predict(X_candidates, return_std=True)
    mu, sigma = mu.flatten(), sigma.flatten()
    y_max = np.max(y_train)

    acq_values = []
    for m, s in zip(mu, sigma):
        if acq_name.upper() == "UCB":
            kappa = 3.0
            acq = acquisition_ucb_Kappa(m, s, iteration, kappa=kappa)
        elif acq_name.upper() == "EI":
            acq = acquisition_ei(m, s, y_max)
        elif acq_name.upper() == "PI":
            acq = acquisition_pi(m, s, y_max)
        elif acq_name.upper() == "THOMPSON":
            acq = acquisition_thompson(m, s)
        else:
            raise ValueError(f"Unknown acquisition {acq_name}")
        acq_values.append(acq)

    acq_values = np.array(acq_values)
    # Multiply acquisition by GP health score
    scores = acq_values * gp_health_score
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]

def ensure_2d_candidate(candidate, input_dim):
    candidate_arr = np.atleast_1d(candidate).reshape(1, -1)
    if candidate_arr.shape[1] != input_dim:
        raise ValueError(f"Candidate shape {candidate_arr.shape} does not match GP input dimension {input_dim}")
    return candidate_arr

#def generate_explore_then_exploit(gp, X_train, i, total_iters, n=500):

    """Stage 1 → global explore, Stage 2 → exploit.Explore-Then-Exploit (Function 7)"""
    if i < 0.3 * total_iters:
        return generate_uncertainty_candidates(gp, X_train, n=n)
    else:
        return generate_local_exploit_candidates(gp, X_train, radius_scale=0.20, n=n)
def generate_dual_peak_candidates(gp, X_train, n=500):
    """Cluster GP predictions into two promising regions and sample both.Dual-Peak Candidate Generator (Function 5)"""
    mu, sigma = gp.predict(X_train, return_std=True)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=5).fit(X_train, sample_weight=mu - mu.min() + 1e-6)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    dim = X_train.shape[1]

    cand = []
    for c in centroids:
        noise = np.random.normal(0, 0.12, size=(n // 2, dim))
        block = c + noise
        cand.append(block)

    cand = np.vstack(cand)
    return np.clip(cand, 0, 1)

def generate_refinement_candidates(gp, X_train, n=350):
    """Refine in narrow band around current best. Used for Functions 3, 8."""
    mu, sigma = gp.predict(X_train, return_std=True)
    best_idx = np.argmax(mu)
    center = X_train[best_idx]

    dim = X_train.shape[1]
    noise = np.random.normal(0, 0.05, size=(n, dim))

    cand = center + noise
    return np.clip(cand, 0, 1)
def generate_uncertainty_candidates(gp, X_train, n=600):
    """Sample where uncertainty σ is largest. Function 2 and early 7 """
    dim = X_train.shape[1]
    lhs = np.random.rand(n, dim)

    mu, sigma = gp.predict(lhs, return_std=True)
    idx = np.argsort(-sigma)[:300]

    return lhs[idx]
def generate_local_exploit_candidates(gp, X_train, radius_scale=0.15, n=400):
    """Generate candidates around GP-predicted local maxima.Used for functions: 1, 4, 6 (peaked, exploit-local maxima)."""
    mu, sigma = gp.predict(X_train, return_std=True)
    best_idx = np.argmax(mu)
    center = X_train[best_idx]

    dim = X_train.shape[1]
    noise = np.random.normal(0, radius_scale, size=(n, dim))
    cand = center + noise

    cand = np.clip(cand, 0.0, 1.0)
    return cand

def generate_candidates_by_strategy1(strategy_type, gp, X_train_scaled, iteration, total_iterations, n_candidates):
    """
    Generate candidate points based on the given strategy type.
    
    Parameters
    ----------
    strategy_type : str
        Strategy name: dense_exploit, global_explore, refinement, mixed, local_exploit, explore_then_exploit
    gp : GaussianProcessRegressor
        Current GP model
    X_train_scaled : np.ndarray
        Current training data
    iteration : int
        Current iteration index
    total_iterations : int
        Total number of iterations
    n_candidates : int
        Number of candidates to generate
    
    Returns
    -------
    X_candidates_scaled : np.ndarray
        Array of candidate points
    """
    if strategy_type == "dense_exploit":
        return generate_local_exploit_candidates(gp, X_train_scaled, n=n_candidates)
    elif strategy_type == "global_explore":
        return generate_uncertainty_candidates(gp, X_train_scaled, n=n_candidates)
    elif strategy_type == "refinement":
        return generate_refinement_candidates(gp, X_train_scaled, n=n_candidates)
    elif strategy_type == "mixed":
        return generate_dual_peak_candidates(gp, X_train_scaled, n=n_candidates)
    elif strategy_type == "local_exploit":
        return generate_local_exploit_candidates(gp, X_train_scaled, radius_scale=0.10, n=n_candidates)
    elif strategy_type == "explore_then_exploit":
        return generate_explore_then_exploit(gp, X_train_scaled, iteration, total_iterations, n=n_candidates)
    else:
        # fallback: random candidates
        input_dim = X_train_scaled.shape[1]
        return np.random.rand(n_candidates, input_dim)
    
def generate_multi_peak_candidates(gp, X_train_scaled, n_candidates=500, top_k_peaks=5, local_scale=0.05, input_dim=None, random_state=42):
    """
    for function 5 , as suspected multi peaks .
    ----
    """
    np.random.seed(random_state)
    # global uniform candidates
    X_global = np.random.uniform(0,1,size=(n_candidates,input_dim))

    # local candidates around top predicted peaks
    mu_all, sigma_all = gp.predict(X_train_scaled, return_std=True)
    ucb = mu_all + 2.0*sigma_all
    top_idx = np.argsort(ucb.ravel())[-top_k_peaks:]
    X_local_list = []
    for idx in top_idx:
        peak = X_train_scaled[idx]
        X_local = peak + local_scale * np.random.randn(n_candidates//top_k_peaks, input_dim)
        X_local = np.clip(X_local, 0, 1)
        X_local_list.append(X_local)
    X_local = np.vstack(X_local_list)
    X_candidates = np.vstack([X_global, X_local])
    return X_candidates



