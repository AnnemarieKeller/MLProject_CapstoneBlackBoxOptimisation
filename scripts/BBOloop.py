import pandas as pd
import numpy as np
import csv
import logging
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from .setup.gpBuilder import *
from .setBoundary import apply_boundary_penalty
from scripts.candidateGeneration import *
from scripts.utils.utils import *
import scripts.configs.functionConfig as funcConfig
from scripts.accquistions import *
from scripts.scaler import *



logger = logging.getLogger("BBO")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)





def bbo_loop(X_train, y_train, function_config, acquisition=None, num_iterations=30,
             model_type="GP", config_override=None, use_boundary=True,
             n_candidates=500, candidate_method="random", verbose=True):
    """
    Modular black-box optimization loop using GP or SVR surrogate.

    Returns:
        best_input, best_output, history (dict with all sampled points)
    """
    X_train = X_train.copy()
    y_train = y_train.copy()
    dim = function_config.get("dim", X_train.shape[1])

    acquisition_name = acquisition or function_config.get("acquisition", "UCB")

    # Track all sampled points for debugging
    history = {"X": X_train.copy(), "y": y_train.copy(), "acquisition": []}
   
    if model_type.upper() == "GP":
       surrogate = build_gp(function_config,X_train, y_train,config_override)
    elif model_type.upper() == "SVR":
       surrogate = build_svr(X_train, y_train, function_config,config_override)
    else:
       raise ValueError(f"Unknown surrogate type: {model_type}")

    for i in range(num_iterations):
    
        surrogate.fit(X_train, y_train)

        # X_candidates = generate_candidates(dim, n_candidates, method=candidate_method)
        X_candidates = generate_candidates(dim, n_candidates, determine_candidate_generation_method(dim) )

       
        if model_type.upper() == "GP":
            mu, sigma = surrogate.predict(X_candidates, return_std=True)
        else:  # SVR
            mu = surrogate.predict(X_candidates)
            sigma = np.full_like(mu, 1e-6)  # pseudo-uncertainty

        y_max = np.max(y_train)
        acquisition_values = select_acquisition(acquisition_name, mu, sigma,
                                                iteration=i, y_max=y_max)

 
        if use_boundary and function_config.get("boundary_penalty", True):
            softness = 0.15 * np.exp(-i / 20)
            acquisition_values *= apply_boundary_penalty(X_candidates, softness)

        history["acquisition"].append(acquisition_values.copy())

        
        next_idx = np.argmax(acquisition_values)
        next_point = X_candidates[next_idx]

        
        if model_type.upper() == "SVR":
            y_next = surrogate.predict(next_point.reshape(1, -1))[0]
        else:
            y_next = mu[next_idx]

        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)

        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        if verbose:
            print(f"Iter {i+1:02d} | Next input: {next_point} | Predicted output: {y_next:.6f}")

    # --- Return best found ---
    best_idx = np.argmax(y_train)
    best_input = X_train[best_idx]
    best_output = y_train[best_idx]

    return best_input, best_output, history
def bbo_loopWith(X_train, y_train, function_config, acquisition=None, num_iterations=30,
             model_type="GP", config_override=None, use_boundary=True,
             n_candidates=500, candidate_method="random", verbose=True):
    """
    Modular black-box optimization loop using GP  surrogate.

    Returns:
        best_input, best_output, history (dict with all sampled points)
    """
    X_train = X_train.copy()
    y_train = y_train.copy()
    dim = function_config.get("dim", X_train.shape[1])

    acquisition_name = acquisition or function_config.get("acquisition", "UCB")

    # Track all sampled points for debugging
    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [np.max(y_train)],
        "acquisition": [],
        "iter": list(range(len(y_train))),
        "kernel_": []
    }

   
  
    for i in range(num_iterations):

    # --- Build GP and optimize kernel dynamically ---
        surrogate = build_dynamic_gp(
                    X_train, y_train,
                    config=function_config,
                    iteration=i,
                    total_iterations=num_iterations
                                     )

    # --- Fit GP to current data ---
        surrogate.fit(X_train, y_train)
        history["kernel_"].append(str(surrogate.kernel_))

    # --- Predict for candidates ---
        # X_candidates = generate_candidates(dim, n_candidates, method=candidate_method)
        X_candidates = generate_candidates(dim,n_candidates, determine_candidate_generation_method(dim) )
        mu, sigma = surrogate.predict(X_candidates, return_std=True)

    # --- Dynamic acquisition function (exploration/exploitation) ---
        initial_kappa = 3.0
        final_kappa = 0.1
        kappa = initial_kappa * np.exp(-i / num_iterations) + final_kappa
        acquisition_values = select_acquisition(acquisition_name, mu, sigma,
                                            iteration=i, y_max=np.max(y_train), kappa=kappa)

    # --- Boundary penalty etc ---
        if use_boundary:
            softness = 0.15 * np.exp(-i / 20)
            acquisition_values *= apply_boundary_penalty(X_candidates, softness)

    # --- Choose next point ---
        next_idx = np.argmax(acquisition_values)
        next_point = X_candidates[next_idx]
        y_next = mu[next_idx]

    # --- Update training data ---
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)

    # --- Return best found ---
        best_idx = np.argmax(y_train)
        best_input = X_train[best_idx]
        best_output = y_train[best_idx]
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        history["best_y"].append(np.max(y_train))
        history["iter"].append(len(history["iter"]))

        print(f"Iter {i+1}: Optimized kernel = {surrogate.kernel_}") 
    return best_input, best_output, history


import numpy as np

def function_free_bbo_dynamic(X_init,y_init, function_config=None, num_iterations=30, n_candidates=1000, verbose=True, use_seed=True, seed=42):
    """
    X_init: initial seed points (2D array)
    function_config: dict with GP/kernel config (length scales, white noise, etc.)
    """
    X_train = X_init.copy()
    dim = X_train.shape[1]

    # Initialize dummy outputs (just for GP fitting)
    y_train = y_init
    
    # Build initial GP using your dynamic GP + kernel builder
    gp = build_gpWhiteKernel(config=function_config,X_train=X_train, y_train=y_train,  kernel_override=None, use_seed=use_seed, seed=seed)
    
    # History tracking
    history = {
        "X": X_train.copy(),
        "pred_mu": y_train.copy(),
        "pred_sigma": np.zeros_like(y_train),
        "kernel": [str(gp.kernel_)]
    }

    for i in range(num_iterations):
        # Generate random candidates
        X_candidates = generate_candidates(dim,n_candidates, determine_candidate_generation_method(dim) )
        # X_candidates = np.random.rand(n_candidates, dim)
        
        # GP predictions
        mu, sigma = gp.predict(X_candidates, return_std=True)
        
        # Exploration-exploitation factor (decaying)
        beta = 2.0 * np.exp(-0.1 * i)
        acquisition = mu + beta * sigma
        
        # Pick next candidate
        next_idx = np.argmax(acquisition)
        next_point = X_candidates[next_idx]
        y_next = mu[next_idx]  # pseudo-output from GP mean

        # Append to training set
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)
        
        # Rebuild GP dynamically (kernel can adjust)
        gp = build_gpWhiteKernel( config=function_config,X_train=X_train, y_train=y_train, kernel_override=None, use_seed=use_seed, seed=seed)
        
        # Update history
        history["X"] = np.vstack([history["X"], next_point])
        history["pred_mu"] = np.append(history["pred_mu"], y_next)
        history["pred_sigma"] = np.append(history["pred_sigma"], sigma[next_idx])
        history["kernel"].append(str(gp.kernel_))
        
        if verbose:
            print(f"Iter {i+1:02d} | Next input: {next_point} | Predicted output: {y_next:.4f} | Kernel: {gp.kernel_}")

    # Return top candidate according to predicted mean
    best_idx = np.argmax(history["pred_mu"])
    best_input = history["X"][best_idx]
    best_output = history["pred_mu"][best_idx]

    return best_input, best_output, history


def function_free_bbo_multi_acq(
    X_init, y_init=None, config=None, num_iterations=30, n_candidates=1000, verbose=True, seed=42
):
    np.random.seed(seed)
    input_dim = X_init.shape[1]

    # Initialize training data
    X_train = X_init.copy()
    if y_init is None:
        y_train = np.random.rand(len(X_train))  # dummy output
    else:
        y_train = y_init.copy()

    history = {"X": X_train.copy(), "pred_mu": y_train.copy(), "pred_sigma": np.zeros_like(y_train)}

    # GP model setup
    gp = build_gpWhiteKernel( config=config,X_train=X_train, y_train=y_train, kernel_override=None)
    gp.fit(X_train, y_train)

    acquisition_list = ["EI", "PI", "UCB", "THOMPSON"]

    for i in range(num_iterations):
        # Generate candidates (Latin Hypercube / uniform)
        # X_candidates = np.random.rand(n_candidates, input_dim)
        X_candidates = generate_candidates(input_dim,n_candidates, determine_candidate_generation_method(input_dim) )

        best_per_acq = {}

        # Evaluate each acquisition function
        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point = None

            for candidate in X_candidates:
                mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
                y_max = np.max(y_train)

                if acq_name == "THOMPSON":
                    acq_value = acquisition_thompson(mu, sigma)
                else:
                    acq_value = select_acquisition(acq_name, mu, sigma=sigma, iteration=i, y_max=y_max)
                # penalty = apply_boundary_penalty(candidate.reshape(1, -1))[0]  # scalar in [0,1]
                # acq_value *= penalty  #
                if acq_value > best_value:
                    best_value = acq_value
                    best_point = candidate

            best_per_acq[acq_name] = (best_point, best_value)

        # Pick best candidate across all acquisitions
        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point = best_per_acq[best_acq_name][0]

        # Predict output for history tracking
        next_mu, next_sigma = gp.predict(next_point.reshape(1, -1), return_std=True)
        y_next = next_mu.item()

        # Update training data
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)

        # Refit GP with updated data
        kernel = build_kernelWithWhiteKernel(config=config, input_dim=input_dim, X_train=X_train, y_train=y_train, iteration=i, total_iterations=num_iterations)
        gp.kernel_ = kernel
        gp.fit(X_train, y_train)

        # Update history
        history["X"] = np.vstack([history["X"], next_point])
        history["pred_mu"] = np.append(history["pred_mu"], y_next)
        history["pred_sigma"] = np.append(history["pred_sigma"], next_sigma.item())

        if verbose:
            print(f"Iter {i+1:02d} | Best acquisition: {best_acq_name} | Next input: {next_point} | Predicted output: {y_next:.6f}")

    # Return best observed point
    best_idx = np.argmax(history["pred_mu"])
    best_input = history["X"][best_idx]
    best_output = history["pred_mu"][best_idx]

    return best_input, best_output, history




def adaptive_bbo_dynamic_with_risk(
    X_init,              
    y_init,               
    config,                  
    acquisition_list=["EI","UCB","PI","THOMPSON"],
    num_iterations=30,
    random_state=42,
    n_candidates=500,
    min_improve_prob=0.7,
    mc_draws=2000
):
    rng = np.random.RandomState(random_state)

    # 1. Scale inputs/outputs for GP training
    X_train, X_scaler = scale_data(X_init, scaler_type='minmax')
    y_train_scaled, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard')
    y_train = y_train_scaled.ravel()

    # History in original scale
    history = {"X": X_init.copy(), "y": np.array(y_init, copy=True)}

    for i in range(num_iterations):
        # 2. Build and fit GP surrogate
        gp = build_gpWhiteKernel(config, X_train, y_train)
        gp.fit(X_train, y_train)

        d = X_train.shape[1]  # input dimension

        # 3. Generate candidate pool (original space) + scale them
        X_cand_orig = generate_candidates(d, n_candidates,
                                          determine_candidate_generation_method(d))
        X_cand = X_scaler.transform(X_cand_orig)

        # 4. Compute GP prediction (mu, sigma) for all candidates
        mu, sigma = gp.predict(X_cand, return_std=True)

        # 5. Compute acquisition values for all candidates
        acq_vals = np.zeros(len(X_cand))
        y_max_train = np.max(y_train)  # best seen (scaled) output

        for acq_name in acquisition_list:
            for j in range(len(X_cand)):
                m = mu[j]
                s = sigma[j]
                if acq_name.upper() == "UCB":
                    # example UCB — adjust kappa schedule if needed
                    kappa = 2.0
                    val = m + kappa * s
                elif acq_name.upper() == "EI":
                    val = acquisition_ei(m, s, y_max_train, xi=0.01)
                elif acq_name.upper() == "PI":
                    val = acquisition_pi(m, s, y_max_train, eta=0.01)
                elif acq_name.upper() == "THOMPSON":
                    val = m + s * rng.randn()
                else:
                    raise ValueError("Unknown acquisition: " + acq_name)
                acq_vals[j] = max(acq_vals[j], val)

        #  Risk filter: estimate probability candidate produces improvement over actual best-observed
        y_best_obs = np.max(history["y"])  # best in original scale
        # Monte‑Carlo sample outputs from GP predictive distribution (in scaled output space!)
        draws = rng.normal(loc=mu[:, None], scale=sigma[:, None], size=(len(X_cand), mc_draws))
        # Convert draws back to original scale
        draws_orig = y_s_inv = y_scaler.inverse_transform(draws)
        # Compare to y_best_obs
        success_probs = (draws_orig > y_best_obs).mean(axis=1)

        # 7. Select next point: filter by min_improve_prob, fallback to best acquisition
        idx_good = np.where(success_probs >= min_improve_prob)[0]
        if len(idx_good) > 0:
            idx_next = idx_good[np.argmax(acq_vals[idx_good])]
        else:
            idx_next = np.argmax(acq_vals)

        next_x_orig = X_cand_orig[idx_next]
        next_x_scaled = X_cand[idx_next].reshape(1, -1)

        # 8. (In real BO) evaluate real black‑box: here we only use GP prediction
        y_next_scaled = gp.predict(next_x_scaled).item()
        y_next = y_s_inv = y_scaler.inverse_transform([[y_next_scaled]])[0,0]

        # 9. Update training data & history
        X_train = np.vstack([X_train, next_x_scaled])
        y_train = np.append(y_train, y_next_scaled)

        history["X"] = np.vstack([history["X"], next_x_orig])
        history["y"] = np.append(history["y"], y_next)

        print(f"Iter {i+1} | next_x = {next_x_orig}, predicted y = {y_next:.4f}, P(improve) = {success_probs[idx_next]:.2%}")

    # After loop, return best observed
    best_idx = np.argmax(history["y"])
    return history["X"][best_idx], history["y"][best_idx], history


def svr_bbo_loop(X_init, y_init,config,  n_iterations=30, n_candidates=1000, verbose=True, seed=42):
    """
    Bayesian optimization loop using SVR surrogate model.
    Returns best input/output and history.
    """
    np.random.seed(seed)
    X_train = X_init.copy()
    y_train = y_init.copy()
    dim = X_train.shape[1]

    # Initialize SVR surrogate
    svr =  build_svr(X_train, y_train, config=config, config_override=None)
    svr.fit(X_train, y_train)

    # History
    history = {"X": X_train.copy(), "y_pred": y_train.copy()}

    for i in range(n_iterations):
        # Generate candidates in [0,1]^dim
        
        X_candidates = generate_candidates(dim, n_candidates, determine_candidate_generation_method(dim) )

        # Predict mean
        mu = svr.predict(X_candidates)

        # Apply boundary penalty
        # penalty = apply_boundary_penalty(X_candidates)
        # mu_penalized = mu * penalty

        # # Choose next point (max exploitation)
        # next_idx = np.argmax(mu_penalized)
        next_idx = np.argmax(mu)
        next_point = X_candidates[next_idx]
        y_next = mu[next_idx]  # SVR is deterministic, no sigma

        # Update dataset
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)
        svr.fit(X_train, y_train)

        # Update history
        history["X"] = np.vstack([history["X"], next_point])
        history["y_pred"] = np.append(history["y_pred"], y_next)

        if verbose:
            print(f"Iter {i+1:02d} | Next input: {next_point} | Predicted output: {y_next:.6f}")

    # Return best candidate
    best_idx = np.argmax(history["y_pred"])
    best_input = history["X"][best_idx]
    best_output = history["y_pred"][best_idx]

    return best_input, best_output, history


def bbo_loop_ForceInwards(X_train, y_train, function_config, acquisition=None, num_iterations=30,
             model_type="GP", config_override=None, use_boundary=True,
             n_candidates=500, candidate_method="random", verbose=True):
    """
    Modular black-box optimization loop using GP or SVR surrogate.

    Returns:
        best_input, best_output, history (dict with all sampled points)
    """
    X_train = X_train.copy()
    y_train = y_train.copy()
    dim = function_config.get("dim", X_train.shape[1])

    acquisition_name = acquisition or function_config.get("acquisition", "UCB")

    # Track all sampled points for debugging
    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [np.max(y_train)],
        "acquisition": [],
        "iter": list(range(len(y_train))),
        "kernel_": []
    }

   
  
    for i in range(num_iterations):

    # --- Build GP and optimize kernel dynamically ---
        surrogate = build_dynamic_gp(
                    X_train, y_train,
                    config=function_config,
                    iteration=i,
                    total_iterations=num_iterations
                                     )

    # --- Fit GP to current data ---
        surrogate.fit(X_train, y_train)
        history["kernel_"].append(str(surrogate.kernel_))

    # --- Predict for candidates ---
        candidate_method = determine_candidate_generation_method(dim)
        X_candidates = generate_candidates(dim, n_candidates, method=candidate_method)
        mu, sigma = surrogate.predict(X_candidates, return_std=True)

        initial_kappa = 3.0
        final_kappa = 0.1
        kappa = initial_kappa * np.exp(-i / num_iterations) + final_kappa

# Base acquisition
        acquisition_values = select_acquisition(acquisition_name,
                                        mu, sigma,
                                        iteration=i,
                                        y_max=np.max(y_train),
                                        kappa=kappa)

# Apply boundary penalty if needed
        if use_boundary:
             softness = 0.15 * np.exp(-i / 20)
             acquisition_values *= apply_boundary_penalty(X_candidates, softness)

# Apply middle boost
        boost_middle = True
        middle_bounds = (0.3, 0.7)
        boost_factor = 2.0
        if boost_middle:
            middle_mask = np.all((X_candidates >= middle_bounds[0]) & (X_candidates <= middle_bounds[1]), axis=1)
            acquisition_values[middle_mask] *= boost_factor

# Pick next point
        next_idx = np.argmax(acquisition_values)
        next_point = X_candidates[next_idx]
        y_next = mu[next_idx]  # or evaluate the real function here

    # --- Update training data ---
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)

    # --- Return best found ---
        best_idx = np.argmax(y_train)
        best_input = X_train[best_idx]
        best_output = y_train[best_idx]
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        history["best_y"].append(np.max(y_train))
        history["iter"].append(len(history["iter"]))

        print(f"Iter {i+1}: Optimized kernel = {surrogate.kernel_}") 
    return best_input, best_output, history


def adaptive_bbo_dynamic(X_init, y_init, config, acquisition_list=["EI","UCB","PI","THOMPSON"], 
                         num_iterations=30, random_state=42):
    np.random.seed(random_state)
    X_train, X_scaler = scale_data(X_init, scaler_type='minmax')
    y_train, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard')

    input_dim = X_init.shape[1]

    history = {"X": X_train.copy(), "y": y_train.copy()}

    for i in range(num_iterations):
        # Build/update GP kernel dynamically
        gp = build_gpWhiteKernel(config, X_train, y_train)
        
        gp.fit(X_train, y_train)

        # Generate candidates
        n_candidates = 500
        # X_candidates = np.random.rand(n_candidates, input_dim)
        X_candidates = generate_candidates(input_dim,n_candidates, determine_candidate_generation_method(input_dim) )

        best_per_acq = {}
        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point = None
            for candidate in X_candidates:
                mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
                mu = mu.item()
                sigma = sigma.item()
                y_max = np.max(y_train)

                # Dynamic exploration/exploitation
                if acq_name.upper() == "UCB":
                    initial_kappa = 3.0
                    final_kappa = 0.1
                    kappa = initial_kappa * np.exp(-i / num_iterations) + final_kappa
                    acq_value = acquisition_ucb_Kappa(mu, sigma, iteration=i, kappa=kappa)
                elif acq_name.upper() == "EI":
                    initial_xi = 0.1
                    final_xi = 0.01
                    xi = initial_xi * np.exp(-i / num_iterations) + final_xi
                    acq_value = acquisition_ei(mu, sigma, y_max, xi=xi)
                elif acq_name.upper() == "PI":
                    initial_eta = 0.1
                    final_eta = 0.01
                    eta = initial_eta * np.exp(-i / num_iterations) + final_eta
                    acq_value = acquisition_pi(mu, sigma, y_max, eta=eta)
                elif acq_name.upper() == "THOMPSON":
                    sigma_dynamic = sigma * (np.exp(-i / num_iterations) + 0.05)
                    acq_value = acquisition_thompson(mu, sigma_dynamic)
                else:
                    raise ValueError(f"Unknown acquisition function: {acq_name}")

                if acq_value > best_value:
                    best_value = acq_value
                    best_point = candidate

           

            best_per_acq[acq_name] = (best_point, best_value)

        # Select the best across all acquisition functions
        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point = best_per_acq[best_acq_name][0]
        # Scale the next candidate point
        next_point_scaled = X_scaler.transform(next_point.reshape(1, -1))

    # Predict with GP on scaled input
        y_next_scaled = gp.predict(next_point_scaled).item()

    # Optionally, inverse scale the predicted output if you want it in original units
        y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item()

    # Update training data (scaled)
        X_train = np.vstack([X_train, next_point_scaled])
        y_train = np.append(y_train, y_next_scaled)  # keep y_train scaled for GP training

    # Update history (store original units for easier tracking)
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        # # Predict output for history
        # y_next = gp.predict(next_point.reshape(1, -1)).item()
        # next_point_scaled = X_scaler.transform(next_point.reshape(1,-1))
        # y_next_scaled = y_scaler.transform(np.array([[y_next]]))


        # # Update training data
        # X_train = np.vstack([X_train, next_point_scaled])
        # y_train = np.append(y_train, y_next_scaled)

        # history["X"] = np.vstack([history["X"], next_point])
        # history["y"] = np.append(history["y"], y_next)

        print(f"Iter {i+1} | Selected {best_acq_name} | Next input: {next_point} | Predicted y: {y_next:.6f}")

    # Return best observed input/output and full history
    best_idx = np.argmax(y_train)
    best_input = X_train[best_idx]
    best_output = y_train[best_idx]

    return best_input, best_output, history



def BOLoop_rf(
    X_init, y_init, input_dim,
    n_iterations=50,
    n_candidates=1000,
    acquisition_methods=["UCB","EI","PI"],
    use_boundary=False,
    boost_middle=False,
    middle_bounds=(0.3,0.7),
    boost_factor=2.0,
    initial_beta=3.0,
    final_beta=0.1
):
    X_train = X_init.copy()
    y_train = y_init.copy()
    dim = input_dim

    history = {"X": [], "y": [], "best_y": [], "iter": []}

    for i in range(n_iterations):
        # --- Fit RF surrogate on current data ---
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, bootstrap=True,
                                   n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)

        # --- Generate candidates ---
        X_candidates = generate_candidates(dim, n_candidates, determine_candidate_generation_method(dim))
        preds = np.array([tree.predict(X_candidates) for tree in rf.estimators_])
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        # --- Dynamic beta ---
        beta = initial_beta * np.exp(-i / n_iterations) + final_beta

        # --- Compute acquisition values ---
        acq_values_dict = {}
        for acq_name in acquisition_methods:
            acq = select_acquisition(acq_name, mu, sigma, iteration=i, y_max=np.max(y_train), kappa=beta)
            if use_boundary:
                softness = 0.15 * np.exp(-i/20)
                acq *= apply_boundary_penalty(X_candidates, softness)
            if boost_middle:
                middle_mask = np.all((X_candidates >= middle_bounds[0]) & (X_candidates <= middle_bounds[1]), axis=1)
                acq[middle_mask] *= boost_factor
            acq_values_dict[acq_name] = acq

        # --- Pick the next new input ---
        best_acq_name = max(acq_values_dict, key=lambda k: acq_values_dict[k].max())
        next_idx = np.argmax(acq_values_dict[best_acq_name])
        next_point = X_candidates[next_idx]
        predicted_output = mu[next_idx]  # predicted by RF

        # --- Save new candidate (no real function call) ---
        history["X"].append(next_point)
        history["y"].append(predicted_output)
        history["best_y"].append(np.max(history["y"]))
        history["iter"].append(i+1)

        # --- Update surrogate training with this new predicted point ---
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, predicted_output)

        print(f"Iter {i+1}: New predicted input, predicted_y={predicted_output:.6f}")

    # Return last generated input/output and full history
    return history["X"][-1], history["y"][-1], history


def bayes_opt_rf_batch(
    X_init, y_init, input_dim,
    n_iterations=50,
    n_candidates=1000,
    batch_size=5,
    acquisition_methods=["UCB","EI","PI"],
    use_boundary=False,
    boost_middle=False,
    middle_bounds=(0.3,0.7),
    boost_factor=2.0,
    initial_beta=3.0,
    final_beta=0.1,
    n_jobs=-1
):
    X_train = X_init.copy()
    y_train = y_init.copy()
    dim = input_dim

    history = {"X": [], "y": [], "best_y": [], "iter": []}

    for i in range(n_iterations):
        # --- Fit RF surrogate ---
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                   bootstrap=True, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)

        # --- Generate candidates ---
        X_candidates = generate_candidates(dim, n_candidates, determine_candidate_generation_method(dim))
        preds = np.array([tree.predict(X_candidates) for tree in rf.estimators_])
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)
        print("candidate generated")

        # --- Dynamic beta ---
        beta = initial_beta * np.exp(-i / n_iterations) + final_beta

        # --- Compute acquisition values for all methods ---
        acq_values_dict = {}
        for acq_name in acquisition_methods:
            acq = select_acquisition(acq_name, mu, sigma, iteration=i, y_max=np.max(y_train), kappa=beta)
            if use_boundary:
                softness = 0.15 * np.exp(-i/20)
                acq *= apply_boundary_penalty(X_candidates, softness)
            if boost_middle:
                middle_mask = np.all((X_candidates >= middle_bounds[0]) & (X_candidates <= middle_bounds[1]), axis=1)
                acq[middle_mask] *= boost_factor
            acq_values_dict[acq_name] = acq

        # --- Combine all acquisition functions: max across methods ---
        combined_acq = np.vstack(list(acq_values_dict.values())).max(axis=0)
        top_indices = combined_acq.argsort()[-batch_size:][::-1]
        X_next_batch = X_candidates[top_indices]
        y_next_batch = mu[top_indices]  # predicted outputs

        # --- Update history ---
        history["X"].extend(X_next_batch)
        history["y"].extend(y_next_batch)
        history["best_y"].append(max(history["y"]))
        history["iter"].append(i+1)


    best_idx = np.argmax(history["y"])
    best_input = history["X"][best_idx]
    best_output = history["y"][best_idx]

    return best_input, best_output, history


def rf_predict(X_train, y_train, input_dim):
  

    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        bootstrap=True
    )
    rf_model.fit(X_train, y_train)

# Candidate points
    
    candidates = generate_candidates(input_dim, n_candidates= 5000,method = 'lhc', )

# Predict mean and std from RF
    all_tree_preds = np.array([tree.predict(candidates) for tree in rf_model.estimators_])
    pred_mean = all_tree_preds.mean(axis=0)
    pred_std = all_tree_preds.std(axis=0)

# Upper Confidence Bound acquisition
    beta = 2.0  # can tune: higher = more exploration
    acquisition = pred_mean + beta * pred_std

    best_idx = np.argmax(acquisition)
    next_input = candidates[best_idx]
    print("Next candidate to try:", next_input)
    print(best_idx)


from .hybridSelector import HybridGP_RF


def BO_hybrid_single(
    X_init,
    y_init,
    input_dim,
    acquisition_methods=["UCB","EI","PI"],
    n_iterations=60,
    n_candidates=2000,
    use_boundary=True,
    boost_middle=True,
    middle_bounds=(0.3,0.7),
    boost_factor=2.0,
    initial_beta=3.0,
    final_beta=0.1,
):

    X_train = X_init.copy()
    y_train = y_init.copy()

    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [y_train.max()],
        "iter": [0]
    }

    candidate_method = determine_candidate_generation_method(input_dim)

    for i in range(n_iterations):

        # --- Fit Hybrid Model ---
        model = HybridGP_RF(input_dim)
        model.fit(X_train, y_train)

        # --- Generate candidate points ---
        X_candidates = generate_candidates(
            input_dim, n_candidates, candidate_method
        )

        # --- Predict ---
        mu, sigma = model.predict(X_candidates)

        # --- Dynamic exploration coefficient ---
        kappa = initial_beta * np.exp(-i / n_iterations) + final_beta

        # --- Evaluate all acquisition functions ---
        acq_dict = {}
        for acq_name in acquisition_methods:
            acq = select_acquisition(
                acq_name, mu, sigma,
                y_max=y_train.max(),
                kappa=kappa,
                iteration=i
            )

            if use_boundary:
                acq *= apply_boundary_penalty(X_candidates)

            if boost_middle:
                mask = np.all((X_candidates >= middle_bounds[0]) &
                              (X_candidates <= middle_bounds[1]), axis=1)
                acq[mask] *= boost_factor

            acq_dict[acq_name] = acq

        # --- Pick best acq method then best point ---
        best_name = max(acq_dict, key=lambda k: acq_dict[k].max())
        next_idx = np.argmax(acq_dict[best_name])
        next_x = X_candidates[next_idx]
        next_y = mu[next_idx]  # PREDICTED! No real function.

        # --- Update dataset ---
        X_train = np.vstack([X_train, next_x])
        y_train = np.append(y_train, next_y)

        # --- Log ---
        history["X"] = np.vstack([history["X"], next_x])
        history["y"] = np.append(history["y"], next_y)
        history["best_y"].append(y_train.max())
        history["iter"].append(i+1)

        print(f"Iter {i+1}: using {best_name}, predicted y={next_y:.4f}")

    best_idx = np.argmax(y_train)
    return X_train[best_idx], y_train[best_idx], history

# from joblib import Parallel, delayed
# import numpy as np


def BO_hybrid_batch(
    X_init,
    y_init,
    input_dim,
    batch_size=5,
    n_iterations=50,
    n_candidates=2000,
    acquisition_methods=["UCB","EI","PI"],
    use_boundary=True,
    boost_middle=True,
    middle_bounds=(0.3,0.7),
    boost_factor=2.0,
    initial_beta=3.0,
    final_beta=0.1,
    n_jobs=-1
):

    X_train = X_init.copy()
    y_train = y_init.copy()

    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [y_train.max()],
        "iter": [0]
    }

    candidate_method = determine_candidate_generation_method(input_dim)

    for i in range(n_iterations):

        model = HybridGP_RF(input_dim)
        model.fit(X_train, y_train)

        X_candidates = generate_candidates(input_dim, n_candidates, candidate_method)
        mu, sigma = model.predict(X_candidates)

        kappa = initial_beta * np.exp(-i / n_iterations) + final_beta

        # All acquisition functions -> choose max over them
        acq_stack = []
        for acq_name in acquisition_methods:
            acq = select_acquisition(
                acq_name, mu, sigma, y_max=y_train.max(), kappa=kappa
            )
            if use_boundary:
                acq *= apply_boundary_penalty(X_candidates)
            if boost_middle:
                mask = np.all((X_candidates >= middle_bounds[0]) &
                              (X_candidates <= middle_bounds[1]), axis=1)
                acq[mask] *= boost_factor
            acq_stack.append(acq)

        combined_acq = np.vstack(acq_stack).max(axis=0)
        top_idx = combined_acq.argsort()[-batch_size:][::-1]
        X_batch = X_candidates[top_idx]
        y_batch = mu[top_idx]  # predicted

        X_train = np.vstack([X_train, X_batch])
        y_train = np.append(y_train, y_batch)

        history["X"] = np.vstack([history["X"], X_batch])
        history["y"] = np.append(history["y"], y_batch)
        history["best_y"].append(y_train.max())
        history["iter"].append(i+1)

        print(f"Iter {i+1}: Batch added, best_y={y_train.max():.4f}")

    best_idx = y_train.argmax()
    return X_train[best_idx], y_train[best_idx], history


from .ModelComittee import CommitteeSurrogate



def BO_committee_single(
    X_init,
    y_init,
    input_dim,
    acquisition_methods=["UCB","EI","PI"],
    n_iterations=60,
    n_candidates=2000,
    use_boundary=False,
    boost_middle=False,
    middle_bounds=(0.3,0.7),
    boost_factor=2.0,
    initial_beta=3.0,
    final_beta=0.1
):

    X_train = X_init.copy()
    y_train = y_init.copy()

    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [y_train.max()],
        "iter": [0]
    }

    cand_method = determine_candidate_generation_method(input_dim)

    for i in range(n_iterations):

        # --- Train committee ---
        model = CommitteeSurrogate(input_dim)
        model.fit(X_train, y_train)

        # --- Candidate generation ---
        X_candidates = generate_candidates(input_dim, n_candidates, cand_method)

        # --- Predict ---
        mu, sigma = model.predict(X_candidates)

        # --- Dynamic UCB beta ---
        kappa = initial_beta * np.exp(-i / n_iterations) + final_beta

        # --- Acquisition over all models ---
        acq_dict = {}
        for acq_name in acquisition_methods:
            acq = select_acquisition(
                acq_name, mu, sigma,
                y_max=y_train.max(),
                kappa=kappa,
                iteration=i
            )

            if use_boundary:
                acq *= apply_boundary_penalty(X_candidates)

            if boost_middle:
                mask = np.all(
                    (X_candidates >= middle_bounds[0]) &
                    (X_candidates <= middle_bounds[1]), axis=1
                )
                acq[mask] *= boost_factor

            acq_dict[acq_name] = acq

        # --- Best acquisition method ---
        best_acq = max(acq_dict, key=lambda k: acq_dict[k].max())
        next_idx = np.argmax(acq_dict[best_acq])

        next_x = X_candidates[next_idx]
        next_y = mu[next_idx]  # predicted

        # --- Update ---
        X_train = np.vstack([X_train, next_x])
        y_train = np.append(y_train, next_y)

        history["X"] = np.vstack([history["X"], next_x])
        history["y"] = np.append(history["y"], next_y)
        history["best_y"].append(y_train.max())
        history["iter"].append(i+1)

        print(f"Iter {i+1}: Model Committee, acq={best_acq}, predicted y={next_y:.5f}")

    best_idx = np.argmax(y_train)
    return X_train[best_idx], y_train[best_idx], history


def bo_committee_single2(
    X_init, y_init, input_dim,
    n_iterations=30,
    n_candidates=2000,
    acquisition="UCB",
    use_boundary=False,
    boost_middle=False
):


    X_train = X_init.copy()
    y_train = y_init.copy()

    surrogate = CommitteeSurrogate(input_dim)

    history = {
        "X": X_train.copy(),
        "y": y_train.copy(),
        "best_y": [np.max(y_train)]
    }

    for t in range(n_iterations):

        surrogate.fit(X_train, y_train)

        X_candidates = generate_candidates(
            input_dim, n_candidates,
            determine_candidate_generation_method(input_dim)
        )

        mu, sigma = surrogate.predict(X_candidates, return_std=True)

        acq = select_acquisition(
            acquisition, mu, sigma,
            iteration=t,
            y_max=np.max(y_train),
            kappa=3.0 * np.exp(-t / n_iterations) + 0.1
        )

        if use_boundary:
            acq *= apply_boundary_penalty(X_candidates)

        next_x = X_candidates[np.argmax(acq)]
        next_y = mu[np.argmax(acq)]     # predicted output

        X_train = np.vstack([X_train, next_x])
        y_train = np.append(y_train, next_y)

        history["X"] = np.vstack([history["X"], next_x])
        history["y"] = np.append(history["y"], next_y)
        history["best_y"].append(np.max(y_train))

        print(f"[Iter {t+1}] Acquisition max = {np.max(acq):.4f}")

    best_idx = np.argmax(y_train)
    return X_train[best_idx], y_train[best_idx], history

def adaptive_bbo_dynamic_full(
    X_init, y_init, config,
    acquisition_list=["EI","UCB","PI","THOMPSON"], 
    # kernel_list=["RBF","Matern","RationalQuadratic"],
    kernel_list=["RBF","Matern"],
    num_iterations=30,
    random_state=42,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=True
):
    """
    Fully adaptive Bayesian Optimization loop with:
        - Multiple kernels
        - Kernel hyperparameter tuning (nu, alpha, C)
        - Multiple acquisition functions
        - Optional candidate scaling
        - Tracks best_results per iteration
    """
    np.random.seed(random_state)

    # Scale inputs/outputs if requested
    X_train, X_scaler = scale_data(X_init, scaler_type='minmax') if scale_candidates else (X_init.copy(), None)
    y_train, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard') if scale_candidates else (np.array(y_init).reshape(-1,1), None)
    input_dim = X_init.shape[1]

    history = {"X": X_train.copy(), "y": y_train.copy()}
    best_results = []

    for i in range(num_iterations):
            gp = build_dynamic_gp(X_train, y_train, config=config, iteration=i, total_iterations=num_iterations, seed=random_state)
            

            params = analyze_for_candidate_generation(X_train, y_train, gp)  
            current_best_x = params['current_best_x']
            current_best_y = params['current_best_y']
            min_prob = params['min_prob']
            local_radius = params['local_radius']
            favor_global = params['favor_global']

# Decide how many candidates
            n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)

            if favor_global:
    # Generate global candidates and filter using estimated success probability
                    X_global = generate_candidates(input_dim, n_candidates, determine_candidate_generation_method(input_dim))
                    X_global_scaled = X_scaler.transform(X_global) if scale_candidates else X_global.copy()
                    prob, mu, sigma = estimate_success_prob(gp, X_global_scaled, current_best_y)
                    idx_pass = prob >= min_prob
                    X_candidates_scaled = X_global_scaled[idx_pass]
            else:
    # Generate local candidates around current best
                    X_candidates_scaled = generate_candidates(
                            input_dim,
                            n_candidates // 2,
                            method="local",
                            local_center=current_best_x,
                            local_radius=local_radius
                            )
                    if scale_candidates:
                            X_candidates_scaled = X_scaler.transform(X_candidates_scaled)
                    
            best_per_acq = {}

        # Loop over acquisition functions
            for acq_name in acquisition_list:
                best_value = -np.inf
                best_point = None
                best_kernel_name = None
                best_params = None

            # Loop over kernels
            for kernel_name in kernel_list:
                cfg_copy = config.copy()
                cfg_copy["kernel_type"] = kernel_name

                # Dynamic kernel hyperparameters
                if kernel_name == "Matern":
                    cfg_copy["nu"] = np.random.choice([1.5, 2.5])
                elif kernel_name == "RationalQuadratic":
                    cfg_copy["alpha_rq"] = np.random.uniform(0.5, 2.0)
                cfg_copy["C"] = np.random.uniform(0.1, 3.0)

                # Build GP dynamically
                gp = build_dynamic_gp(X_train, y_train, config=cfg_copy, iteration=i, total_iterations=num_iterations, seed=random_state)

                # Evaluate all candidates
                for candidate in X_candidates_scaled:
                    mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
                    mu, sigma = mu.item(), sigma.item()
                    y_max = np.max(y_train)

                    # Acquisition function
                    if acq_name.upper() == "UCB":
                        kappa = 3.0 * np.exp(-i / num_iterations) + 0.1
                        acq_value = acquisition_ucb_Kappa(mu, sigma, iteration=i, kappa=kappa)
                    elif acq_name.upper() == "EI":
                        xi = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_ei(mu, sigma, y_max, xi=xi)
                    elif acq_name.upper() == "PI":
                        eta = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_pi(mu, sigma, y_max, eta=eta)
                    elif acq_name.upper() == "THOMPSON":
                        sigma_dynamic = sigma * (np.exp(-i / num_iterations) + 0.05)
                        acq_value = acquisition_thompson(mu, sigma_dynamic)
                    else:
                        raise ValueError(f"Unknown acquisition function: {acq_name}")

                    # Track best candidate for this acquisition
                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = candidate
                        best_kernel_name = kernel_name
                        best_params = cfg_copy.copy()

            best_per_acq[acq_name] = (best_point, best_value, best_kernel_name, best_params)

        # Select best across acquisitions
            best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
            next_point_scaled, _, kernel_used, kernel_params = best_per_acq[best_acq_name]
            next_point = X_scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten() if scale_candidates else next_point_scaled.copy()

        # Fit GP with selected kernel/hyperparams to predict next y
            gp = build_dynamic_gp(X_train, y_train, config=kernel_params, iteration=i, total_iterations=num_iterations, seed=random_state)
            y_next_scaled = gp.predict(next_point_scaled.reshape(1, -1)).item()
            y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item() if scale_candidates else y_next_scaled

        # Update training data
            X_train = np.vstack([X_train, next_point_scaled])
            y_train = np.append(y_train, y_next_scaled)
            history["X"] = np.vstack([history["X"], next_point])
            history["y"] = np.append(history["y"], y_next)

        # Record iteration best
            best_idx_iter = np.argmax(y_train)
            best_input_iter = X_scaler.inverse_transform(X_train[best_idx_iter].reshape(1,-1)).flatten() if scale_candidates else X_train[best_idx_iter]
            best_output_iter = y_scaler.inverse_transform(np.array([[y_train[best_idx_iter]]])).item() if scale_candidates else y_train[best_idx_iter]

            best_results.append({
            "iteration": i+1,
            "best_input": best_input_iter,
            "best_output": best_output_iter,
            "kernel": kernel_used,
            "kernel_params": kernel_params,
            "acquisition": best_acq_name,
            "n_candidates": n_candidates
            })

            print(f"Iter {i+1} | Acq: {best_acq_name} | Kernel: {kernel_used} | Next input: {next_point} | Predicted y: {best_output_iter:.6f} | Candidates: {n_candidates}")

    # Final overall best
    best_idx = np.argmax(y_train)
    best_input = X_scaler.inverse_transform(X_train[best_idx].reshape(1,-1)).flatten() if scale_candidates else X_train[best_idx]
    best_output = y_scaler.inverse_transform(np.array([[y_train[best_idx]]])).item() if scale_candidates else y_train[best_idx]

    return best_input, best_output, history, best_results



def bo_exploit_with_rare_explore(
      X_init, y_init, config,
      n_iter = 50,
      n_candidates = 500,
      exploit_fraction = 0.8,     # fraction of iterations to exploit
      random_state = 42
):
    rng = np.random.RandomState(random_state)

    history = {"X": X_init.copy(), "y": np.array(y_init, copy=True)}

    for i in range(n_iter):

        # Build and fit GP surrogate on training data
        gp = build_gpWhiteKernel(config, X_train, y_train)
        gp.fit(X_train, y_train)

        dim = X_train.shape[1]

        # Generate a candidate pool in original domain, then scale
        X_cand = generate_candidates(dim, n_candidates,
                                          method=determine_candidate_generation_method(dim))
       

        # Predict mean and std for all candidates
        mu, sigma = gp.predict(X_cand, return_std=True)

        # Decide whether this iteration will be exploit or explore
        if rng.rand() < exploit_fraction:
            mode = "exploit"
        else:
            mode = "explore"

        if mode == "exploit":
            # Choose candidate(s) that maximize predicted mean (greedy exploitation)
            idx_next = np.argmax(mu)
        else:
            # Exploration mode: choose candidate(s) that maximize uncertainty (std)
            idx_next = np.argmax(sigma)

        next_x= X_cand[idx_next]
      

  
        y_next = gp.predict(next_x).item()
      

        # Update training data & history
        X_train = np.vstack([X_train, next_x])
        y_train = np.append(y_train, y_next)

        history["X"] = np.vstack([history["X"], next_x])
        history["y"] = np.append(history["y"], y_next)

        print(f"[Iter {i+1:2d}] mode={mode}  x_next={next_x}  pred_y={y_next:.4f}")

    best_idx = np.argmax(history["y"])
    return history["X"][best_idx], history["y"][best_idx], history


def log_iteration_csv(writer, iteration, acq_name, next_x, mu, sigma, acq_val, y_pred):
    writer.writerow([
        iteration, acq_name, next_x.tolist(), mu, sigma, acq_val, y_pred
    ])

def adaptive_bbo_with_csv_logging(
    X_init, y_init, config,
    acquisition_list=["EI","UCB","PI","THOMPSON"],
    kernel_list=["RBF","Matern"],
    num_iterations=30,
    random_state=42,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=True,
    export_prefix="bo_history",
    test_size=0.2
):
    np.random.seed(random_state)

    # Scaling
    X_train_all, X_scaler = scale_data(X_init, scaler_type='minmax') if scale_candidates else (X_init.copy(), None)
    y_train_all, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard') if scale_candidates else (np.array(y_init).reshape(-1,1), None)
    input_dim = X_init.shape[1]

    history = {"X": X_train_all.copy(), "y": y_train_all.copy()}
    best_results = []

    # Prepare CSV file for iteration logging
    csv_file = f"{export_prefix}_iter.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "acquisition", "x_next", "mu", "sigma", "acq_val", "y_pred", "rmse_test", "corr_test"])

    for i in range(num_iterations):
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_all, y_train_all, test_size=test_size, random_state=random_state+i
        )

        # Candidate generation
        params = analyze_for_candidate_generation(X_train, y_train, gp=None)
        current_best_x = params['current_best_x']
        current_best_y = params['current_best_y']
        min_prob = params['min_prob']
        local_radius = params['local_radius']
        favor_global = params['favor_global']

        n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)

        if favor_global:
            X_global = generate_candidates(input_dim, n_candidates, determine_candidate_generation_method(input_dim))
            X_global_scaled = X_scaler.transform(X_global) if scale_candidates else X_global.copy()
            X_candidates_scaled = X_global_scaled  # simplified; could include success-prob filtering
        else:
            X_candidates_scaled = generate_candidates(
                input_dim, n_candidates // 2, method="local",
                local_center=current_best_x, local_radius=local_radius
            )
            if scale_candidates:
                X_candidates_scaled = X_scaler.transform(X_candidates_scaled)

        best_per_acq = {}

        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point = None
            best_kernel_name = None
            best_params = None

            for kernel_name in kernel_list:
                cfg_copy = config.copy()
                cfg_copy["kernel_type"] = kernel_name
                if kernel_name == "Matern":
                    cfg_copy["nu"] = np.random.choice([1.5, 2.5])
                elif kernel_name == "RationalQuadratic":
                    cfg_copy["alpha_rq"] = np.random.uniform(0.5, 2.0)
                cfg_copy["C"] = np.random.uniform(0.1, 3.0)

                gp_inner = build_dynamic_gp(X_train, y_train, config=cfg_copy, iteration=i, total_iterations=num_iterations, seed=random_state)

                for candidate in X_candidates_scaled:
                    mu, sigma = gp_inner.predict(candidate.reshape(1, -1), return_std=True)
                    mu, sigma = mu.item(), sigma.item()
                    y_max = np.max(y_train)

                    if acq_name.upper() == "UCB":
                        kappa = 3.0 * np.exp(-i / num_iterations) + 0.1
                        acq_value = acquisition_ucb_Kappa(mu, sigma, iteration=i, kappa=kappa)
                    elif acq_name.upper() == "EI":
                        xi = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_ei(mu, sigma, y_max, xi=xi)
                    elif acq_name.upper() == "PI":
                        eta = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_pi(mu, sigma, y_max, eta=eta)
                    elif acq_name.upper() == "THOMPSON":
                        sigma_dynamic = sigma * (np.exp(-i / num_iterations) + 0.05)
                        acq_value = acquisition_thompson(mu, sigma_dynamic)
                    else:
                        raise ValueError(f"Unknown acquisition function: {acq_name}")

                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = candidate
                        best_kernel_name = kernel_name
                        best_params = cfg_copy.copy()

            best_per_acq[acq_name] = (best_point, best_value, best_kernel_name, best_params)

        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point_scaled, _, kernel_used, kernel_params = best_per_acq[best_acq_name]
        next_point = X_scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten() if scale_candidates else next_point_scaled.copy()

        # Predict next y using GP trained on full data
        gp_full = build_dynamic_gp(X_train_all, y_train_all, config=kernel_params, iteration=i, total_iterations=num_iterations, seed=random_state)
        y_next_scaled = gp_full.predict(next_point_scaled.reshape(1, -1)).item()
        y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item() if scale_candidates else y_next_scaled

        # Evaluate GP on test set
        y_pred_test, _ = gp_full.predict(X_test, return_std=True)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        corr_test = np.corrcoef(y_test.flatten(), y_pred_test.flatten())[0,1]

        # Update data
        X_train_all = np.vstack([X_train_all, next_point_scaled])
        y_train_all = np.append(y_train_all, y_next_scaled)
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        best_results.append({
            "iteration": i+1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": kernel_used,
            "kernel_params": kernel_params,
            "acquisition": best_acq_name,
            "rmse_test": rmse_test,
            "corr_test": corr_test
        })

        # Log to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1, best_acq_name, next_point.tolist(), mu, sigma, best_value, y_next, rmse_test, corr_test
            ])
        print(f"Iter {i+1}: y_pred={y_next:.4f}, RMSE_test={rmse_test:.4f}, Corr_test={corr_test:.3f}")

    # Save full history
    X_all = np.vstack([X_init, np.array(history["X"])])
    y_all = np.concatenate([np.array(y_init), np.array(history["y"])])
    full_csv = f"{export_prefix}_all.csv"
    df = pd.DataFrame(X_all, columns=[f"x{i}" for i in range(X_all.shape[1])])
    df["y"] = y_all
    df.to_csv(full_csv, index=False)
    np.savez(f"{export_prefix}.npz", X=X_all, y=y_all)

    best_idx = np.argmax(y_all)
    best_x = X_all[best_idx]
    best_y = y_all[best_idx]

    return best_x, best_y, history, best_results
def adaptive_bbo_with_phases(
    X_init, y_init, config,
    acquisition_list=["EI","UCB","PI","THOMPSON"],
    kernel_list=["RBF","Matern"],
    num_iterations=30,
    random_state=42,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=False,
    export_prefix="bo_history",
    test_size=0.2
):
    """
    Adaptive BBO with dynamic kernels, acquisitions, logging, test evaluation, and
    automatic exploration/exploitation phase changes.
    """
    np.random.seed(random_state)

    # Scaling
    X_train_all, X_scaler = scale_data(X_init, scaler_type='minmax') if scale_candidates else (X_init.copy(), None)
    y_train_all, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard') if scale_candidates else (np.array(y_init).reshape(-1,1), None)
    input_dim = X_init.shape[1]

    history = {"X": X_train_all.copy(), "y": y_train_all.copy()}
    best_results = []

    # CSV setup
    csv_file = f"{export_prefix}_iter.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "phase", "acquisition", "x_next", "mu", "sigma", "acq_val", "y_pred", "rmse_test", "corr_test"])

    for i in range(num_iterations):
        # --- Phase control ---
        progress = i / num_iterations
        if progress < 0.5:
            # Initial phase: mostly global exploration
            global_frac = 0.8
            local_radius = 0.2
            acq_phase = "UCB"
        elif progress < 0.75:
            # Mid phase: mixed
            global_frac = 0.4
            local_radius = 0.1
            acq_phase = "UCB"  # or "EI"
        else:
            # Final phase: mostly local refinement
            global_frac = 0.2
            local_radius = 0.05
            acq_phase = "EI"  # or predicted mean

        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_all, y_train_all, test_size=test_size, random_state=random_state+i
        )

        # Candidate generation
        params = analyze_for_candidate_generation(X_train, y_train, gp=None)
        current_best_x = params['current_best_x']
        favor_global = params['favor_global']
        n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)

        if favor_global:
            n_global = int(n_candidates * global_frac)
            X_global = generate_candidates(input_dim, n_global, determine_candidate_generation_method(input_dim))
            X_candidates_scaled = X_scaler.transform(X_global) if scale_candidates else X_global.copy()
        else:
            n_local = n_candidates - int(n_candidates * global_frac)
            X_local = generate_candidates(input_dim, n_local, method="local", local_center=current_best_x, local_radius=local_radius)
            if scale_candidates:
                X_local = X_scaler.transform(X_local)
            X_candidates_scaled = X_local

        best_per_acq = {}
        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point = None
            best_kernel_name = None
            best_params = None

            for kernel_name in kernel_list:
                cfg_copy = config.copy()
                cfg_copy["kernel_type"] = kernel_name
                if kernel_name == "Matern":
                    cfg_copy["nu"] = np.random.choice([1.5, 2.5])
                elif kernel_name == "RationalQuadratic":
                    cfg_copy["alpha_rq"] = np.random.uniform(0.5, 2.0)
                cfg_copy["C"] = np.random.uniform(0.1, 3.0)

                gp_inner = build_dynamic_gp(X_train, y_train, config=cfg_copy, iteration=i, total_iterations=num_iterations, seed=random_state)

                for candidate in X_candidates_scaled:
                    mu, sigma = gp_inner.predict(candidate.reshape(1, -1), return_std=True)
                    mu, sigma = mu.item(), sigma.item()
                    y_max = np.max(y_train)

                    if acq_name.upper() == "UCB":
                        kappa = 3.0 * np.exp(-i / num_iterations) + 0.1
                        acq_value = acquisition_ucb_Kappa(mu, sigma, iteration=i, kappa=kappa)
                    elif acq_name.upper() == "EI":
                        xi = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_ei(mu, sigma, y_max, xi=xi)
                    elif acq_name.upper() == "PI":
                        eta = 0.1 * np.exp(-i / num_iterations) + 0.01
                        acq_value = acquisition_pi(mu, sigma, y_max, eta=eta)
                    elif acq_name.upper() == "THOMPSON":
                        sigma_dynamic = sigma * (np.exp(-i / num_iterations) + 0.05)
                        acq_value = acquisition_thompson(mu, sigma_dynamic)
                    else:
                        raise ValueError(f"Unknown acquisition function: {acq_name}")

                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = candidate
                        best_kernel_name = kernel_name
                        best_params = cfg_copy.copy()

            best_per_acq[acq_name] = (best_point, best_value, best_kernel_name, best_params)

        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point_scaled, _, kernel_used, kernel_params = best_per_acq[best_acq_name]
        next_point = X_scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten() if scale_candidates else next_point_scaled.copy()

        # Predict next y using full data GP
        gp_full = build_dynamic_gp(X_train_all, y_train_all, config=kernel_params, iteration=i, total_iterations=num_iterations, seed=random_state)
        y_next_scaled = gp_full.predict(next_point_scaled.reshape(1, -1)).item()
        y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item() if scale_candidates else y_next_scaled

        # Evaluate on test set
        y_pred_test, _ = gp_full.predict(X_test, return_std=True)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        corr_test = np.corrcoef(y_test.flatten(), y_pred_test.flatten())[0,1]

        # Update training data
        X_train_all = np.vstack([X_train_all, next_point_scaled])
        y_train_all = np.append(y_train_all, y_next_scaled)
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        best_results.append({
            "iteration": i+1,
            "phase": f"{global_frac:.2f}-{local_radius:.2f}",
            "best_input": next_point,
            "best_output": y_next,
            "kernel": kernel_used,
            "kernel_params": kernel_params,
            "acquisition": best_acq_name,
            "rmse_test": rmse_test,
            "corr_test": corr_test
        })

        # Write to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1, f"{global_frac:.2f}-{local_radius:.2f}", best_acq_name, next_point.tolist(), mu, sigma, best_value, y_next, rmse_test, corr_test
            ])

        print(f"Iter {i+1} | Phase {global_frac:.2f}-{local_radius:.2f} | y_pred={y_next:.4f}, RMSE_test={rmse_test:.4f}, Corr_test={corr_test:.3f}")

    # Save full history
    X_all = np.vstack([X_init, np.array(history["X"])])
    y_all = np.concatenate([np.array(y_init), np.array(history["y"])])
    df = pd.DataFrame(X_all, columns=[f"x{i}" for i in range(X_all.shape[1])])
    df["y"] = y_all
    df.to_csv(f"{export_prefix}_all.csv", index=False)
    np.savez(f"{export_prefix}.npz", X=X_all, y=y_all)

    best_idx = np.argmax(y_all)
    best_x = X_all[best_idx]
    best_y = y_all[best_idx]

    return best_x, best_y, history, best_results

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import csv
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def adaptive_bbo_with_gp_test_check(
    X_init, y_init, config,
    acquisition_list=["EI", "UCB", "PI", "THOMPSON"],
    kernel_list=["RBF", "Matern"],
    num_iterations=30,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=False,
    test_fraction=0.2,
    rmse_threshold=0.5,
    corr_threshold=0.3,
    export_prefix="bo_history",
    random_state=42
):
    """
    Adaptive Bayesian Optimization loop with:
      - Dynamic candidate generation (global/local)
      - Multiple kernels & acquisitions
      - GP quality check via train/test split
      - Logging and CSV export
      - Skips candidates if GP fails predictive quality
    """
    np.random.seed(random_state)

    # Scale inputs/outputs
    if scale_candidates:
        X_train, X_scaler = scale_data(X_init, scaler_type='minmax')
        y_train, y_scaler = scale_data(np.array(y_init).reshape(-1,1), scaler_type='standard')
    else:
        X_train, X_scaler = X_init.copy(), None
        y_train, y_scaler = np.array(y_init).reshape(-1,1), None

    input_dim = X_train.shape[1]
    history = {"X": X_train.copy(), "y": y_train.copy()}
    best_results = []
    

# Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



    # Prepare CSV
    csv_file = f"{export_prefix}_iter{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "acquisition", "kernel", "x_next", "y_pred", "rmse_test", "corr_test"])

    for i in range(num_iterations):
        # -----------------------------
        # 1. Train GP
        # -----------------------------
        kernel_name = np.random.choice(kernel_list)
        cfg_copy = config.copy()
        cfg_copy["kernel_type"] = kernel_name

        gp = build_dynamic_gp(X_train, y_train, config=cfg_copy, iteration=i,
                              total_iterations=num_iterations, seed=random_state)

        # -----------------------------
        # 2. Train/test split for GP check
        # -----------------------------
        if X_train.shape[0] > 5:  # enough points to split
            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=test_fraction, random_state=random_state)
            gp.fit(X_tr, y_tr)
            y_pred_test = gp.predict(X_te)
            rmse_test = np.sqrt(mean_squared_error(y_te, y_pred_test))
            corr_test = np.corrcoef(y_te.flatten(), y_pred_test.flatten())[0,1]
        else:
            rmse_test, corr_test = 0.0, 1.0  # skip check early

        if rmse_test > rmse_threshold or corr_test < corr_threshold:
            logger.warning(f"Iteration {i+1}: GP test failed (RMSE={rmse_test:.3f}, Corr={corr_test:.2f}), skipping iteration.")
            continue  # skip candidate selection this iteration

        # -----------------------------
        # 3. Candidate generation
        # -----------------------------
        # Determine global/local fraction dynamically (phases)
        if i < num_iterations // 3:
            global_frac = 0.8; local_radius = 0.2
        elif i < 2 * num_iterations // 3:
            global_frac = 0.4; local_radius = 0.1
        else:
            global_frac = 0.2; local_radius = 0.05

        n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)
        n_global = int(n_candidates * global_frac)
        n_local = n_candidates - n_global

        # Global candidates
        X_global = np.random.rand(n_global, input_dim)
        # Local candidates around current best
        idx_best = np.argmax(y_train)
        x_best = X_train[idx_best]
        X_local = x_best + (np.random.rand(n_local, input_dim) - 0.5) * 2 * local_radius
        X_local = np.clip(X_local, 0.0, 1.0)

        X_cand = np.vstack([X_global, X_local])
        if scale_candidates:
            X_cand_scaled = X_scaler.transform(X_cand)
        else:
            X_cand_scaled = X_cand.copy()

        # -----------------------------
        # 4. Acquisition function
        # -----------------------------
        mu, sigma = gp.predict(X_cand_scaled, return_std=True)
        y_max = np.max(y_train)
        acq_vals = select_acquisition(acquisition_list[0], mu=mu, sigma=sigma, iteration=i, y_max=y_max, model_type="GP")
        idx_next = np.argmax(acq_vals)

        next_x = X_cand[idx_next]
        y_next = gp.predict(next_x.reshape(1,-1)).item()

        # -----------------------------
        # 5. Update training data
        # -----------------------------
        X_train = np.vstack([X_train, next_x.reshape(1,-1)])
        y_train = np.append(y_train, y_next)
        history["X"] = np.vstack([history["X"], next_x.reshape(1,-1)])
        history["y"] = np.append(history["y"], y_next)

        # -----------------------------
        # 6. Log iteration
        # -----------------------------
        logger.info(f"Iter {i+1} | x_next={next_x} | y_pred={y_next:.6f} | rmse_test={rmse_test:.4f}, corr_test={corr_test:.2f}")

        # CSV write
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i+1, acquisition_list[0], kernel_name, next_x.tolist(), y_next, rmse_test, corr_test])

        # -----------------------------
        # 7. Track iteration best
        # -----------------------------
        best_idx_iter = np.argmax(y_train)
        best_results.append({
            "iteration": i+1,
            "best_input": X_train[best_idx_iter],
            "best_output": y_train[best_idx_iter],
            "kernel": kernel_name,
            "acquisition": acquisition_list[0],  # add this
            "rmse_test": rmse_test,
            "corr_test": corr_test
            })


    # -----------------------------
    # Export full history
    # -----------------------------
    df_all = pd.DataFrame(history["X"], columns=[f"x{i}" for i in range(input_dim)])
    df_all["y"] = history["y"]
    df_all.to_csv(f"{export_prefix}_all.csv", index=False)
    np.savez(f"{export_prefix}.npz", X=history["X"], y=history["y"])
    logger.info(f"Exported full BO history to {export_prefix}_all.csv and .npz")

    # Return overall best
    best_idx = np.argmax(history["y"])
    best_x = history["X"][best_idx]
    best_y = history["y"][best_idx]

    return best_x, best_y, history, best_results


def adaptive_bbo_with_logging(
    X_init,
    y_init,
    config,
    acquisition_list=["EI", "UCB", "PI", "THOMPSON"],
    kernel_list=["RBF", "Matern"],
    num_iterations=30,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=False,
    test_fraction=0.2,
    export_prefix="bo_history",
    random_state=42
):
    np.random.seed(random_state)

    # --- Scaling ---
    X_train_full, X_scaler = (
        (MinMaxScaler().fit_transform(X_init), MinMaxScaler().fit(X_init))
        if scale_candidates else (X_init.copy(), None)
    )
    y_train_full, y_scaler = (
        (StandardScaler().fit_transform(np.array(y_init).reshape(-1, 1)), StandardScaler().fit(np.array(y_init).reshape(-1, 1)))
        if scale_candidates else (np.array(y_init).reshape(-1, 1), None)
    )

    input_dim = X_init.shape[1]

    history = {"X": [], "y": [], "gp_failed": []}
    best_results = []


    # --- CSV logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    function_name = config.get("name", "unknown_func")

    # Prepare CSV
    csv_file = f"{export_prefix}_iter{timestamp}_func{function_name}.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "function_name","acquisition","kernel","kernel_param", "x_next", "y_pred", "gp_valid"])

    # --- Main BO Loop ---
    for i in range(num_iterations):

        # --- Split for GP validation ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=test_fraction, random_state=random_state
        )

        gp_valid = True
        # try:
        #     # Evaluate GP predictive accuracy on test set
        # gp = build_dynamic_gp(X_train, y_train, config=config, iteration=i, total_iterations=num_iterations, seed=random_state)
        # y_test_pred = gp.predict(X_test)
        # test_rmse = np.sqrt(np.mean((y_test_pred - y_test.flatten())**2))
        # # Compute adaptive threshold

        # y_range = y_train.max() - y_train.min()
        # dynamic_threshold = max(3.0, 0.25 * y_range)

        # gp_valid = (
        #     not np.isnan(test_rmse) and
        #     test_rmse < dynamic_threshold
        # )

        #     test_rmse = np.sqrt(np.mean((y_test_pred - y_test.flatten())**2))
        #     if np.isnan(test_rmse) or test_rmse > 1e6:  # arbitrary threshold
        #         gp_valid = False
        # except Exception as e:
        #     logger.warning(f"GP training failed at iteration {i}: {e}")
        #     gp_valid = False

        history["gp_failed"].append(not gp_valid)

        # --- Dynamic candidate generation ---
        n_candidates = base_candidates + candidate_scale * input_dim + int(i / num_iterations * candidate_scale * input_dim)
        global_frac = max(0.2, 0.8 - 0.6 * i / num_iterations)   # dynamic
        local_radius = max(0.05, 0.2 - 0.15 * i / num_iterations)  # dynamic
        n_global = int(n_candidates * global_frac)
        n_local = n_candidates - n_global

        # Global candidates
        X_global = np.random.rand(n_global, input_dim)

        # Local candidates around current best
        idx_best = np.argmax(y_train_full)
        x_best = X_train_full[idx_best]
        X_local = x_best + (np.random.rand(n_local, input_dim) - 0.5) * 2 * local_radius
        X_local = np.clip(X_local, 0.0, 1.0)

        X_candidates = np.vstack([X_global, X_local])
        X_candidates_scaled = X_scaler.transform(X_candidates) if scale_candidates else X_candidates.copy()

        best_per_acq = {}

        # --- Acquisition + Kernel loop ---
       
        # --- Log to CSV ---
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1,
                function_name,
                best_acq_name,
                best_kernel_name,
                kernel_params,
                next_point.tolist(),
                y_next,
                gp_valid
            ])

        best_results.append({
            "iteration": i+1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": kernel_used,
            "kernel_params": kernel_params,
            "acquisition": best_acq_name,
            "gp_valid": gp_valid
        })

    # --- Export full history ---
    X_all = np.vstack([X_init, np.array(history["X"])])
    y_all = np.concatenate([np.array(y_init), np.array(history["y"])])
    df = pd.DataFrame(X_all, columns=[f"x{i}" for i in range(X_all.shape[1])])
    df["y"] = y_all
    df.to_csv(f"{export_prefix}_all.csv", index=False)
    np.savez(f"{export_prefix}.npz", X=X_all, y=y_all)

    best_idx = np.argmax(y_all)
    best_x = X_all[best_idx]
    best_y = y_all[best_idx]

    return best_x, best_y, history, best_results

