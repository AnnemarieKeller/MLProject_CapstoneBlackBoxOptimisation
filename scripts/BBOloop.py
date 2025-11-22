# import accquistionscripts as acq
import scripts.functionConfig as funcConfig
from scripts.accquistions import *
from scripts.scaler import *

import numpy as np
# from gpBuilder import *
from .setup.gpBuilder import *
from .setBoundary import apply_boundary_penalty
import numpy as np
from .candidateGeneration import *
from scripts.utils.utils import *
from sklearn.ensemble import RandomForestRegressor





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

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import Parallel, delayed  # for parallel evaluations

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
    kernel_list=["RBF","Matern","RationalQuadratic"],
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
        # Dynamic candidate generation
        n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)
        X_candidates = generate_candidates(input_dim, n_candidates, determine_candidate_generation_method(input_dim))
        X_candidates_scaled = X_scaler.transform(X_candidates) if scale_candidates else X_candidates.copy()

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

