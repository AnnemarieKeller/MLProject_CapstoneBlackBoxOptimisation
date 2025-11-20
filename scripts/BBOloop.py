# import accquistionscripts as acq
import scripts.functionConfig as funcConfig
from scripts.accquistions import *
from scripts.scaler import *

import numpy as np
# from gpBuilder import *
from .setup.gpBuilder import *
from .setBoundary import apply_boundary_penalty
import numpy as np


def generate_candidates(input_dim, n_candidates=500, method="random"):
 
    if method == "random":
        return np.random.rand(n_candidates, input_dim)
    elif method == "grid":
        # only works for small dim <= 3
        lin = [np.linspace(0,1,int(np.ceil(n_candidates**(1/input_dim)))) for _ in range(input_dim)]
        mesh = np.meshgrid(*lin)
        X = np.column_stack([m.ravel() for m in mesh])
        return X
    elif method == "sobol":
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=input_dim, scramble=True)
            X = sampler.random_base2(m=int(np.log2(n_candidates)))
            return X
        except ImportError:
            print("Sobol requires scipy >= 1.7. Falling back to random.")
            return np.random.rand(n_candidates, input_dim)
    else:
        raise ValueError(f"Unknown candidate generation method: {method}")

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

        X_candidates = generate_candidates(dim, n_candidates, method=candidate_method)

       
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
        X_candidates = generate_candidates(dim, n_candidates, method=candidate_method)
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
        X_candidates = np.random.rand(n_candidates, dim)
        
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
        X_candidates = np.random.rand(n_candidates, input_dim)

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
        X_candidates = np.random.rand(n_candidates, input_dim)

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
                    acq_value = acquisition_ucb(mu, sigma, iteration=i, kappa=kappa)
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

            X_train, X_scaler = scale_data(X_train, scaler = X_scaler)
            y_train, y_scaler = scale_data(y_train, scaler = y_scaler)
        

            best_per_acq[acq_name] = (best_point, best_value)

        # Select the best across all acquisition functions
        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point = best_per_acq[best_acq_name][0]

        # Predict output for history
        y_next = gp.predict(next_point.reshape(1, -1)).item()

        # Update training data
        X_train = np.vstack([X_train, next_point])
        y_train = np.append(y_train, y_next)

        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

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
        X_candidates = np.random.rand(n_candidates, dim)

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

