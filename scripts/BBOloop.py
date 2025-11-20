# import accquistionscripts as acq
import functionConfig as funcConfig
from accquistions import select_acquisition


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

   
    if model_type.upper() == "GP":
       surrogate = build_gpWhiteKernel(X_train, y_train,config_override)
    elif model_type.upper() == "SVR":
       surrogate = build_svr(X_train, y_train, function_config,config_override)
    else:
       raise ValueError(f"Unknown surrogate type: {model_type}")

    for i in range(num_iterations):

    # --- Build GP and optimize kernel dynamically ---
        if model_type.upper() == "GP":
            surrogate = build_gpWhiteKernel(
            X_train=X_train,
            y_train=y_train
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
