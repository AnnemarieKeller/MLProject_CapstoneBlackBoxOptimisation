# import accquistionscripts as acq
import scripts.configs.functionConfig as funcConfig
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
def adaptive_bbo_dynamic_full2(
    X_init, y_init, config,
    acquisition_list=["EI", "UCB", "PI", "THOMPSON"],
    kernel_list=["RBF", "Matern", "RationalQuadratic"],
    num_iterations=30,
    random_state=42,
    base_candidates=500,
    candidate_scale=200,
    scale_candidates=True
):
    """
    Fully adaptive Bayesian Optimization loop:
    - Multiple kernels
    - Kernel hyperparameter tuning
    - Multiple acquisition functions
    - Dynamic candidate generation
    - Tracks history and best_results
    - Handles hihger kappa for noisy functions 
    """

    import numpy as np
    np.random.seed(random_state)

    # ----------------------------
    # Scale inputs/outputs
    # ----------------------------
    X_train, X_scaler = scale_data(X_init, scaler_type='minmax') if scale_candidates else (X_init.copy(), None)
    y_train, y_scaler = scale_data(np.array(y_init).reshape(-1, 1), scaler_type='standard') if scale_candidates else (np.array(y_init).reshape(-1, 1), None)
    input_dim = X_init.shape[1]

    history = {"X": X_train.copy(), "y": y_train.copy()}
    best_results = []

    for i in range(num_iterations):
        # ----------------------------
        # Dynamic candidate generation
        # ----------------------------
        n_candidates = base_candidates + candidate_scale * input_dim + int(i / num_iterations * candidate_scale * input_dim)
        X_candidates = generate_candidates(input_dim, n_candidates, determine_candidate_generation_method(input_dim))
        X_candidates_scaled = X_scaler.transform(X_candidates) if scale_candidates else X_candidates.copy()

        best_per_acq = {}

        # ----------------------------
        # Loop over acquisition functions
        # ----------------------------
        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point = None
            best_kernel_name = None
            best_params = None

            # ----------------------------
            # Loop over kernels
            # ----------------------------
            for kernel_name in kernel_list:
                cfg_copy = config.copy()
                cfg_copy["kernel_type"] = kernel_name

                # Dynamic kernel hyperparameters
                if kernel_name == "Matern":
                    cfg_copy["nu"] = np.random.choice([1.5, 2.5])
                elif kernel_name == "RationalQuadratic":
                    cfg_copy["alpha_rq"] = np.random.uniform(0.5, 2.0)
                cfg_copy["C"] = np.random.uniform(0.1, 3.0)

                # Build GP using patched dynamic kernel
                kernel = build_dynamic_kernel(X_train, y_train, config=cfg_copy, iteration=i, total_iterations=num_iterations)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=3, random_state=random_state)
                gp.fit(X_train, y_train)

                # ----------------------------
                # Evaluate candidates
                # ----------------------------
                for candidate in X_candidates_scaled:
                    mu, sigma = gp.predict(candidate.reshape(1, -1), return_std=True)
                    mu, sigma = mu.item(), sigma.item()
                    y_max = np.max(y_train)

                    # Acquisition function
                    if acq_name.upper() == "UCB":
                        # Determine noise level from y_train
                        y_std = np.std(y_train)
                        noise_factor = max(1.0, y_std / 2)  # scale exploration with noise


                        kappa_base = 3.0 * np.exp(-i / num_iterations) + 0.1

                        kappa = kappa_base * noise_factor

                        
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

        # ----------------------------
        # Select best acquisition
        # ----------------------------
        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point_scaled, _, kernel_used, kernel_params = best_per_acq[best_acq_name]
        next_point = X_scaler.inverse_transform(next_point_scaled.reshape(1, -1)).flatten() if scale_candidates else next_point_scaled.copy()

        # Fit GP with selected kernel
        kernel = build_dynamic_kernel(X_train, y_train, config=kernel_params, iteration=i, total_iterations=num_iterations)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False, n_restarts_optimizer=3, random_state=random_state)
        gp.fit(X_train, y_train)

        y_next_scaled = gp.predict(next_point_scaled.reshape(1, -1)).item()
        y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item() if scale_candidates else y_next_scaled

        # Update training set
        X_train = np.vstack([X_train, next_point_scaled])
        y_train = np.append(y_train, y_next_scaled)
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        # Record iteration best
        best_idx_iter = np.argmax(y_train)
        best_input_iter = X_scaler.inverse_transform(X_train[best_idx_iter].reshape(1, -1)).flatten() if scale_candidates else X_train[best_idx_iter]
        best_output_iter = y_scaler.inverse_transform(np.array([[y_train[best_idx_iter]]])).item() if scale_candidates else y_train[best_idx_iter]

        best_results.append({
            "iteration": i + 1,
            "best_input": best_input_iter,
            "best_output": best_output_iter,
            "kernel": kernel_used,
            "kernel_params": kernel_params,
            "acquisition": best_acq_name,
            "n_candidates": n_candidates
        })

        print(f"Iter {i+1} | Acq: {best_acq_name} | Kernel: {kernel_used} | Next input: {next_point} | Predicted y: {best_output_iter:.6f} | Candidates: {n_candidates}")

    # ----------------------------
    # Final best
    # ----------------------------
    best_idx = np.argmax(y_train)
    best_input = X_scaler.inverse_transform(X_train[best_idx].reshape(1, -1)).flatten() if scale_candidates else X_train[best_idx]
    best_output = y_scaler.inverse_transform(np.array([[y_train[best_idx]]])).item() if scale_candidates else y_train[best_idx]

    return best_input, best_output, history, best_results
def adaptive_bbo_dynamic_fullNoisenothandled(
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
    Fully adaptive Bayesian Optimization with:
    - Multiple kernels
    - Multiple acquisition functions
    - Dynamic noise scaling
    - Candidate scaling
    - Tracks best results per iteration
    """

    np.random.seed(random_state)

    # Scaling
    X_train, X_scaler = scale_data(X_init, 'minmax') if scale_candidates else (X_init.copy(), None)
    y_train, y_scaler = scale_data(np.array(y_init).reshape(-1,1), 'standard') if scale_candidates else (np.array(y_init).reshape(-1,1), None)
    input_dim = X_init.shape[1]

    history = {"X": X_train.copy(), "y": y_train.copy()}
    best_results = []

    for i in range(num_iterations):
        # Candidate generation
        n_candidates = base_candidates + candidate_scale * input_dim + int(i/num_iterations * candidate_scale * input_dim)
        X_candidates = generate_candidates(input_dim, n_candidates, determine_candidate_generation_method(input_dim))
        X_candidates_scaled = X_scaler.transform(X_candidates) if scale_candidates else X_candidates.copy()

        best_per_acq = {}

        # Loop acquisition functions
        for acq_name in acquisition_list:
            best_value = -np.inf
            best_point, best_kernel_name, best_params = None, None, None

            # Loop kernels
            for kernel_name in kernel_list:
                cfg_copy = config.copy()
                cfg_copy["kernel_type"] = kernel_name

                # Kernel hyperparameters
                if kernel_name == "Matern":
                    cfg_copy["nu"] = np.random.choice([1.5, 2.5])
                elif kernel_name == "RationalQuadratic":
                    cfg_copy["alpha_rq"] = np.random.uniform(0.5, 2.0)
                cfg_copy["C"] = np.random.uniform(0.1, 3.0)

                # Build GP with dynamic kernel
                kernel = build_dynamic_kernel(X_train, y_train, cfg_copy, iteration=i, total_iterations=num_iterations)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=10, normalize_y=True, random_state=random_state)
                gp.fit(X_train, y_train)

                # Evaluate candidates
                for candidate in X_candidates_scaled:
                    mu, sigma = gp.predict(candidate.reshape(1,-1), return_std=True)
                    mu, sigma = mu.item(), sigma.item()
                    y_max = np.max(y_train)

                    # Acquisition
                    if acq_name.upper() == "UCB":
                        kappa = 3.0 * np.exp(-i / num_iterations) + 0.1  # dynamic kappa
                        acq_value = mu + kappa * sigma
                    elif acq_name.upper() == "EI":
                        xi = 0.1 * np.exp(-i/num_iterations) + 0.01
                        Z = (mu - y_max - xi) / max(sigma, 1e-8)
                        from scipy.stats import norm
                        acq_value = (mu - y_max - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
                    elif acq_name.upper() == "PI":
                        eta = 0.1 * np.exp(-i/num_iterations) + 0.01
                        Z = (mu - y_max - eta) / max(sigma, 1e-8)
                        from scipy.stats import norm
                        acq_value = norm.cdf(Z)
                    elif acq_name.upper() == "THOMPSON":
                        acq_value = np.random.normal(mu, sigma)
                    else:
                        raise ValueError(f"Unknown acquisition function: {acq_name}")

                    # Track best
                    if acq_value > best_value:
                        best_value = acq_value
                        best_point = candidate
                        best_kernel_name = kernel_name
                        best_params = cfg_copy.copy()

            best_per_acq[acq_name] = (best_point, best_value, best_kernel_name, best_params)

        # Choose best acquisition
        best_acq_name = max(best_per_acq, key=lambda k: best_per_acq[k][1])
        next_point_scaled, _, kernel_used, kernel_params = best_per_acq[best_acq_name]
        next_point = X_scaler.inverse_transform(next_point_scaled.reshape(1,-1)).flatten() if scale_candidates else next_point_scaled.copy()

        # Fit GP and get predicted y
        kernel = build_dynamic_kernel(X_train, y_train, kernel_params, iteration=i, total_iterations=num_iterations)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=10, normalize_y=True, random_state=random_state)
        gp.fit(X_train, y_train)
        y_next_scaled = gp.predict(next_point_scaled.reshape(1,-1)).item()
        y_next = y_scaler.inverse_transform(np.array([[y_next_scaled]])).item() if scale_candidates else y_next_scaled

        # Update training data
        X_train = np.vstack([X_train, next_point_scaled])
        y_train = np.append(y_train, y_next_scaled)
        history["X"] = np.vstack([history["X"], next_point])
        history["y"] = np.append(history["y"], y_next)

        # Record best so far
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
