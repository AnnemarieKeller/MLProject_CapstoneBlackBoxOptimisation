
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import pdist
from scripts.utils.reports.reportbuilder import generate_pro_report
from scripts.analysis.gphealth import gp_health_score
from scripts.utils.logging.logger import * 
from scripts.configs.functionConfig import *
from scripts.exploration.candidateGeneration import * 
from scripts.exploration.accquistions import * 

def build_hybridRBF_kernel_safe(X_train, nu_list=[1.5, 2.5], noise_level=1e-1):
    """Build additive and multiplicative RBF+Matern hybrid kernels safely for multi-dim input."""
    dim = X_train.shape[1]
    n_samples = X_train.shape[0]

    # estimate initial length scale from average pairwise distance
    if n_samples > 2:
        avg_dist = np.mean(pdist(X_train))
    else:
        avg_dist = 1.0

    length_init = np.ones(dim) * avg_dist
    # bounds per dim
    lb = np.full(dim, avg_dist / 100)
    # ub = np.full(dim, avg_dist * 300)
    ub = np.full(dim, avg_dist * 1000)
    length_init = np.ones(dim) * avg_dist


# stack lower/upper bounds to shape (dim, 2)
    length_bounds = np.stack([lb, ub], axis=1)


    hybrid_kernels = []
    for nu in nu_list:
        # additive hybrid
        # k_add = C(1.0, (1e-3, 1e3)) * (
        k_add = C(1.0, (1e-5, 1e8)) * (
            RBF(length_scale=length_init, length_scale_bounds=length_bounds) +
            Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-4, 1e1))
        hybrid_kernels.append(k_add)

        # multiplicative hybrid
    

        k_mul = C(1.0, (1e-3, 1e3)) * (
            RBF(length_scale=length_init, length_scale_bounds=length_bounds) *
            Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-4, 1e1))
        hybrid_kernels.append(k_mul)

    return hybrid_kernels


def adaptive_bbo_multi_peak(X_init, y_init,cfg, num_iterations=30,
                            base_candidates=500, candidate_scale=200,
                            scale_candidates=False, random_state=42,
                            export_prefix="bo_multi_peak"):
    np.random.seed(random_state)
    input_dim = X_init.shape[1]
    log_dir = os.path.join("reports", export_prefix)
    os.makedirs(log_dir, exist_ok=True)
    # cfg = FUNCTION_CONFIG[4]

    start_log(cfg,export_prefix=export_prefix)



    # --------------------------
    # Scale data
    # --------------------------
    if scale_candidates:
        X_scaler = MinMaxScaler().fit(X_init)
        X_train_scaled = X_scaler.transform(X_init)
        y_scaler = StandardScaler().fit(y_init.reshape(-1,1))
        y_train_scaled = y_scaler.transform(y_init.reshape(-1,1))
    else:
        X_train_scaled = X_init.copy()
        y_train_scaled = y_init.reshape(-1,1)
        X_scaler = y_scaler = None

    # --------------------------
    # Logging
    # --------------------------
    history_X, history_y = [], []
    best_results = []
    gp_health_history = []
    iteration_acq_history = []
    history_sigma = []

    # --------------------------
    # Main loop
    # --------------------------
    for i in range(num_iterations):
        # --- Hybrid kernel candidates ---
        hybrid_kernels = build_hybridRBF_kernel_safe(X_train_scaled, nu_list=[0.5,1, 1.5])

        best_gp = None
        best_gp_health = -np.inf
        best_kernel_name = None
        anomalies = None

        # --- Evaluate all kernels ---
        for idx, kern in enumerate(hybrid_kernels):
            gp_tmp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=3, random_state=random_state)
            gp_tmp.fit(X_train_scaled, y_train_scaled.ravel())
            gph_dict = gp_health_score(gp_tmp, X_train_scaled, y_train_scaled.ravel())
            gph = gph_dict["score"]

            if gph > best_gp_health:
                best_gp_health = gph
                best_gp = gp_tmp
                best_kernel_name = f"hybrid_{idx}"
                anomalies = []

        gp = best_gp
        gp_health_history.append(best_gp_health)
        

        # --- Generate multi-peak candidates ---
        n_candidates = (base_candidates + input_dim * candidate_scale)*2
        # X_candidates_scaled = generate_multi_peak_candidates(
        #     gp, X_train_scaled, n_candidates=n_candidates,
        #     top_k_peaks=3, 
        #     local_scale=0.3 ,
        #     input_dim=input_dim,
        #     random_state=random_state

        # )
        X_candidates_scaled = generate_multi_peak_candidates(
            gp, X_train_scaled, n_candidates=n_candidates,
            top_k_peaks=5, 
            local_scale=0.15 ,
            input_dim=input_dim,
            random_state=random_state

        )
        mask = np.array([
        np.any(np.all(np.isclose(candidate, X_train_scaled), axis=1))
            for candidate in X_candidates_scaled
        ])
        X_candidates_scaled = X_candidates_scaled[~mask]

        # --- Acquisition: UCB with decaying beta ---
        mu_all, sigma_all = gp.predict(X_candidates_scaled, return_std=True)
        y_max = np.max(y_train_scaled)
        best_val, next_scaled, best_acq_params = -np.inf, None, {}
        iteration_sigma = []

        # beta_0 = 7
        # beta_min = 5

        beta_0 = 3
        beta_min = 1.5
        decay_rate = 0.0001
        beta = beta_min + (beta_0 - beta_min) * np.exp(-decay_rate * i) 
        # beta = beta_0 - (beta_0 - beta_min) * ((i / num_iterations)*2)
        beta = max(beta * best_gp_health, beta_min)
        best_acq_params = {"UCB_beta": None}  # default

        for idx, (mu, sigma) in enumerate(zip(mu_all, sigma_all)):
            mu, sigma = mu.item(), sigma.item()
            iteration_sigma.append(sigma)
            # val = mu + beta * sigma
            val = acquisition_ei(mu, sigma, y_max)

            # val *= best_gp_health
            if val > best_val:
                best_val = val
                next_scaled = X_candidates_scaled[idx]
                best_acq_params = {"EI": beta}

        history_sigma.append(np.array(iteration_sigma))

        # --- Update training set ---
        y_next_scaled = gp.predict(next_scaled.reshape(1,-1)).item()
        if scale_candidates:
            next_point = X_scaler.inverse_transform(next_scaled.reshape(1,-1)).flatten()
            y_next = y_scaler.inverse_transform([[y_next_scaled]]).item()
        else:
            next_point, y_next = next_scaled, y_next_scaled

        X_train_scaled = np.vstack([X_train_scaled, next_scaled.reshape(1,-1)])
        y_train_scaled = np.vstack([y_train_scaled, [[y_next_scaled]]])
        history_X.append(next_point)
        history_y.append(y_next)
        acq_vals = []

        for mu, sigma in zip(mu_all, sigma_all):
            mu, sigma = mu.item(), sigma.item()
            val = (mu + beta * sigma) * best_gp_health
            acq_vals.append(val)


        iteration_acq_history.append({
            "iteration": i+1,
            "X_candidates": X_candidates_scaled,
            "mu": mu_all,
            "sigma": sigma_all,
            "acq_values": {"UCB": acq_vals},
            "kernel": best_kernel_name,
            "gp_health": best_gp_health,
            "anomalies": anomalies
        })
        acq_vals = mu_all + beta * sigma_all



        best_results.append({
            "iteration": i+1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": best_kernel_name,
            "gp_health": best_gp_health,
            "iteration_acq": iteration_acq_history[-1],
            "acquisition": best_acq_params 
          
        })

    # --- Generate report ---
    log_dir = os.path.join("reports", export_prefix)
    os.makedirs(log_dir, exist_ok=True)
    generate_pro_report(
        function_id=5,
        gp_health_history=gp_health_history,
        iteration_acq_history=iteration_acq_history,
        history_X=history_X,
        history_y=history_y,
        best_results=best_results,
        save_dir=log_dir,
        history_sigma=history_sigma
    )

    # Return best point
    best_idx = np.argmax(history_y)
    return history_X[best_idx], history_y[best_idx], {"X": history_X, "y": history_y}, best_results

