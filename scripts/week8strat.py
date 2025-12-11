import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scripts.utils.logging.logger import * 
from scripts.accquistions import * 
from scripts.candidateGeneration import * 
from scripts.setup.gpBuilder import * 
from scripts.utils.reports.reportbuilder import *
from scripts.utils.selfHealing_BO import *
from scripts.analysis.gphealth import * 


# ------------------------------
# Function-specific strategies
# ------------------------------
# original functions strategy map -run was not good health results so decided to change strategy
function_strategy_map = {
     1: {"type": "dense_exploit", "acq": ["EI", "UCB"]}, #changed to ucb 3.0
    # 1: {"type": "dense_exploit", "acq": [ "UCB"]},
     2: {"type": "global_explore", "acq": ["EI"]}, # ucb 1 -2 , 3 would be an overkill 
    # 2: {"type": "global_explore", "acq": ["UCB", "PORTFOLIO"]},
    3: {"type": "refinement", "acq": ["UCB"]}, # increase 3-4 in early stage for exploration and later on 1.5 -2
    4: {"type": "dense_exploit", "acq": ["EI", "UCB"]},# keep ubs 2-3 high early 
    5: {"type": "mixed", "acq": ["EI", "UCB"]},
    6: {"type": "local_exploit", "acq": ["PI", "EI"]},
    7: {"type": "explore_then_exploit", "acq": ["EI", "THOMPSON"]},
    8: {"type": "refinement", "acq": ["UCB", "EI"]}
}
# ------------------------------
# Function strategy mapping with recommended acquisition values
# type: general strategy type
# acq: list of dicts with acquisition function and recommended parameters
# function_strategy_map = {
#     1: {"type": "dense_exploit", 
#         "acq": [{"name": "UCB", "kappa": 3.0}]},  # focus on exploitation
#     2: {"type": "global_explore", 
#         "acq": [{"name": "UCB", "kappa": 1.5}, {"name": "PORTFOLIO", "weights": [0.5, 0.5]}]},  # moderate exploration
#     3: {"type": "refinement", 
#         "acq": [{"name": "UCB", "kappa": 3.0}]},  # early exploration high, reduce to 1.5-2 later
#     4: {"type": "dense_exploit", 
#         "acq": [{"name": "EI", "xi": 0.01}, {"name": "UCB", "kappa": 2.5}]},  # exploit with EI, exploration with UCB
#     5: {"type": "mixed", 
#         "acq": [{"name": "EI", "xi": 0.01}, {"name": "UCB", "kappa": 2.0}]},  # balanced strategy
#     6: {"type": "local_exploit", 
#         "acq": [{"name": "PI", "xi": 0.01}, {"name": "EI", "xi": 0.01}]},  # focus on improving local maxima
#     7: {"type": "explore_then_exploit", 
#         "acq": [{"name": "EI", "xi": 0.1}, {"name": "THOMPSON", "samples": 5}]},  # explore with EI, exploit with Thompson
#     8: {"type": "refinement", 
#         "acq": [{"name": "UCB", "kappa": 2.0}, {"name": "EI", "xi": 0.01}]}  # UCB for exploration, EI for local improvements
# }

#
# ------------------------------
# Main function-centric BO loop
# ------------------------------

def adaptive_bbo_weekly_strategy100(
    function_id, cfg, X_init, y_init, num_iterations=30,
    base_candidates=500, candidate_scale=200,
    scale_candidates=False, export_prefix="bo_weekly",
    random_state=42
):
    import numpy as np
    import os
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

    np.random.seed(random_state)

    # --------------------------------
    # Load function-specific strategy
    # --------------------------------
    strat = function_strategy_map[function_id]
    acq_list = strat["acq"]
    strategy_type = strat["type"]

    # ------------------------------------
    # Data scaling
    # ------------------------------------
    if scale_candidates:
        X_scaler = MinMaxScaler().fit(X_init)
        X_train_scaled = X_scaler.transform(X_init)
        y_scaler = StandardScaler().fit(np.array(y_init).reshape(-1, 1))
        y_train_scaled = y_scaler.transform(np.array(y_init).reshape(-1, 1))
    else:
        X_scaler = None
        y_scaler = None
        X_train_scaled = X_init.copy()
        y_train_scaled = np.array(y_init).reshape(-1, 1)

    # ------------------------------------
    # Logging containers
    # ------------------------------------
    history_X, history_y = [], []
    best_results = []
    gp_health_history = []
    iteration_acq_history = []
    history_sigma = []  # add this with history_X and history_y




    # ------------------------------------
    # Create folder under reports using export_prefix
    # ------------------------------------
    log_dir = os.path.join("reports", export_prefix)
    os.makedirs(log_dir, exist_ok=True)

    start_log(cfg, export_prefix, log_dir)
    input_dim = X_init.shape[1]

    # ==========================================================
    #                  Main Loop
    # ==========================================================
    for i in range(num_iterations):
        # --- Kernel search + GP health + anomaly detection ---
        candidate_kernels = ["RBF", "Matern"]
        best_gp, best_gp_health, best_kernel_name = None, -np.inf, None
        cond, avg_sigma, anomalies = None, None, None

        for kname in candidate_kernels:
            cfg_k = cfg.copy()
            cfg_k["kernel_type"] = kname
            if kname == "Matern":
                cfg_k["nu"] = np.random.choice([1.5, 2.5])

            gp_tmp = build_dynamic_gp(
                X_train_scaled, y_train_scaled, cfg_k,
                iteration=i, total_iterations=num_iterations,
                seed=random_state
            )

            # Compute health & anomalies
            gp_health, cond, avg_sigma, loglike = get_gp_health(gp_tmp, X_train_scaled, y_train_scaled)
            anom = detect_gp_anomalies(gp_tmp, X_train_scaled, y_train_scaled)

            if gp_health > best_gp_health:
                best_gp_health = gp_health
                best_gp = gp_tmp
                best_kernel_name = kname
                cond, avg_sigma, anomalies = cond, avg_sigma, anom

        # Assign best GP
        gp = best_gp
        gp_health_history.append(best_gp_health)


        # Log health and anomalies
        # anomalies_detected = log_gp_health_iteration_updated(
        #     i, best_gp_health, cond, avg_sigma, loglike, gp, X_train_scaled, y_train_scaled
        # )
        log_gp_health_iteration_updated(
            i, best_gp_health, cond, avg_sigma, loglike, gp, X_train_scaled, y_train_scaled
        )

        # # Optional self-repair
        # if anomalies_detected:
        #     cfg = bo_repair_action(anomalies_detected, cfg)
        #     gp = build_dynamic_gp(cfg)
        #     gp.fit(X_train_scaled, y_train_scaled)

        # Kernel repair if GP health is weak
        if best_gp_health < 0.6:
            kernel_pool = [
                RBF(1.0), Matern(1.0, nu=1.5), C(1.0)*RBF(1.0)+WhiteKernel(1e-3),
                RBF(1.0)+Matern(1.0), RBF(1.0)*Matern(1.0)
            ]
            best_cgp, best_cgp_health, best_ckernel = None, -np.inf, None
            for kern in kernel_pool:
                gp_tmp = build_dynamic_gp(
                    X_train_scaled, y_train_scaled, cfg,
                    iteration=i, total_iterations=num_iterations,
                    seed=random_state, kernel_override=kern
                )
                gph, cd, a_sigma, anom = get_gp_health_score(gp_tmp, X_train_scaled, y_train_scaled)
                if gph > best_cgp_health:
                    best_cgp_health = gph
                    best_cgp = gp_tmp
                    best_ckernel = kern
                    cond, avg_sigma, anomalies = cd, a_sigma, anom

            gp, best_gp_health, best_kernel_name = best_cgp, best_cgp_health, str(best_ckernel)
            log_gp_health_iteration(
                i, best_gp_health, cond=cond, avg_sigma=avg_sigma,
                loglike=getattr(gp, "log_marginal_likelihood_value_", 0),
                anomalies=anomalies
            )

        # Candidate generation
        n_candidates = base_candidates + input_dim * candidate_scale
        X_candidates_scaled = generate_candidates_by_strategy1(strategy_type, gp, X_train_scaled, i, num_iterations, n_candidates)

        # Acquisition evaluation
        main_acq = acq_list[0].upper()
        best_val, next_scaled, best_acq_params = -np.inf, None, {}
        mu_all, sigma_all = gp.predict(X_candidates_scaled, return_std=True)
        y_max = np.max(y_train_scaled)
        # history_sigma.append(sigma_all.copy())
        iteration_sigma = []

        for idx, (mu, sigma) in enumerate(zip(mu_all, sigma_all)):
            mu, sigma = mu.item(), sigma.item()
            iteration_sigma.append(sigma)
            val, params = evaluate_acquisition(main_acq, mu, sigma, y_max, i, num_iterations)
            val *= best_gp_health
            if val > best_val:
                best_val = val
                next_scaled = X_candidates_scaled[idx]
                best_acq_params = params

        # Log acquisition values for all methods
        history_sigma.append(np.array(iteration_sigma))
        # --- Acquisition evaluation for all methods ---
        acq_values = log_all_acquisitions(mu_all, sigma_all, y_max, best_gp_health, i)

# Determine best acquisition across all candidates and methods
        best_acq_name = max(acq_values, key=lambda k: np.max(acq_values[k]))
        best_idx = np.argmax(acq_values[best_acq_name])
        best_acq_params = X_candidates_scaled[best_idx]

# Append iteration info
        anomalies_str = [a["type"] if isinstance(a, dict) and "type" in a else str(a) for a in anomalies or []]
        iteration_acq_history.append({
                 "iteration": i + 1,
                 "X_candidates": X_candidates_scaled,
                    "acq_values": acq_values,
                    "best_acq_name": best_acq_name,
                     "best_acq_params": best_acq_params,
                    "kernel": best_kernel_name,
                     "kernel_params": gp.kernel_.get_params(),
                     "anomalies": anomalies_str
                            })


        # Predict output + update training
        y_next_scaled = gp.predict(next_scaled.reshape(1, -1)).item()
        if scale_candidates:
            next_point = X_scaler.inverse_transform(next_scaled.reshape(1, -1)).flatten()
            y_next = y_scaler.inverse_transform([[y_next_scaled]]).item()
        else:
            next_point, y_next = next_scaled, y_next_scaled

        X_train_scaled = np.vstack([X_train_scaled, next_scaled.reshape(1, -1)])
        y_train_scaled = np.vstack([y_train_scaled, [[y_next_scaled]]])
        history_X.append(next_point)
        history_y.append(y_next)

        middle_log_ano(
            i, cfg["name"], main_acq,
            gp, next_point, y_next, best_gp_health,
            anomalies=anomalies,
            acq_params={"acq": main_acq}
        )

        best_results.append({
            "iteration": i + 1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": best_kernel_name,
            "acquisition": main_acq,
            "gp_health": best_gp_health,
            "iteration_acq": iteration_acq_history[-1]
        })

    # Finalization
    X_all = np.vstack([X_init, np.array(history_X)])
    y_all = np.concatenate([np.array(y_init), np.array(history_y)])
    save_log(X_all, y_all, export_prefix=f"{export_prefix}_func{function_id}")
    save_gp_health_plot(gp_health_history, export_prefix)
    generate_pro_report( function_id,
        gp_health_history,
        iteration_acq_history,
        history_X,
        history_y,
        best_results,
        save_dir=log_dir,
        history_sigma=history_sigma
        )
    generate_pro_report1( function_id,
        gp_health_history,
        iteration_acq_history,
        history_X,
        history_y,
        best_results,
        save_dir=log_dir,
        history_sigma=history_sigma
        )

    # create_function_report(
    #     function_id,
    #     gp_health_history,
    #     iteration_acq_history,
    #     history_X,
    #     history_y,
    #     best_results,
    #     save_dir=log_dir,
    #     external_ref=export_prefix  # same as export_prefix
    # )

    best_idx = np.argmax(history_y)
    return history_X[best_idx], history_y[best_idx], {"X": history_X, "y": history_y}, best_results

