from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scripts.utils.logging.logger import * 
from scripts.utils.selfHealing_BO import * 
from scripts.accquistions import * 
from scripts.candidateGeneration import * 
from scripts.setup.gpBuilder import * 
from scripts.utils.reports.reportbuilder import *
from scripts.utils.selfHealing_BO import *
from scripts.analysis.gphealth import *
from scripts.configs.configs import * 
from scripts.analysis.gphealth import * 

import numpy as np
def adaptive_bbo_weekly_strategy_updated(
    function_id, cfg, X_init, y_init, num_iterations=30,
    base_candidates=500, candidate_scale=200,
    scale_candidates=False, export_prefix="bo_updated_strat",
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
    history_sigma = []

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

            gp_health, cond, avg_sigma, loglike = get_gp_health(gp_tmp, X_train_scaled, y_train_scaled)
            anom = detect_gp_anomalies(gp_tmp, X_train_scaled, y_train_scaled)

            if gp_health > best_gp_health:
                best_gp_health = gp_health
                best_gp = gp_tmp
                best_kernel_name = kname
                cond, avg_sigma, anomalies = cond, avg_sigma, anom

        gp = best_gp
        gp_health_history.append(best_gp_health)

        # Log GP health
        log_gp_health_iteration_updated(
            i, best_gp_health, cond, avg_sigma, loglike, gp, X_train_scaled, y_train_scaled
        )

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

        # -------------------------------
        # Candidate generation (FULLY INTEGRATED)
        # -------------------------------
        n_candidates = base_candidates + input_dim * candidate_scale
        X_candidates_scaled = generate_candidates_by_strategy(
           function_id,
            strategy_type, gp, X_train_scaled, i, num_iterations
        )

        # Acquisition evaluation
  
        main_acq = acq_list[0]["name"].upper()

        best_val, next_scaled, best_acq_params = -np.inf, None, {}
        mu_all, sigma_all = gp.predict(X_candidates_scaled, return_std=True)
        y_max = np.max(y_train_scaled)
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

        history_sigma.append(np.array(iteration_sigma))
        acq_values = log_all_acquisitions(mu_all, sigma_all, y_max, best_gp_health, i)

        best_acq_name = max(acq_values, key=lambda k: np.max(acq_values[k]))
        best_idx = np.argmax(acq_values[best_acq_name])
        best_acq_params = X_candidates_scaled[best_idx]

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

    # -------------------------------
    # Finalization / Report Generation
    # -------------------------------
    X_all = np.vstack([X_init, np.array(history_X)])
    y_all = np.concatenate([np.array(y_init), np.array(history_y)])
    save_log(X_all, y_all, export_prefix=f"{export_prefix}_func{function_id}")
    save_gp_health_plot(gp_health_history, export_prefix)

    generate_pro_report(
        function_id,
        gp_health_history,
        iteration_acq_history,
        history_X,
        history_y,
        best_results,
        save_dir=log_dir,
        history_sigma=history_sigma
    )
    # generate_pro_report1(
    #     function_id,
    #     gp_health_history,
    #     iteration_acq_history,
    #     history_X,
    #     history_y,
    #     best_results,
    #     save_dir=log_dir,
    #     history_sigma=history_sigma
    # )

    best_idx = np.argmax(history_y)
    return history_X[best_idx], history_y[best_idx], {"X": history_X, "y": history_y}, best_results
