from datetime import datetime
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scripts.utils.logging.logger import *
from scripts.utils.selfHealing_BO import *
from scripts.exploration.accquistions import *
from scripts.exploration.candidateGeneration import *
from scripts.setup.gpBuilder import *
from scripts.utils.reports.reportbuilder import *
from scripts.analysis.gphealth import *
from scripts.configs.configs import *

def pick_top_2_ucb_candidates(gp, bounds, n_samples=1000, beta=5.0, random_state=42):
    """
    Pick top 2 points in 2D input space maximizing UCB acquisition function.
    """
    np.random.seed(random_state)
    X_candidates = np.column_stack([
        np.random.uniform(low, high, n_samples) for (low, high) in bounds
    ])
    mu, sigma = gp.predict(X_candidates, return_std=True)
    ucb = mu + beta * sigma
    top_indices = np.argsort(ucb)[-2:]
    return X_candidates[top_indices]

def week9_funct1(
    function_id, cfg, X_init, y_init, num_iterations=30,
    scale_candidates=False, export_prefix="function1_week9",
    random_state=42
):
    np.random.seed(random_state)

    # -----------------------
    # Scaling
    # -----------------------
    if scale_candidates:
        X_scaler = MinMaxScaler().fit(X_init)
        X_train_scaled = X_scaler.transform(X_init)
        y_scaler = StandardScaler().fit(np.array(y_init).reshape(-1, 1))
        y_train_scaled = y_scaler.transform(np.array(y_init).reshape(-1, 1))
    else:
        X_scaler, y_scaler = None, None
        X_train_scaled = X_init.copy()
        y_train_scaled = np.array(y_init).reshape(-1, 1)

    # -----------------------
    # Logging containers
    # -----------------------
    history_X, history_y = [], []
    best_results = []
    gp_health_history = []
    iteration_acq_history = []
    history_sigma = []

    # -----------------------
    # Reports folder
    # -----------------------
    log_dir = os.path.join("reports", export_prefix)
    os.makedirs(log_dir, exist_ok=True)
    start_log(cfg, export_prefix, log_dir)

    input_dim = X_init.shape[1]
    bounds = [(0, 1)] * input_dim  # 2D normalized

    # -----------------------
    # Main BO loop
    # -----------------------
    for i in range(num_iterations):
        # -----------------------
        # GP training + health check
        # -----------------------
        candidate_kernels = ["Matern"]
        best_gp, best_gp_health, best_kernel_name = None, -np.inf, None
        cond, avg_sigma, anomalies = None, None, None

        for kname in candidate_kernels:
            cfg_k = cfg.copy()
            cfg_k["kernel_type"] = kname
            if kname == "Matern":
                cfg_k["nu"] = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5])

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
        log_gp_health_iteration_updated(i, best_gp_health, cond, avg_sigma, loglike, gp, X_train_scaled, y_train_scaled)

        # Kernel repair if GP is weak
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

        # -----------------------
        # Pick next point(s) via UCB
        # -----------------------
        top_2 = pick_top_2_ucb_candidates(gp, bounds, n_samples=2000, beta=5.0)
        print(f"Iteration {i+1} - Top 2 UCB candidates:\n", top_2)

        next_scaled = top_2[0].reshape(1, -1)  # pick the first candidate
        y_next_scaled = gp.predict(next_scaled).item()

        # Scale back if needed
        if scale_candidates:
            next_point = X_scaler.inverse_transform(next_scaled).flatten()
            y_next = y_scaler.inverse_transform([[y_next_scaled]]).item()
        else:
            next_point, y_next = next_scaled.flatten(), y_next_scaled
        # After picking top_2:
        mu_all, sigma_all = gp.predict(top_2, return_std=True)
        y_max = np.max(y_train_scaled)
        beta = 5
        acq_values = mu_all + beta * sigma_all  # beta=5

        iteration_acq_history.append({
    "iteration": i + 1,
    "X_candidates": top_2,
    "acq_values": {"UCB": beta},
    "best_acq_name": "UCB",
    "best_acq_params": next_point,
    "kernel": best_kernel_name,
    "gp_health": best_gp_health,
    "anomalies": anomalies
        })

# Also append σ for plotting
        # If you only have one selected next point, still wrap sigma:
        history_sigma.append(np.array([sigma_all[0]]))


        # Update training data
        X_train_scaled = np.vstack([X_train_scaled, next_scaled])
        y_train_scaled = np.vstack([y_train_scaled, [[y_next_scaled]]])
        history_X.append(next_point)
        history_y.append(y_next)

        # Log iteration
        middle_log_ano(i, cfg["name"], "UCB", gp, next_point, y_next, best_gp_health, anomalies=anomalies, acq_params=None)
        best_results.append({
            "iteration": i + 1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": best_kernel_name,
            "acquisition": "UCB",
            "gp_health": best_gp_health
        })

    # -----------------------
    # Final report / save
    # -----------------------
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

    # Return best observed
    best_idx = np.argmax(history_y)
    return history_X[best_idx], history_y[best_idx], {"X": history_X, "y": history_y}, best_results

