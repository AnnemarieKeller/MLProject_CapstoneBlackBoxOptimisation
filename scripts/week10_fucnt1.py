import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

from scripts.setup.gpBuilder import build_dynamic_gp
from scripts.analysis.gphealth import (
    get_gp_health,
    detect_gp_anomalies,
    get_gp_health_score
)
from scripts.utils.logging.logger import *
from scripts.utils.reports.reportbuilder import *


def pick_top_2_ucb_candidates(gp, bounds, n_samples=2000, beta=0.5, seed=42):
    """
    Pick top 2 points in 2D input space maximizing UCB acquisition function.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(low, high, n_samples) for low, high in bounds
    ])
    mu, sigma = gp.predict(X, return_std=True)
    ucb = mu + beta * sigma
    top_idx = np.argsort(ucb)[-2:]
    return X[top_idx], mu[top_idx], sigma[top_idx]



def week10_funct1(
    function_id,
    cfg,
    X_init,
    y_init,
    num_iterations=30,
    scale_candidates=False,
    export_prefix="function1_week10",
    random_state=42
):
    np.random.seed(random_state)

    # -----------------------
    # Scaling
    # -----------------------
    if scale_candidates:
        X_scaler = MinMaxScaler().fit(X_init)
        y_scaler = StandardScaler().fit(np.array(y_init).reshape(-1, 1))
        X_train = X_scaler.transform(X_init)
        y_train = y_scaler.transform(np.array(y_init).reshape(-1, 1))
    else:
        X_scaler = y_scaler = None
        X_train = X_init.copy()
        y_train = np.array(y_init).reshape(-1, 1)

    # -----------------------
    # Logging containers
    # -----------------------
    history_X, history_y = [], []
    gp_health_history = []
    iteration_acq_history = []
    history_sigma = []
    best_results = []

    # -----------------------
    # Setup logging
    # -----------------------
    log_dir = os.path.join("reports", export_prefix)
    os.makedirs(log_dir, exist_ok=True)
    start_log(cfg, export_prefix, log_dir)

    bounds = [(0.0, 1.0)] * X_train.shape[1]

    # =====================================================
    # Main BO loop
    # =====================================================
    for i in range(num_iterations):

        # -----------------------
        # Train GP + select kernel
        # -----------------------
        best_gp = None
        best_health = -np.inf
        best_kernel = None
        cond = avg_sigma = anomalies = loglike = None

        for nu in [0.5, 1.5, 2.5]:
            cfg_k = cfg.copy()
            cfg_k["kernel_type"] = "Matern"
            cfg_k["nu"] = nu

            gp_tmp = build_dynamic_gp(
                X_train, y_train,
                cfg_k,
                iteration=i,
                total_iterations=num_iterations,
                seed=random_state
            )

            health, cond, avg_sigma, loglike = get_gp_health(
                gp_tmp, X_train, y_train
            )
            anom = detect_gp_anomalies(gp_tmp, X_train, y_train)

            if health > best_health:
                best_gp = gp_tmp
                best_health = health
                best_kernel = f"Matern(nu={nu})"
                anomalies = anom

        gp = best_gp
        gp_health_history.append(best_health)

        log_gp_health_iteration_updated(
            i, best_health, cond, avg_sigma, loglike,
            gp, X_train, y_train
        )

        # -----------------------
        # Kernel repair (self-healing)
        # -----------------------
        if best_health < 0.6:
            kernel_pool = [
                RBF(),
                Matern(nu=1.5),
                C(1.0) * RBF() + WhiteKernel(1e-3),
                RBF() + Matern(),
                RBF() * Matern()
            ]

            repaired_gp = None
            repaired_health = -np.inf

            for kern in kernel_pool:
                gp_tmp = build_dynamic_gp(
                    X_train, y_train,
                    cfg,
                    iteration=i,
                    total_iterations=num_iterations,
                    seed=random_state,
                    kernel_override=kern
                )

                h, cond, avg_sigma, anom = get_gp_health_score(
                    gp_tmp, X_train, y_train
                )

                if h > repaired_health:
                    repaired_gp = gp_tmp
                    repaired_health = h
                    best_kernel = str(kern)
                    anomalies = anom

            gp = repaired_gp
            best_health = repaired_health

        # -----------------------
        # UCB candidate selection
        # -----------------------
        (X_cand, mu, sigma) = pick_top_2_ucb_candidates(
            gp, bounds, beta=5.0, seed=random_state
        )

        next_scaled = X_cand[0].reshape(1, -1)
        history_sigma.append(np.array([sigma[0]]))

        # -----------------------
        # Predict
        # -----------------------
        y_next_scaled = gp.predict(next_scaled).item()

        if scale_candidates:
            next_point = X_scaler.inverse_transform(next_scaled).flatten()
            y_next = y_scaler.inverse_transform([[y_next_scaled]]).item()
        else:
            next_point = next_scaled.flatten()
            y_next = y_next_scaled

        # -----------------------
        # Logging
        # -----------------------
        iteration_acq_history.append({
            "iteration": i + 1,
            "X_candidates": X_cand,
            "acq_values": {
                "UCB": mu + 5.0 * sigma
                },
            "best_acq_name": "UCB",
            "kernel": best_kernel,
            "gp_health": best_health,
            "anomalies": anomalies
        })

        # middle_log_ano(
        #     i, cfg["name"], "UCB",
        #     gp, next_point, y_next,
        #     best_health, anomalies=anomalies
        # )

        best_results.append({
            "iteration": i + 1,
            "best_input": next_point,
            "best_output": y_next,
            "kernel": best_kernel,
            "acquisition": "UCB",
            "gp_health": best_health
        })

        # -----------------------
        # Update training data
        # -----------------------
        X_train = np.vstack([X_train, next_scaled])
        y_train = np.vstack([y_train, [[y_next_scaled]]])
        history_X.append(next_point)
        history_y.append(y_next)

    # =====================================================
    # Final report
    # =====================================================
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

    best_idx = np.argmax(history_y)
    return history_X[best_idx], history_y[best_idx], {"X": history_X, "y": history_y}, best_results
