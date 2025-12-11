import numpy as np
import pandas as pd
import os
import csv
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
from scripts.utils.selfHealing_BO import *
from scripts.analysis.gphealth import *
from datetime import datetime
import matplotlib.pyplot as plt



logger = logging.getLogger("BBO")
logger.setLevel(logging.INFO)
if not logger.handlers:  # Only add handler if none exist
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

csv_file = None
function_name = None
gp_health_history = []
folder_name = None
GP_HEALTH_THRESHOLD = 0.5  # customize this threshold




def start_log(config, export_prefix="BO_experiments", log_dir=None):
    """
    Initialize logging folder for weekly BBO experiments.

    Parameters
    ----------
    config : dict
        Function configuration dictionary.
    export_prefix : str
        Prefix for folder/log files.
    log_dir : str or None
        If provided, use this folder instead of creating a new one.

    Returns
    -------
    str
        Full folder path where logs are saved.
    """
    from datetime import datetime
    import os
    global csv_file, function_name
    folder_name = make_date_folder()  
    if log_dir is None:
        folder_name = make_date_folder()  # e.g., "20251209"
        log_dir = os.path.join(
            "analysis", "data", "weeklybestpredictions", "logs", folder_name, export_prefix
        )
    log_dir = os.path.join(
            log_dir, folder_name
        )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")[8:]
    function_name = config.get("name", "unknown_func")

    # Optional: set global variables if still used elsewhere
  
    csv_file = os.path.join(log_dir, f"{function_name}_{timestamp}.csv")
    folder_name = os.path.basename(log_dir)
    

    


def log_start_strategy(config, export_prefix="BO_strategy_experiments"):
    folder_name = make_date_folder()

    out_dir = os.path.join(
        "analysis", "data", "weeklybestpredictions", "logs", folder_name, "strat"
    )
    os.makedirs(out_dir, exist_ok=True)
     
    global csv_file
    timestamp = datetime.now().strftime("%b%d_%y_%H%M%S").lower()

    function_name = config.get("name", "unknown_func")
     
    file_name = f"{export_prefix}_iter_{function_name}_{timestamp}.csv"
    out_path = os.path.join(out_dir, function_name)
    csv_file = out_path, file_name

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "function_name", 
            "acquisition", "kernel", "kernel_param",
            "x_next", "y_pred", 
            "gp_health", "cond", "avg_sigma", "loglike",
            "strategy"
        ])

def log_middle_strategy(i, function_name,
               best_acq_name, best_kernel_name, kernel_used,
               kernel_params, next_point, y_next,
               gp_health, cond, avg_sigma, loglike,
               strategy):

    logger.info(
        f"Iter {i+1} | Strategy:{strategy} | GP health:{gp_health:.3f} | "
        f"Acq:{best_acq_name} | Kernel:{kernel_used} | "
        f"Next X:{next_point} | Pred y:{y_next:.6f}"
    )
    global csv_file

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
            gp_health,
            cond,
            avg_sigma,
            loglike,
            strategy
        ])
  



def middle_log(i,function_name, best_acq_name, best_kernel_name, kernel_used, kernel_params, next_point, y_next, gp_valid):
      # --- Log to console ---
        logger.info(
            f"Iter {i+1} | Acq: {best_acq_name} | Kernel: {kernel_used} | Next input: {next_point} | Predicted y: {y_next:.6f} | GP valid: {gp_valid}"
        )
        global csv_file

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


def save_log(X_all, y_all,  export_prefix="BO_experiments", threshold=1e-3, window=5):
    """
    Saves the full history of X and y with convergence info.
    Adds a 'convergence_iter' column showing iterations since convergence.
    Includes threshold and window as metadata in the CSV.
    """
    global function_name
    folder_name = make_date_folder()

    out_dir = os.path.join(
        "analysis", "data", "weeklybestpredictions", "logs", folder_name, export_prefix, "all"
    )
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Convergence detection ---
    convergence_iter = np.zeros(len(y_all), dtype=int)
    first_converged_idx = None
    
    for i in range(window, len(y_all)):
        recent_y = y_all[i-window:i+1]
        if np.max(recent_y) - np.min(recent_y) < threshold:
            first_converged_idx = i
            break
    
    if first_converged_idx is not None:
        # Count iterations since first convergence
        for j in range(first_converged_idx, len(y_all)):
            convergence_iter[j] = j - first_converged_idx + 1

    # --- Build DataFrame ---
    df = pd.DataFrame(X_all, columns=[f"x{i}" for i in range(X_all.shape[1])])
    df["y"] = y_all
    df["convergence_iter"] = convergence_iter

    # --- Save ---
    short_id = datetime.now().strftime("%b%d_%y_%H%M%S").lower()
    file_path = os.path.join(out_dir, f"{export_prefix}_all_{function_name}_{short_id}.csv")

    # Write metadata + DataFrame
    with open(file_path, "w") as f:
        f.write(f"# threshold={threshold}\n")
        f.write(f"# window={window}\n")
        df.to_csv(f, index=False)

    return file_path

def log_gp_health(gp_health, cond, avg_sigma, loglike, iteration=None, prefix="GP"):
    """
    Log Gaussian Process health metrics in a reusable way.
    """
    if iteration is not None:
        prefix = f"Iter {iteration+1} | {prefix}"

    logger.info(
        f"{prefix} | health={gp_health:.3f} | cond={cond:.3e} | "
        f"avg_sigma={avg_sigma:.4f} | loglike={loglike:.3f}"
    )

def middle_log_ano(
    i,
    function_name,
    best_acq_name,
    gp,
    next_point,
    y_next,
    gp_health_score,
    anomalies, 
    acq_params
):
    global csv_file

    kernel_str = str(gp.kernel_)
    kernel_params = gp.kernel_.get_params()

    logger.info(
        f"Iter {i+1} | Acq:{best_acq_name} | Kernel:{kernel_str} | "
        f"X:{next_point.tolist()} | y_pred:{y_next:.6f} | "
        f"GP health:{gp_health_score:.4f} | anomalies:{anomalies}"
    )

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            function_name,
            best_acq_name,
            kernel_str,
            kernel_params,
            next_point.tolist(),
            y_next,
            gp_health_score,
            anomalies, acq_params
        ])

def make_date_folder():
    now = datetime.now()
    
    # Month
    month = now.strftime("%b").lower()  # dec
    
    # Day number
    day = now.day
    
    # Ordinal suffix
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1:"st", 2:"nd", 3:"rd"}.get(day % 10, "th")
    
    day_str = f"{day}{suffix}"  # "9th"
    
    # year
    year = now.strftime("%y")  # 25
    
    return f"{month}{day_str}{year}"   # dec9th25



def save_gp_health_plot(health_history, export_prefix = "plots"):
    global folder_name, function_name
    folder_name = make_date_folder()
    
    out_dir = os.path.join(
        "analysis", "data", "weeklybestpredictions", "logs", folder_name,export_prefix,"plot"
    )
    timestamp = datetime.now().strftime("%b%d_%y_%H%M%S").lower()
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10,4))
    plt.plot(health_history, marker="o")
    plt.title("GP Health Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Health Score")
    plt.grid()

    path = os.path.join(out_dir, f"{function_name}gp_health_plot{timestamp}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"GP Health plot saved → {path}")





def log_gp_health_iteration(i, gp_health, cond, avg_sigma, loglike, anomalies=None, csv_path=None):
    """
    Log GP health for a single iteration and optionally write to CSV.
    """
    # Ensure numeric scalars
    gp_health_val = float(gp_health)
    cond_val = float(cond) if not np.isnan(cond) else -1.0
    avg_sigma_val = float(avg_sigma)
    loglike_val = float(loglike)

    n_anom = len(anomalies) if anomalies else 0
    sev_anom = np.mean([a['severity'] for a in anomalies]) if anomalies else 0

    # Compose log message
    msg = (f"Iter {i+1} | GP health={gp_health_val:.3f} | "
           f"cond={cond_val:.3e} | avg_sigma={avg_sigma_val:.4f} | loglike={loglike_val:.3f} "
           f"| anomalies={n_anom} (avg severity={sev_anom:.3f})")

    # Log to console
    import logging
    logger = logging.getLogger(__name__)
    logger.info(msg)

    # Save to CSV
    with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
    
            i + 1,             # iteration
            gp_health_val,     # GP health
            cond_val,          # condition number
            avg_sigma_val,     # average sigma
            loglike_val,       # log-likelihood
            n_anom,            # number of anomalies
            sev_anom           # average severity
    ])

    if gp_health_val < GP_HEALTH_THRESHOLD:
        logger.warning(f"Iter {i+1} → GP health BAD: {gp_health_val:.3f}")
        # Optional: log to CSV immediately
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i+1, "GP health BAD", gp_health_val])


def log_gp_health_iteration1(i, gp_health, cond, avg_sigma, loglike):
    """
    Log GP health and append to history. Trigger alert if health is bad.
    """
   
    gp_health_val = safe_mean(gp_health)
    cond_val = safe_mean(cond)
    avg_sigma_val = safe_mean(avg_sigma)
    loglike_val = safe_mean(loglike)

    logger.info(
    f"Iter {i+1} | GP health={gp_health_val:.3f} | cond={cond_val:.3e} | "
    f"avg_sigma={avg_sigma_val:.4f} | loglike={loglike_val:.3f}"
    )

    global gp_health_history
    gp_health_history.append(gp_health_val)
  
    

    if gp_health_val < GP_HEALTH_THRESHOLD:
        logger.warning(f"Iter {i+1} → GP health BAD: {gp_health_val:.3f}")
        # Optional: log to CSV immediately
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i+1, "GP health BAD", gp_health_val])


def log_acquisition(i, acq_name, x_next, acq_value, **params):
    """
    Log acquisition function info including parameters like gamma/kappa/etc.
    """
    global csv_file

    logger.info(
        f"Iter {i+1} | Acquisition:{acq_name} | X_next:{x_next.tolist()} | "
        f"Value:{acq_value:.6f} | Params:{params}"
    )

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            "acquisition",
            acq_name,
            x_next.tolist(),
            acq_value,
            params
        ])
def log_iteration(
    i,
    function_name,
    acquisition_name,
    acq_params,
    kernel_name,
    kernel_params,
    x_next,
    y_pred,
    gp_health,
    cond,
    avg_sigma,
    loglike,
    anomalies=None
):
    """
    Unified logging for BO iteration.
    Logs kernel, acquisition, GP health, next point, predicted y, and anomalies.
    """
    global csv_file
    
    # --- Console logging ---
    logger.info(
        f"Iter {i+1} | Acq:{acquisition_name}({acq_params}) | Kernel:{kernel_name}({kernel_params}) | "
        f"X:{x_next.tolist()} | y_pred:{y_pred:.6f} | GP health:{gp_health:.3f} | "
        f"cond:{cond:.3e} | avg_sigma:{avg_sigma:.4f} | loglike:{loglike:.3f}" +
        (f" | anomalies:{anomalies}" if anomalies is not None else "")
    )

    # --- CSV logging ---
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            function_name,
            acquisition_name,
            acq_params,
            kernel_name,
            kernel_params,
            x_next.tolist(),
            y_pred,
            gp_health,
            cond,
            avg_sigma,
            loglike,
            anomalies
        ])

    # --- Return a flag if GP health is too bad ---
    if gp_health < GP_HEALTH_THRESHOLD:
        logger.warning(f"Iter {i+1} → GP health BAD: {gp_health:.3f}")
        return True
    return False

def safe_mean(value):
    if np.isscalar(value):
        return value
    elif isinstance(value, dict):
        return np.mean(list(value.values()))
    elif isinstance(value, (list, np.ndarray)):
        # Check if it's a list of dicts
        if all(isinstance(v, dict) for v in value):
            all_vals = []
            for d in value:
                all_vals.extend(d.values())
            return np.mean(all_vals)
        else:
            return np.mean(value)
    else:
        return float(value)

def log_gp_anomalies(i, anomalies, kernel_name, kernel_params):
    """
    Logs GP anomalies to console and CSV.
    """
    global csv_file
    n_anom = len(anomalies)
    sev_anom = np.mean([a["severity"] for a in anomalies]) if anomalies else 0.0

    msg = f"Iter {i+1} | GP anomalies={n_anom} (avg severity={sev_anom:.2f})"
    if n_anom > 0:
        msg += " | " + ", ".join([a["type"] for a in anomalies])

    logger.warning(msg)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            "GP anomalies",
            kernel_name,
            kernel_params,
            n_anom,
            sev_anom,
            [a["type"] for a in anomalies]
        ])


def log_gp_health_iteration_updated(i, gp_health, cond, avg_sigma, loglike, gp, X, y):
    """
    Logs GP health and anomalies in one go. Updates history.
    """
    global gp_health_history

    # --- Compute GP health ---
    gp_health_history.append(float(gp_health))

    # --- Detect anomalies ---
    anomalies = detect_gp_anomalies(gp, X, y)

    # --- Console logging ---
    kernel_name = gp.kernel_.__class__.__name__
    kernel_params = gp.kernel_.get_params()
    logger.info(
        f"Iter {i+1} | GP health={gp_health:.3f} | cond={cond:.3e} | "
        f"avg_sigma={avg_sigma:.4f} | loglike={loglike:.3f} | "
        f"anomalies={len(anomalies)}"
    )

    # --- CSV logging ---
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            "GP health",
            gp_health,
            cond,
            avg_sigma,
            loglike,
            len(anomalies),
            np.mean([a["severity"] for a in anomalies]) if anomalies else 0.0,
            [a["type"] for a in anomalies]
        ])

    # --- Individual anomaly logging ---
    if anomalies:
        log_gp_anomalies(i, anomalies, kernel_name, kernel_params)

    # --- Trigger repair if needed ---
    if gp_health < GP_HEALTH_THRESHOLD or anomalies:
        logger.warning(f"Iter {i+1} → GP health low or anomalies detected. Consider repair/restart.")
        return anomalies
    return None

def log_gp_anomaly(iteration, anomaly, kernel_name, kernel_params):
    global csv_file

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"ANOMALY-{iteration}",
            anomaly["type"],
            anomaly["severity"],
            kernel_name,
            kernel_params,
            anomaly["details"],
        ])

    logger.warning(
        f"[GP ANOMALY] Iter {iteration}: {anomaly['type']} | "
        f"Severity={anomaly['severity']} | Details={anomaly['details']}"
    )





