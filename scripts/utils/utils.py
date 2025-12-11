    
import os
import numpy as np 
import pandas as pd
import uuid
import scripts.configs.functionConfig as funcConfig

def set_alpha(y_train):
    """Automatically determine alpha from y scale
    """

    y_range = np.max(y_train) - np.min(y_train)
    alpha = max(1e-6, 1e-4 * y_range)
    return alpha  # small fraction of y range


# def save_weekly_results(results, weekno, name="experiment"):
#     # define output directory (relative to project root or script path)
#     out_dir = os.path.join("analysis", "data", "weeklybestpredictions",f"week{weekno}")
#     # create the directory tree if it doesn't exist (safe if already exists)
#     os.makedirs(out_dir, exist_ok=True)
  

#     short_id = uuid.uuid4().hex[:6]  

#     fname = f"overall_best_predictions_week{weekno}_{name}_{short_id}.csv"
#     out_path = os.path.join(out_dir, fname)


#     rows = []


#     for iter_num, res in results.items():
#         fname = funcConfig.FUNCTION_CONFIG[iter_num]["name"]
#         best_method = max(res, key=lambda m: res[m]["best_y"])
#         best_y = res[best_method]["best_y"]
#         best_x = res[best_method]["best_x"]

#     # Find iteration that produced best_y
#         best_iter = max(res[best_method]["best_results"], key=lambda r: r["best_output"])
    
#         kernel = best_iter.get("kernel", "N/A")
#         kernel_params = best_iter.get("kernel_params", {})
#         gp_health = best_iter.get("gp_health", "N/A")
#         n_candidates = best_iter.get("n_candidates", "N/A")
#         iteration = best_iter.get("iteration", "N/A")

#     # Acquisition-specific parameters
#         acq_params = {}
#         if best_iter["acquisition"].upper() == "UCB":
#             acq_params["kappa"] = best_iter.get("kappa", "N/A")
#         elif best_iter["acquisition"].upper() == "EI":
#             acq_params["gamma"] = best_iter.get("gamma", "N/A")
#         elif best_iter["acquisition"].upper() == "PI":
#             acq_params["eta"] = best_iter.get("eta", "N/A")
    
#     # Flatten input and kernel params for CSV
#         row = {
#         "Function": fname,
#         "Best Method": best_method,
#         "Best Y": best_y,
#         "Best X": "-".join([f"{v:.6f}" for v in best_x]),
#         "Kernel": kernel,
#         "Kernel Params": ", ".join([f"{k}={v}" for k, v in kernel_params.items()]),
#         "GP Health": gp_health,
#         "N Candidates": n_candidates,
#         "Iteration": iteration,
#         "Acquisition Params": ", ".join([f"{k}={v}" for k,v in acq_params.items()])
#         }

#         rows.append(row)


# # Convert to DataFrame and save as CSV
#     df = pd.DataFrame(rows)
#     df.to_csv(out_path, index=False)
#     print("CSV file '{fname}.csv' created.")
def save_weekly_results(results, weekno, name="experiment"):
    import os, uuid, pandas as pd

    # -------------------------
    # Create output directory
    # -------------------------
    out_dir = os.path.join("analysis", "data", "weeklybestpredictions", f"week{weekno}")
    os.makedirs(out_dir, exist_ok=True)

    short_id = uuid.uuid4().hex[:6]
    fname = f"overall_best_predictions_week{weekno}_{name}_{short_id}.csv"
    out_path = os.path.join(out_dir, fname)

    rows = []

    # ------------------------------------------------------
    # MAIN LOOP — iterate through all functions in results
    # ------------------------------------------------------
    for func_id, res in results.items():

        if not res:
            print(f"[Warning] Function {func_id} has no acquisition results — skipping.")
            continue

        func_name = funcConfig.FUNCTION_CONFIG[func_id]["name"]

        # ------------------------------------------------------
        # Find best acquisition method (EI/UCB/etc.)
        # ------------------------------------------------------
        try:
            best_method = max(res, key=lambda m: res[m]["best_y"])
        except Exception:
            print(f"[Warning] Function {func_id} has incomplete acquisition entries — skipping.")
            continue

        best_entry = res[best_method]
        best_y = best_entry.get("best_y", None)
        best_x = best_entry.get("best_x", None)
        best_results_list = best_entry.get("best_results", [])

        if not best_results_list:
            print(f"[Warning] Function {func_id} has no best_results — skipping.")
            continue

        # ------------------------------------------------------
        # Pick the iteration that produced the absolute best output
        # ------------------------------------------------------
        try:
            best_iter = max(best_results_list, key=lambda r: r.get("best_output", float('-inf')))
        except Exception as e:
            print(f"[Warning] Could not select best iteration for function {func_id}: {e}")
            continue

        # ------------------------------------------------------
        # Extract stored details
        # ------------------------------------------------------
        kernel = best_iter.get("kernel", "N/A")
        kernel_params = best_iter.get("kernel_params", {})
        gp_health = best_iter.get("gp_health", "N/A")
        n_candidates = best_iter.get("n_candidates", "N/A")
        iteration = best_iter.get("iteration", "N/A")

        # ------------------------------------------------------
        # Acquisition + parameters
        # ------------------------------------------------------
        acq_name = best_iter.get("acquisition", "N/A")
        acq_params_dict = best_iter.get("acq_params", {})

        # Flatten acquisition params into string
        acq_params_str = ", ".join([f"{k}={v}" for k, v in acq_params_dict.items()]) \
                         if acq_params_dict else "N/A"

        # ------------------------------------------------------
        # Compose CSV row
        # ------------------------------------------------------
        row = {
            "Function": func_name,
            "Best Method": best_method,              # EI / UCB / PI / THOMPSON
            "Acquisition": acq_name,                 # name stored in best_iter
            "Acquisition Params": acq_params_str,    # kappa=..., gamma=..., etc.
            "Best Y": best_y,
            "Best X": "-".join([f"{v:.6f}" for v in best_x]) if best_x is not None else "N/A",
            "Kernel": kernel,
            "Kernel Params": ", ".join([f"{k}={v}" for k, v in kernel_params.items()]),
            "GP Health": gp_health,
            "N Candidates": n_candidates,
            "Iteration": iteration
        }

        rows.append(row)

    # -------------------------
    # Save CSV
    # -------------------------
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"CSV file '{fname}' created at: {out_path}")


def get_func_number_from_config(config_dict, target_config):
    for k, v in config_dict.items():
        if v == target_config:
            return k
    return None  # not found

def get_func_number_by_name(config_dict, func_name):
    for k, v in config_dict.items():
        if v.get("name") == func_name:
            return k
    return None
def check_convergence(history, threshold=1e-3, window=5):
    """
    Checks if the best output has not improved significantly over the last `window` iterations.
    Returns True if convergence is detected.
    """
    if len(history["y"]) < window + 1:
        return False
    recent_best = history["y"][-(window+1):]
    improvement = np.max(recent_best) - np.min(recent_best)
    return improvement < threshold


