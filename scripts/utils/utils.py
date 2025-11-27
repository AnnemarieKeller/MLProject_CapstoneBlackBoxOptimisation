    
import os
import numpy as np 
import pandas as pd
import scripts.functionConfig as funcConfig

def set_alpha(y_train):
    """Automatically determine alpha from y scale
    """

    y_range = np.max(y_train) - np.min(y_train)
    alpha = max(1e-6, 1e-4 * y_range)
    return alpha  # small fraction of y range
import pandas as pd


def save_weekly_results(results, weekno):
    # define output directory (relative to project root or script path)
    out_dir = os.path.join("analysis", "data", "weeklybestresults")
    # create the directory tree if it doesn't exist (safe if already exists)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"overall_best_results_week{weekno}.csv"
    out_path = os.path.join(out_dir, fname)


    rows = []

    for iter_num, res in results.items():
        fname = funcConfig.FUNCTION_CONFIG[iter_num]["name"]
    
    # Find best method for this function
        best_method = max(res, key=lambda m: res[m]["best_y"])
        best_y = res[best_method]["best_y"]
        best_x = res[best_method]["best_x"]
    
    # Find the iteration that produced best_y to get kernel info and n_candidates
        best_iter = max(res[best_method]["best_results"], key=lambda r: r["best_output"])
        kernel = best_iter.get("kernel", "N/A")
        kernel_params = best_iter.get("kernel_params", {})
        n_candidates = best_iter.get("n_candidates", "N/A")
    
    # Flatten input and kernel params for CSV
        row = {
        "Function": fname,
        "Best Method": best_method,
        "Best Y": best_y,
        "Best X": ", ".join([f"{v:.6f}" for v in best_x]),
        "Kernel": kernel,
        "Kernel Params": ", ".join([f"{k}={v}" for k, v in kernel_params.items()]),
        "N Candidates": n_candidates
        }
    
        rows.append(row)
# Convert to DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("CSV file 'overall_best_results.csv' created.")
    