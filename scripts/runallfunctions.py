from collections import defaultdict
import numpy as np
import scripts.configs.functionConfig as funcConfig
from scripts.utils.generateX_Y import generate_data
from scripts.week8strat import *
from scripts.updatedweek8strat import *
from scripts.utils.utils import save_weekly_results
import random
import string
from collections import defaultdict
from scripts.function5tryit import * 


def runallfunctions(weekno, externalref=None):
    """
    Run adaptive strategy BBO for all functions of a given week, save results and CSV.

    Parameters:
        weekno (int): week number
        externalref (str, optional): string to identify this run. 
                                      If None, a random string is generated.
    Returns:
        results (dict): nested dict of results for all functions.
    """
    # Generate random string if not provided
    if externalref is None:
        externalref = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    results = defaultdict(dict)

    for iter_num, cfg in funcConfig.FUNCTION_CONFIG.items():
        print(f"Function {iter_num} -> {cfg['name']}, dim={cfg['dim']}, acquisition={cfg['acquisition']}")

        # Generate initial data
        X_train, y_train = generate_data(iter_num, weekno)

        # Run adaptive BBO
        # best_input, best_output, history, best_results = adaptive_bbo_weekly_strategy100(
        #     iter_num, cfg, X_train, y_train,
        #     export_prefix=f"log_{externalref}_week{weekno}"
           
        # )  
        best_input, best_output, history, best_results =   adaptive_bbo_weekly_strategy_updated(
            iter_num, cfg, X_train, y_train,
            export_prefix=f"upd{externalref}_week{weekno}"
           
            )  
  

        print(f"Function {iter_num} finished. Best output: {best_output} Best Input {best_input}")

        # Find best iteration
        best_idx = np.argmax([r["best_output"] for r in best_results])
        best_result = best_results[best_idx]

        best_input = best_result["best_input"]
        best_output = best_result["best_output"]
        best_acq_name = best_result["acquisition"]
        best_gp_health = best_result["gp_health"]

        # Store using acquisition name as key
        results[iter_num][best_acq_name] = {
            "best_x": best_input,
            "best_y": best_output,
            "history": history,
            "best_results": best_results,
            "best_gp_health": best_gp_health,
            "best_iteration": best_idx + 1
        }

    # Save CSV for all functions
    save_weekly_results(results, weekno, f"week{weekno}_{externalref}")

    # Print summary
    for iter_num, res in results.items():
        fname = funcConfig.FUNCTION_CONFIG[iter_num]["name"]
        best_method = max(res, key=lambda m: res[m]["best_y"])
        best_y = res[best_method]["best_y"]
        best_x = res[best_method]["best_x"]
        print(f"Function {fname} -> Best method: {best_method}, best_y: {best_y:.6f}, best_x: {best_x}")

    return results


def runFunct5(week_no):
    X_train, y_train = generate_data(5,week_no)
    adaptive_bbo_multi_peak(
    X_train, y_train, num_iterations=30,
    base_candidates=500, candidate_scale=200,
    scale_candidates=False, random_state=42,
    export_prefix="Adapted_length_bounds_funct5"
    )
