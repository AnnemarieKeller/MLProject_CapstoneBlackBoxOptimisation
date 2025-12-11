import os
import csv
import logging
from datetime import datetime
import pandas as pd


class BOLogger:
    """
    Centralized logger for Bayesian Optimization experiments.
    Handles:
    - start of run logging
    - iteration logging
    - strategy logging
    - GP health logging
    - export of all X/y data at end
    """

    def __init__(self, log_dir="analysis/data/weeklybestpredictions/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Python logger for console output
        self.logger = logging.getLogger("BBO")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.csv_path = None   # will be filled in start
        self.active = False

    # -----------------------------------------------------
    # Start of experiment logging
    # -----------------------------------------------------
    def log_start(self, config, prefix="BO_experiments"):
        """
        Create a CSV file and write the header for basic BO runs.
        """
        function_name = config.get("name", "unknown_func")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")[8:]

        filename = f"{prefix}_func_{function_name}_{timestamp}.csv"
        self.csv_path = os.path.join(self.log_dir, filename)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "function_name",
                "acquisition",
                "kernel",
                "kernel_params",
                "x_next",
                "y_pred",
                "gp_health"
            ])

        self.active = True
        self.logger.info(f"Logging started → {self.csv_path}")

    # -----------------------------------------------------
    # Start logging for strategy-based BO
    # -----------------------------------------------------
    def log_start_strategy(self, config, prefix="BO_strategy_experiments"):
        """
        Create a CSV file with expanded fields for strategy-based runs.
        """
        function_name = config.get("name", "unknown_func")
        timestamp = datetime.now().strftime("%b%d_%y_%H%M%S").lower()

        filename = f"{prefix}_func_{function_name}_{timestamp}.csv"
        self.csv_path = os.path.join(self.log_dir, filename)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "function_name",
                "acquisition",
                "kernel",
                "kernel_params",
                "x_next",
                "y_pred",
                "gp_health",
                "cond",
                "avg_sigma",
                "loglike",
                "strategy"
            ])

        self.active = True
        self.logger.info(f"Strategy logging started → {self.csv_path}")

    # -----------------------------------------------------
    # Log a single iteration (simple BO)
    # -----------------------------------------------------
    def log_iteration(self, i, function_name, acquisition, kernel,
                      kernel_params, x_next, y_pred, gp_health):
        """
        Log one iteration into CSV + console (simple BO version).
        """
        if not self.active:
            raise RuntimeError("log_iteration called before log_start")

        self.logger.info(
            f"Iter {i+1} | Acq:{acquisition} | Kernel:{kernel} | "
            f"Next X:{x_next} | Pred y:{y_pred:.6f} | GP health:{gp_health:.3f}"
        )

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1,
                function_name,
                acquisition,
                kernel,
                kernel_params,
                x_next,
                y_pred,
                gp_health
            ])

    # -----------------------------------------------------
    # Log a single iteration (strategy BO)
    # -----------------------------------------------------
    def log_iteration_strategy(self, i, function_name,
                               acquisition, kernel, kernel_params,
                               x_next, y_pred,
                               gp_health, cond, avg_sigma, loglike,
                               strategy):
        """
        Full-feature iteration logger for runs that track GP internals.
        """
        if not self.active:
            raise RuntimeError("log_iteration_strategy called before log_start_strategy")

        self.logger.info(
            f"Iter {i+1} | Strategy:{strategy} | GP health:{gp_health:.3f} | "
            f"Acq:{acquisition} | Kernel:{kernel} | Next X:{x_next} | Pred y:{y_pred:.6f}"
        )

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i+1,
                function_name,
                acquisition,
                kernel,
                kernel_params,
                x_next,
                y_pred,
                gp_health,
                cond,
                avg_sigma,
                loglike,
                strategy
            ])

    # -----------------------------------------------------
    # Log full dataset at end of run
    # -----------------------------------------------------
    def log_save_history(self, X_all, y_all, prefix="BO_experiments"):
        short_id = datetime.now().strftime("%b%d_%y_%H%M%S").lower()
        filename = f"{prefix}_all_{short_id}.csv"
        out_path = os.path.join(self.log_dir, filename)

        df = pd.DataFrame(X_all, columns=[f"x{i}" for i in range(X_all.shape[1])])
        df["y"] = y_all
        df.to_csv(out_path, index=False)

        self.logger.info(f"Full history saved → {out_path}")

