from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from .kernelBuilding import *
from .defaultKernelSettings import DEFAULT_KERNEL_SETTINGS
from scripts.utils.utils import *


def build_gp(config=None, X_train=None, y_train=None, kernel_override=None, use_seed = True, seed= 42):
 
    print(X_train.shape[1])
    input_dim = X_train.shape[1] if X_train is not None else None
    kernel = build_kernel_from_config(config=config, input_dim=input_dim, kernel_override=kernel_override)

    alpha = config.get("alpha", 1e-6) if config else set_alpha(input_dim)
    normalize_y = config.get("normalize_y", True) if config else True
    n_restarts_optimizer = config.get("n_restarts_optimizer", 5) if config else 5
    if use_seed:
        gp = GaussianProcessRegressor(
             kernel=kernel,
             alpha=alpha,
             normalize_y=normalize_y,
             n_restarts_optimizer=n_restarts_optimizer,
             random_state=seed
        )
    else:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer
    )

    if X_train is not None and y_train is not None:
        gp.fit(X_train, y_train)

    return gp
def build_gpWhiteKernel(config = None,  X_train=None, y_train=None,
             kernel_override=None, use_seed=True, seed=42):

    input_dim = X_train.shape[1] if X_train is not None else None

    kernel = build_kernelWithWhiteKernel(
        config = config,
        input_dim=input_dim,
        kernel_override=kernel_override
    )

    # GP should have alpha = 0 since WhiteKernel handles noise
    alpha = 0.0

    normalize_y =  True
    n_restarts = 20

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=seed if use_seed else None
    )

    if X_train is not None and y_train is not None:
        gp.fit(X_train, y_train)

    return gp


def build_svr(X_train, y_train, config=None, config_override=None):
   
    final_config = {**(config or {}), **(config_override or {})}
    svr = build_svrKernel_from_config(config, config_override)

    # svr = SVR(
    #     kernel=final_config.get("svr_kernel", "rbf"),
    #     C=final_config.get("C", 1.0),
    #     epsilon=final_config.get("epsilon", 0.01),
    #     gamma=final_config.get("gamma", "scale")
    # )

    svr.fit(X_train, y_train)
    return svr


def build_dynamic_gp1(
    X_train, y_train, config=None, kernel_override=None,
    iteration=0, total_iterations=30, seed=42
):

    # Build kernel dynamically
    kernel = build_dynamic_kernel_length_handled(
        X_train=X_train,
        y_train=y_train,
        config=config,
        kernel_override=kernel_override,
        iteration=iteration,
        total_iterations=total_iterations
    )

    dim = X_train.shape[1]
    alpha = set_alpha(dim)

    n_restarts = config.get("n_restarts_optimizer", 10) if config else 10

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed
    )

    gp.fit(X_train, y_train)
    # print(f"Built dynamic GP with kernel: {kernel}")
    return gp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import Kernel

def build_dynamic_gp(X_train, y_train, config=None, kernel_override=None,
                     iteration=0, total_iterations=30, seed=None, alpha=1e-10):
    """
    Build a dynamic GP regressor with robust kernel handling.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training input data, shape (n_samples, n_features)
    y_train : np.ndarray
        Training output data, shape (n_samples,)
    config : dict
        Configuration for kernel, noise, length scales, etc.
    kernel_override : sklearn.gaussian_process.kernels.Kernel or None
        If provided, use this kernel directly
    iteration : int
        Current iteration (for dynamic noise decay)
    total_iterations : int
        Total iterations (for noise decay)
    seed : int or None
        Random seed
    alpha : float
        GP noise term for numerical stability
    
    Returns
    -------
    gp : GaussianProcessRegressor
        A fitted GP regressor
    """
    
    if kernel_override is not None:
        if not isinstance(kernel_override, Kernel):
            raise ValueError(f"kernel_override must be a valid sklearn kernel, got {type(kernel_override)}")
        kernel = kernel_override
    else:
        cfg = config or {}

        # Determine dimension
        dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

        # Handle small dataset fallback
        if X_train is None or y_train is None or len(X_train) < 2:
            base_kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
            white = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            kernel = base_kernel + white
        else:
            # Dynamic noise
            y_std = np.std(y_train) if len(y_train) > 0 else 1.0
            noise_init = max(1e-6, 0.1 * y_std)
            decay = np.exp(-iteration / total_iterations)
            noise_upper = max(1e-3, y_std * decay)
            noise_bounds = (1e-8, noise_upper)

            #Dynamic length scales
            avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
            length_init = np.ones(dim) * avg_dist
            length_bounds = cfg.get("length_bounds", (avg_dist / 100, avg_dist * 10))
            if isinstance(length_bounds[0], (int, float)):
                length_bounds = (float(length_bounds[0]), float(length_bounds[1]))
            else:
                lb, ub = np.array(length_bounds[0]), np.array(length_bounds[1])
                if len(lb) != dim:
                    lb = np.full(dim, lb[0])
                if len(ub) != dim:
                    ub = np.full(dim, ub[0])
                length_bounds = (lb, ub)
            
            # Kernel type
            kernel_type = cfg.get("kernel_type", "rbf").lower()
            if kernel_type == "rbf":
                base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
            elif kernel_type == "matern":
                nu = cfg.get("nu", 2.5)
                base_kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
            elif kernel_type == "rationalquadratic":
                alpha_rq = cfg.get("alpha_rq", 1.0)
                base_kernel = RationalQuadratic(length_scale=length_init,
                                                length_scale_bounds=length_bounds,
                                                alpha=alpha_rq)
            else:
                print(f"Unknown kernel type '{kernel_type}', using RBF")
                base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

            # Constant kernel multiplier
            C_val = max(cfg.get("C", 1.0), 1e-6)
            C_bounds = cfg.get("C_bounds", (1e-3, 1e3))
            C_bounds = (max(float(C_bounds[0]), 1e-6), max(float(C_bounds[1]), 1e-6))
            base_kernel = C(C_val, C_bounds) * base_kernel

            #  White kernel
            add_white = cfg.get("add_white", True)
            if add_white:
                base_kernel += WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

            kernel = base_kernel
    cfg = config or {}
    # Create GP regressor
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=alpha,
                                  n_restarts_optimizer=cfg.get("n_restarts_optimizer", 5),
                                  normalize_y=cfg.get("normalize_y", True),
                                  random_state=seed)
    
    # Fit GP
    gp.fit(X_train, y_train)

    return gp
