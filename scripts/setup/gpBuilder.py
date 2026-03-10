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
    
            length_bounds = cfg.get("length_scale_bounds", (avg_dist / 100, avg_dist * 10))
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
# import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, Kernel
from scipy.spatial.distance import pdist

def _clone_bounds(bounds):
    """Ensure each kernel gets independent bounds arrays."""
    lb, ub = bounds
    lb = np.array(lb, copy=True)
    ub = np.array(ub, copy=True)
    return (lb, ub)

def _ensure_length_bounds(raw_bounds, dim):
    """Convert user-provided length_scale_bounds to correct shape."""
    if isinstance(raw_bounds[0], (int, float)):
        return (float(raw_bounds[0]), float(raw_bounds[1]))
    else:
        lb, ub = np.asarray(raw_bounds[0]), np.asarray(raw_bounds[1])
        lb = np.full(dim, lb[0]) if lb.size != dim else lb
        ub = np.full(dim, ub[0]) if ub.size != dim else ub
        return (lb, ub)

# import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C, WhiteKernel, RationalQuadratic, Kernel
)

# def build_dynamic_gp_improved(
#     X_train,
#     y_train,
#     config=None,
#     kernel_override=None,
#     iteration=0,
#     total_iterations=30,
#     seed=None,
#     alpha=1e-10
# ):
#     """
#     Build a dynamic Gaussian Process regressor with robust kernel handling,
#     including composite kernels: RBF + Matern + Constant + WhiteKernel.

#     Ensures that the bounds always match the number of hyperparameters.
#     """

#     cfg = config or {}
#     dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

#     # -----------------------------
#     # User kernel override
#     # -----------------------------
#     if kernel_override is not None:
#         if not isinstance(kernel_override, Kernel):
#             raise ValueError(f"kernel_override must be a sklearn Kernel, got {type(kernel_override)}")
#         kernel = kernel_override
#     else:
#         # -----------------------------
#         # Small dataset fallback
#         # -----------------------------
#         if X_train is None or y_train is None or len(X_train) < 2:
#             kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2)) \
#                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
#         else:
#             # -----------------------------
#             # Dynamic noise level
#             # -----------------------------
#             y_std = np.std(y_train) if len(y_train) > 0 else 1.0
#             noise_init = max(1e-6, 0.1 * y_std)
#             decay = np.exp(-iteration / max(total_iterations, 1))
#             noise_upper = max(1e-3, y_std * decay)
#             noise_bounds = (1e-8, noise_upper)

#             # -----------------------------
#             # Length scale initialization
#             # -----------------------------
#             avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
#             length_init = np.full(dim, avg_dist)
#             length_bounds = cfg.get("length_scale_bounds", (avg_dist / 100, avg_dist * 10))
#             # ensure bounds are arrays of correct shape
#             if isinstance(length_bounds[0], (int, float)):
#                 length_bounds = (np.full(dim, length_bounds[0]), np.full(dim, length_bounds[1]))
#             else:
#                 lb, ub = np.asarray(length_bounds[0]), np.asarray(length_bounds[1])
#                 lb = np.full(dim, lb[0]) if lb.size != dim else lb
#                 ub = np.full(dim, ub[0]) if ub.size != dim else ub
#                 length_bounds = (lb, ub)

#             # -----------------------------
#             # Kernel selection
#             # -----------------------------
#             kernel_type = cfg.get("kernel_type", "composite").lower()

#             if kernel_type == "rbf":
#                 kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#             elif kernel_type == "matern":
#                 nu = cfg.get("nu", 2.5)
#                 kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
#             elif kernel_type == "rationalquadratic":
#                 alpha_init = max(cfg.get("alpha_rq", 1.0), 1e-3)
#                 alpha_bounds = cfg.get("alpha_rq_bounds", (1e-3, 1e3))
#                 alpha_bounds = (max(float(alpha_bounds[0]), 1e-6), max(float(alpha_bounds[1]), 1e-6))
#                 kernel = RationalQuadratic(length_scale=length_init,
#                                            length_scale_bounds=length_bounds,
#                                            alpha=alpha_init,
#                                            alpha_bounds=alpha_bounds)
#             else:
#                 # Composite default: Constant * (RBF + Matern) + WhiteKernel
#                 c_val = 1.0
#                 c_bounds = (1e-6, 1e2)
#                 c_kernel = C(c_val, c_bounds)
                
#                 rbf = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#                 matern = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=2.5)
#                 white = WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

#                 kernel = c_kernel * (rbf + matern) + white

#     # -----------------------------
#     # Build GP regressor
#     # -----------------------------
#     gp = GaussianProcessRegressor(
#     kernel=kernel,
#     alpha=alpha,
#     n_restarts_optimizer=5,
#     optimizer= None,  # default, can switch to None if you want no optimization
#     normalize_y=True,
#     random_state=seed
# )


#     # -----------------------------
#     # Fit GP
#     # -----------------------------
#     gp.fit(X_train, y_train)

#     return gp
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C, WhiteKernel, RationalQuadratic, Kernel
)

# def build_dynamic_gp_improved(
#     X_train,
#     y_train,
#     config=None,
#     kernel_override=None,
#     iteration=0,
#     total_iterations=30,
#     seed=None,
#     alpha=1e-10
# ):
#     """
#     Robust dynamic Gaussian Process builder.
    
#     - Handles small datasets safely.
#     - Automatically adjusts length scales and bounds to match input dimension.
#     - Supports RBF, Matern, RationalQuadratic, or composite kernels.
#     - Avoids theta mismatch errors for all dimensions.
#     """

#     cfg = config or {}
#     dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

#     # -----------------------------
#     # User kernel override
#     # -----------------------------
#     if kernel_override is not None:
#         if not isinstance(kernel_override, Kernel):
#             raise ValueError(f"kernel_override must be a sklearn Kernel, got {type(kernel_override)}")
#         kernel = kernel_override
#     else:
#         # -----------------------------
#         # Small dataset fallback
#         # -----------------------------
#         if X_train is None or y_train is None or len(X_train) < 2:
#             kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2*np.ones(dim), 1e2*np.ones(dim))) \
#                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
#         else:
#             # -----------------------------
#             # Dynamic noise level
#             # -----------------------------
#             y_std = np.std(y_train) if len(y_train) > 0 else 1.0
#             noise_init = max(1e-6, 0.1 * y_std)
#             decay = np.exp(-iteration / max(total_iterations, 1))
#             noise_upper = max(1e-3, y_std * decay)
#             noise_bounds = (1e-8, noise_upper)

#             # -----------------------------
#             # Length scale initialization
#             # -----------------------------
#             avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
#             length_init = np.full(dim, avg_dist)

#             # Default bounds
#             length_bounds = cfg.get("length_scale_bounds", (avg_dist/100, avg_dist*10))
#             # Ensure bounds arrays match dimension
#             if isinstance(length_bounds[0], (int, float)):
#                 length_bounds = (np.full(dim, length_bounds[0]), np.full(dim, length_bounds[1]))
#             else:
#                 lb, ub = np.asarray(length_bounds[0]), np.asarray(length_bounds[1])
#                 lb = np.full(dim, lb[0]) if lb.size != dim else lb
#                 ub = np.full(dim, ub[0]) if ub.size != dim else ub
#                 length_bounds = (lb, ub)

#             # -----------------------------
#             # Kernel selection
#             # -----------------------------
#             kernel_type = cfg.get("kernel_type", "composite").lower()

#             if kernel_type == "rbf":
#                 kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#             elif kernel_type == "matern":
#                 nu = cfg.get("nu", 2.5)
#                 kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
#             elif kernel_type == "rationalquadratic":
#                 alpha_init = max(cfg.get("alpha_rq", 1.0), 1e-3)
#                 alpha_bounds = cfg.get("alpha_rq_bounds", (1e-3, 1e3))
#                 alpha_bounds = (max(float(alpha_bounds[0]), 1e-6), max(float(alpha_bounds[1]), 1e-6))
#                 kernel = RationalQuadratic(length_scale=length_init,
#                                            length_scale_bounds=length_bounds,
#                                            alpha=alpha_init,
#                                            alpha_bounds=alpha_bounds)
#             else:
#                 # -----------------------------
#                 # Composite kernel (safe)
#                 # -----------------------------
#                 c_kernel = C(1.0, (1e-3, 1e2))
#                 rbf = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#                 matern = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=2.5)
#                 white = WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

#                 # Ensure all sub-kernels have correct hyperparameter dimensions
#                 kernel = c_kernel * (rbf + matern) + white

#     # -----------------------------
#     # Build GP regressor
#     # -----------------------------
#     gp = GaussianProcessRegressor(
#         kernel=kernel,
#         alpha=alpha,
#         n_restarts_optimizer=5,  # keep optimizer on, safe for small datasets
#         normalize_y=True,
#         random_state=seed
#     )

#     # -----------------------------
#     # Fit GP
#     # -----------------------------
#     gp.fit(X_train, y_train)

#     return gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)
import numpy as np
from scipy.spatial.distance import pdist

# def build_dynamic_gp_improved(
#     X_train,
#     y_train,
#     config=None,
#     kernel_override=None,
#     iteration=0,
#     total_iterations=30,
#     seed=None,
#     alpha=1e-10
# ):
#     """
#     Build a dynamic Gaussian Process regressor with robust kernel handling,
#     including composite kernels: RBF + Matern + Constant + WhiteKernel.
    
#     Automatically ensures hyperparameter bounds match number of parameters.
#     """
#     cfg = config or {}
#     dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

#     # -----------------------------
#     # User kernel override
#     # -----------------------------
#     if kernel_override is not None:
#         if not hasattr(kernel_override, "theta"):
#             raise ValueError(f"kernel_override must be a sklearn Kernel, got {type(kernel_override)}")
#         kernel = kernel_override

#     else:
#         # -----------------------------
#         # Small dataset fallback
#         # -----------------------------
#         if X_train is None or y_train is None or len(X_train) < 2:
#             kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2)) \
#                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
#         else:
#             # -----------------------------
#             # Dynamic noise level
#             # -----------------------------
#             y_std = np.std(y_train) if len(y_train) > 0 else 1.0
#             noise_init = max(1e-6, 0.1 * y_std)
#             decay = np.exp(-iteration / max(total_iterations, 1))
#             noise_upper = max(1e-3, y_std * decay)
#             noise_bounds = np.array([[1e-8, noise_upper]])  # must be (1,2)

#             # -----------------------------
#             # Length scale initialization
#             # -----------------------------
#             avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
#             length_init = np.full(dim, avg_dist)

#             # Ensure bounds are arrays with shape (dim,2)
#             length_bounds = cfg.get("length_scale_bounds", (avg_dist / 100, avg_dist * 10))
#             if isinstance(length_bounds[0], (int, float)):
#                 length_bounds = np.tile(np.array(length_bounds), (dim, 1))
#             else:
#                 lb = np.asarray(length_bounds[0])
#                 ub = np.asarray(length_bounds[1])
#                 lb = np.full(dim, lb[0]) if lb.size != dim else lb
#                 ub = np.full(dim, ub[0]) if ub.size != dim else ub
#                 length_bounds = np.stack([lb, ub], axis=1)  # shape (dim,2)

#             # -----------------------------
#             # Kernel selection
#             # -----------------------------
#             kernel_type = cfg.get("kernel_type", "composite").lower()

#             if kernel_type == "rbf":
#                 kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#             elif kernel_type == "matern":
#                 nu = cfg.get("nu", 2.5)
#                 kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
#             elif kernel_type == "rationalquadratic":
#                 alpha_init = max(cfg.get("alpha_rq", 1.0), 1e-3)
#                 alpha_bounds = cfg.get("alpha_rq_bounds", (1e-3, 1e3))
#                 kernel = RationalQuadratic(
#                     length_scale=length_init,
#                     length_scale_bounds=length_bounds,
#                     alpha=alpha_init,
#                     alpha_bounds=alpha_bounds
#                 )
#             else:
#                 # Composite default: Constant * (RBF + Matern) + WhiteKernel
#                 c_val = 1.0
#                 c_bounds = (1e-6, 1e2)
#                 c_kernel = C(c_val, c_bounds)

#                 rbf = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#                 matern = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=2.5)
#                 white = WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

#                 kernel = c_kernel * (rbf + matern) + white

#     # -----------------------------
#     # Build GP regressor
#     # -----------------------------
#     gp = GaussianProcessRegressor(
#         kernel=kernel,
#         alpha=alpha,
#         n_restarts_optimizer=5,
#         optimizer="fmin_l_bfgs_b",  # you can switch to None to skip optimization
#         normalize_y=True,
#         random_state=seed
#     )

#     # -----------------------------
#     # Fit GP
#     # -----------------------------
#     gp.fit(X_train, y_train)

#     return gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C, RationalQuadratic
import numpy as np
from scipy.spatial.distance import pdist

# def build_dynamic_gp_improved(
#     X_train,
#     y_train,
#     config=None,
#     kernel_override=None,
#     iteration=0,
#     total_iterations=30,
#     seed=None,
#     alpha=1e-10
# ):
#     """
#     Robust GP builder that dynamically handles composite kernels and
#     ensures all bounds match the number of hyperparameters (theta).

#     Works for: RBF, Matern, RationalQuadratic, and composite kernels.
#     """
    
#     cfg = config or {}
#     dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

#     # -----------------------------
#     # Kernel override
#     # -----------------------------
#     if kernel_override is not None:
#         kernel = kernel_override
#     else:
#         # -----------------------------
#         # Compute dynamic noise
#         # -----------------------------
#         y_std = np.std(y_train) if len(y_train) > 0 else 1.0
#         noise_init = max(1e-8, 0.1 * y_std)
#         decay = np.exp(-iteration / max(total_iterations,1))
#         noise_upper = max(1e-8, y_std * decay)
#         noise_bounds = (1e-8, noise_upper)

#         # -----------------------------
#         # Compute length scale
#         # -----------------------------
#         avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
#         length_init = np.full(dim, avg_dist)
#         length_bounds = np.tile([avg_dist/100, avg_dist*10], (dim,1))  # shape (dim,2)
        
#         # -----------------------------
#         # Choose kernel
#         # -----------------------------
#         ktype = cfg.get("kernel_type", "composite").lower()
        
#         if ktype == "rbf":
#             kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds) + WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)
#         elif ktype == "matern":
#             kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=2.5) + WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)
#         elif ktype == "rationalquadratic":
#             alpha_init = cfg.get("alpha_rq", 1.0)
#             alpha_bounds = cfg.get("alpha_rq_bounds", (1e-3, 1e3))
#             kernel = RationalQuadratic(length_scale=length_init, length_scale_bounds=length_bounds,
#                                        alpha=alpha_init, alpha_bounds=alpha_bounds) + WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)
#         else:
#             # Composite default: Constant*(RBF + Matern) + White
#             c_kernel = C(1.0, (1e-6,1e2))
#             rbf = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
#             matern = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=2.5)
#             white = WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)
#             kernel = c_kernel * (rbf + matern) + white

#     # -----------------------------
#     # Build GP regressor
#     # -----------------------------
#     gp = GaussianProcessRegressor(
#         kernel=kernel,
#         alpha=alpha,
#         n_restarts_optimizer=5,
#         normalize_y=True,
#         random_state=seed
#     )

#     # -----------------------------
#     # Fit GP
#     # -----------------------------
#     gp.fit(X_train, y_train)

#     return gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C, RationalQuadratic
import numpy as np
from scipy.spatial.distance import pdist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

# def build_dynamic_gp_improved(
#     X_train, y_train, config,
#     kernel_override=None,
#     iteration=0,
#     total_iterations=1,
#     seed=None,
#     alpha=1e-6,
#     disable_optimizer=False
# ):
#     """
#     Build a Gaussian Process using the improved adaptive method.
#     Fixes theta mismatch by always using a stable kernel structure.
#     """
#     ls_floor = 1e-2
#     ls_ceil = 5.0

    
#     # -----------------------------
#     # Create kernel
#     # -----------------------------
#     if kernel_override is not None:
#         kernel = kernel_override
#     else:
#         # Ensure stable number of hyperparameters
#         # rbf = RBF(length_scale=config.get("rbf_length_scale", 1.0))
#         # matern = Matern(length_scale=config.get("matern_length_scale", 1.0), nu=1.5)
#         # white = WhiteKernel(noise_level=config.get("white_noise", 1e-6))
#         # constant = C(constant_value=config.get("constant_value", 1.0))
#         # -----------------------------
# # Stable, regularized kernel (NO collapse)
# # -----------------------------
#         dim = X_train.shape[1]

# # Lengthscale floor prevents collapse
#         # ls_floor = 1e-2
#         # ls_ceil = 5.0

#         rbf = RBF(
#         length_scale=np.full(dim, config.get("rbf_length_scale", 0.5)),
#     length_scale_bounds=(ls_floor, ls_ceil)
#         )

#         matern = Matern(
#         length_scale=np.full(dim, config.get("matern_length_scale", 0.5)),
#         length_scale_bounds=(ls_floor, ls_ceil),
#         nu=config.get("nu", 1.5)
#         )

# # Noise floor prevents GP cheating
#         white = WhiteKernel(
#         noise_level=max(config.get("white_noise", 1e-4), 1e-4),
#         noise_level_bounds=(1e-4, 1e-1)
#         )

# # Constant kernel with sane bounds
#         constant = C(
#         constant_value=config.get("constant_value", 1.0),
#         constant_value_bounds=(1e-2, 10.0)
#         )

#         # Compose kernel (theta count stable)
#         kernel = constant * (rbf + matern) + white
#         # This ensures 4 hyperparameters: constant, rbf, matern, white
    
#     # -----------------------------
#     # Build GP
#     # -----------------------------
#     gp = GaussianProcessRegressor(
#         kernel=kernel,
#         alpha=alpha,
#         normalize_y=True,
#         random_state=seed,
#         n_restarts_optimizer=5 if not disable_optimizer else 0
#     )
    
#     # -----------------------------
#     # Fit GP
#     # -----------------------------
#     gp.fit(X_train, y_train)
#     gp._lengthscale_floor = ls_floor

    
#     return gp
def build_dynamic_gp_improved(
    X_train,
    y_train,
    config,
    kernel_override=None,
    iteration=0,
    total_iterations=1,
    seed=None,
    alpha=1e-6,
    disable_optimizer=False
):
    """
  GP BUILDR
    """

    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, Matern, WhiteKernel, ConstantKernel as C, Kernel
    )

    dim = X_train.shape[1]
    cfg = config or {}

    # -----------------------------
    # Kernel override
    # -----------------------------
    if kernel_override is not None:
        if not isinstance(kernel_override, Kernel):
            raise ValueError("kernel_override must be a sklearn Kernel")
        kernel = kernel_override

    else:
        # -----------------------------
        # Correct bounds shape (dim, 2)
        # -----------------------------
        ls_floor = 1e-2
        ls_ceil = 30.0

        length_bounds = np.column_stack([
            np.full(dim, ls_floor),
            np.full(dim, ls_ceil)
        ])

        rbf = RBF(
            length_scale=np.full(dim, cfg.get("rbf_length_scale", 0.5)),
            length_scale_bounds=length_bounds
        )

        matern = Matern(
            length_scale=np.full(dim, cfg.get("matern_length_scale", 0.5)),
            length_scale_bounds=length_bounds,
            nu=cfg.get("nu", 1.5)
        )

        white = WhiteKernel(
            noise_level=max(cfg.get("white_noise", 1e-4), 1e-4),
            noise_level_bounds=(1e-4, 1e-1)
        )

        constant = C(
            constant_value=cfg.get("constant_value", 1.0),
            constant_value_bounds=(1e-2, 10.0)
        )

        kernel = constant * (rbf + matern) + white

    # -----------------------------
    # GP
    # -----------------------------
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        random_state=seed,
        optimizer=None if disable_optimizer else "fmin_l_bfgs_b",
        n_restarts_optimizer=0 if disable_optimizer else 5
    )

    gp.fit(X_train, y_train)
    return gp
