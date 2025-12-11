from sklearn.gaussian_process.kernels import (
    Matern, RBF, WhiteKernel, ConstantKernel as C, RationalQuadratic, DotProduct, ExpSineSquared
)
import importlib
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
import numpy as np
from scipy.spatial.distance import pdist

from .defaultKernelSettings import DEFAULT_KERNEL_SETTINGS, KERNEL_CLASSES

DEFAULT_KERNEL_SETTINGS = {
    "class": "Matern",
    "C": 1.0,
    "c_bounds": (1e-3, 1e3),
    "length_scale": 1.0,
    "length_scale_bounds": (1e-2, 1e2),
    "nu": 2.5,
    "add_white": True,
    "white_noise": 1e-6,
    "white_bounds": (1e-9, 1e-1)
}

KERNEL_CLASSES = {
    "Matern": Matern,
    "RBF": RBF,
    "RationalQuadratic": RationalQuadratic,
    "DotProduct": DotProduct,
    "ExpSineSquared": ExpSineSquared,
    # add more without touching kernel creation code
}


# def build_kernel_from_config(config, input_dim=None, default_kernel=None):
#   """
#     Build a GP kerne from config  Falls back to default.
    
#     Parameters:
#         configs : 
#             input fromthe functionconfig file 
#         input_dim : int or None
#             Dimensionality of the input (can be calculated from X_train)
#         default_kernel : kernel object or None
#             Kernel to use if kernel_cls is None
#         kwargs : dict
#             Extra arguments for kernel
#     """
#     # Merge defaults with user settings
#     cfg = {**DEFAULT_KERNEL_SETTINGS, **config}
    
#     kernel_type = cfg.get("kernel_type", cfg["class"])
#     kernel_cls = KERNEL_CLASSES.get(kernel_type)
    
#     if kernel_cls is None:
#         print("No kernel class provided, using default kernel instead.")
#         return default_kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*input_dim))
    
#     # Determine input_dim if not explicitly given
#     if input_dim is None:
#         input_dim = cfg.get("dim", 1)

#     # Build kernel depending on type
#     if kernel_type in ["Matern", "RBF", "ExpSineSquared", "RationalQuadratic"]:
#         length_scale = cfg.get("length_scale", [1.0]*input_dim)
#         if isinstance(length_scale, (float, int)):
#             length_scale = [length_scale]*input_dim

#         if kernel_type == "Matern":
#             kernel = kernel_cls(length_scale=length_scale,
#                                 length_scale_bounds=cfg.get("length_scale_bounds", (1e-2, 1e2)),
#                                 nu=cfg.get("nu", 2.5))
#         elif kernel_type == "RBF":
#             kernel = kernel_cls(length_scale=length_scale,
#                                 length_scale_bounds=cfg.get("length_scale_bounds", (1e-2, 1e2)))
#         elif kernel_type == "RationalQuadratic":
#             kernel = kernel_cls(length_scale=length_scale,
#                                 alpha=cfg.get("alpha_rq", 1.0))
#         elif kernel_type == "ExpSineSquared":
#             kernel = kernel_cls(length_scale=length_scale,
#                                 periodicity=cfg.get("periodicity", 1.0))

#     elif kernel_type == "DotProduct":
#         kernel = kernel_cls(sigma_0=cfg.get("sigma_0", 1.0))

#     elif kernel_type == "Polynomial":
#         kernel = kernel_cls( kernel = "poly",
                                # degree=cfg.get("degree", 3),
#                             coef0=cfg.get("coef0", 1.0))
#     else:
#         # fallback
#         kernel = RBF(length_scale=[1.0]*input_dim)
    
#     # Optionally add white noise
#     if cfg.get("add_white", True):
#         kernel += WhiteKernel(
#             noise_level=cfg.get("white_noise", 1e-6),
#             noise_level_bounds=cfg.get("white_bounds", (1e-9, 1e-1))
#         )
    
#     # Optionally scale by ConstantKernel
#     kernel *= C(cfg.get("C", 1.0), cfg.get("C_bounds", (1e-3, 1e3)))
    
#     return kernel


def create_kernel_from_config(config, input_dim=None):
    # Merge defaults with function-specific settings
    cfg = {**DEFAULT_KERNEL_SETTINGS, **config}

    kernel_cls = KERNEL_CLASSES.get(cfg["class"])
    if kernel_cls is None:
        print("No kernel class provided, using default kernel instead.")
        kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*input_dim))

    # Handle length_scale for multi-dimensional inputs
    length_scale = cfg["length_scale"]
    if input_dim is not None and isinstance(length_scale, (float, int)):
        length_scale = [length_scale] * input_dim

    # Instantiate main kernel dynamically
    kwargs = {k: v for k, v in cfg.items() if k not in ["class", "C", "c_bounds", "add_white", "white_noise", "white_bounds"]}
    kernel = C(cfg["C"], cfg["c_bounds"]) * kernel_cls(**kwargs)

    # Optionally add WhiteKernel
    if cfg.get("add_white", True):
        kernel += WhiteKernel(cfg["white_noise"], cfg["white_bounds"])

    return kernel



def build_kernel(kernel_cls=None, input_dim=1, add_white=False, default_kernel=None, **kwargs):
    """
    Build a GP kernel dynamically. Falls back to default if kernel_cls is None or unrecognized.
    
    Parameters:
        kernel_cls : class or None
            Kernel class (RBF, Matern, etc.)
        input_dim : int
            Dimensionality of the input (used for length_scale if needed)
        add_white : bool
            Whether to add WhiteKernel for noise
        default_kernel : kernel object
            Kernel to use if kernel_cls is None
        kwargs : dict
            Extra arguments for kernel
    """
    if kernel_cls is None:
        print("No kernel class provided, using default kernel instead.")
        kernel = default_kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*input_dim))
    else:
        # instantiate kernel dynamically
        kernel = kernel_cls(**kwargs)
        # If input_dim is relevant (like length_scale for RBF/Matern)
        if hasattr(kernel, "length_scale"):
            kernel.length_scale = [1.0]*input_dim

    # Optionally add WhiteKernel
    if add_white:
        kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1))
    
    return kernel


def create_kernel(settings, input_dim=None):
    kernel_type = settings["kernel_type"]
    kernel_class = KERNEL_CLASSES.get(kernel_type)
    if kernel_class is None:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Prepare length_scale if needed
    length_scale = settings.get("length_scale", 1.0)
    if input_dim and isinstance(length_scale, float):
        length_scale = [length_scale] * input_dim

    # Instantiate kernel with dynamic params
    if kernel_type in ["Matern", "RBF"]:
        kern = C(1.0, settings["c_bounds"]) * kernel_class(
            length_scale=length_scale,
            length_scale_bounds=settings["length_scale_bounds"],
            **({ "nu": settings.get("nu", 2.5) } if kernel_type=="Matern" else {})
        )
    elif kernel_type == "RationalQuadratic":
        kern = C(1.0, settings["c_bounds"]) * kernel_class(
            length_scale=length_scale,
            alpha=settings.get("alpha_rq", 1.0)
        )
    elif kernel_type == "DotProduct":
        kern = kernel_class(sigma_0=settings.get("sigma_0", 1.0))
    elif kernel_type == "ExpSineSquared":
        kern = C(1.0, settings["c_bounds"]) * kernel_class(
            length_scale=length_scale,
            periodicity=settings.get("periodicity", 1.0)
        )
    # elif kernel_type == "Polynomial":
    #     kern = kernel_class(
    #         degree=settings.get("degree", 3),
    #         coef0=settings.get("coef0", 1)
    #     )

    # Add WhiteKernel if requested
    if settings.get("add_white", True):
        kern += WhiteKernel(
            noise_level=settings.get("white_noise", 1e-6),
            noise_level_bounds=settings.get("white_bounds", (1e-9, 1e-1))
        )
    
    return kern





def build_kernel_from_config(config=None, input_dim=None, default_kernel=None, kernel_override=None):
    if kernel_override is not None:
        print("Using provided kernel_override instead of building from config.")
        return kernel_override

    # Use centralized default if no config provided
    cfg = {**DEFAULT_KERNEL_SETTINGS, **(config or {})}

    kernel_type = cfg.get("kernel_type", cfg["class"])
    kernel_cls = KERNEL_CLASSES.get(kernel_type)

    if input_dim is None:
        input_dim = cfg.get("dim", 1)

    if kernel_cls is None:
        print(f"No kernel class found for type {kernel_type}, using default kernel.")
        return default_kernel or (C(cfg.get("C", 1.0), cfg.get("C_bounds", (1e-3, 1e3))) *
                                  RBF(length_scale=[cfg.get("length_scale", 1.0)]*input_dim))

    # Build kwargs dynamically
    kwargs = {}
    if kernel_type in ["Matern", "RBF", "RationalQuadratic", "ExpSineSquared"]:
        ls = cfg.get("length_scale", [1.0]*input_dim)
        if isinstance(ls, (int, float)):
            ls = [ls]*input_dim
        kwargs["length_scale"] = ls

    for param in ["nu", "alpha_rq", "periodicity", "degree", "coef0", "sigma_0"]:
        if param in cfg:
            kwargs[param] = cfg[param]

    kernel = kernel_cls(**kwargs)

    if cfg.get("add_white", True):
        kernel += WhiteKernel(
            noise_level=cfg.get("white_noise", 1e-6),
            noise_level_bounds=cfg.get("white_bounds", (1e-9, 1e-1))
        )

    kernel *= C(cfg.get("C", 1.0), cfg.get("C_bounds", (1e-3, 1e3)))

    return kernel
from sklearn.svm import SVR

def build_svrKernel_from_config(config=None, model_override=None):
    """
    Build an SVR model from config dictionary.
    Supports kernels: 'linear', 'rbf', 'poly', 'sigmoid'.
    """
    if model_override is not None:
        return model_override

    cfg = config or {}
    kernel_type = cfg.get("kernel_type", "rbf")  # default RBF
    degree = cfg.get("degree", 3)
    coef0 = cfg.get("coef0", 1)
    C_val = cfg.get("C", 1.0)
    epsilon = cfg.get("epsilon", 0.1)
    gamma = cfg.get("gamma", "scale")  # 'scale', 'auto', or float

    # Build SVR model
   
    if kernel_type.lower() not in ["linear", "poly", "rbf", "sigmoid"]:
        print(f"Warning: kernel_type '{kernel_type}' not supported for SVR. Using 'rbf' instead.")
        kernel_type = "rbf"
        svr_model = SVR(kernel=kernel_type.lower(), C=C_val, epsilon=epsilon, gamma=gamma)

    if kernel_type.lower() == "poly":
        svr_model = SVR(kernel="poly", degree=degree, coef0=coef0, C=C_val,
                        epsilon=epsilon, gamma=gamma)
    elif kernel_type.lower() in ["rbf", "linear", "sigmoid"]:
        svr_model = SVR(kernel=kernel_type.lower(), C=C_val, epsilon=epsilon, gamma=gamma)
    else:
        raise ValueError(f"Unknown SVR kernel_type: {kernel_type}")

    return svr_model

def build_kernelWithWhiteKernel(config=None, input_dim=None, kernel_override=None,
                                X_train=None, y_train=None, iteration=0, total_iterations=30):

    if kernel_override is not None:
        return kernel_override

    cfg = {**DEFAULT_KERNEL_SETTINGS, **(config or {})}
    kernel_type = cfg.get("kernel_type", cfg.get("class", "RBF"))

    # --- Compute dynamic stats if data exists ---
    if X_train is not None and len(X_train) > 1:

        # Average pairwise distance between points
        from scipy.spatial.distance import pdist
        avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0

        length_init = np.ones(input_dim) * avg_dist
        length_bounds = (avg_dist / 100, avg_dist * 10)

        # Dynamic noise estimation
        y_std = np.std(y_train)
        decay = np.exp(-iteration / total_iterations)
        noise_init = max(1e-6, 0.1 * y_std)
        # noise_upper = max(1e-3, y_std * decay)
        # noise_upper = min(noise_upper, 1.0)
        noise_upper = max(1e-3, np.std(y_train) * np.exp(-iteration/total_iterations))
        noise_bounds = (1e-8, noise_upper)

    else:
        # Fallback early in BO loop
        length_init = np.ones(input_dim)
        length_bounds = (1e-2, 1e2)
        noise_init = 1e-3
        noise_bounds = (1e-6, 1e-1)

    # ------------------------------------------------------------
    # Build kernel safely, passing only valid parameters
    # ------------------------------------------------------------
    if kernel_type == "RBF":
        base_kernel = RBF(length_scale=length_init,
                          length_scale_bounds=length_bounds)

    elif kernel_type == "Matern":
        base_kernel = Matern(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            nu=cfg.get("nu", 2.5)
        )

    elif kernel_type == "RationalQuadratic":
        base_kernel = RationalQuadratic(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            alpha=cfg.get("alpha", 1.0)
        )

    elif kernel_type == "ExpSineSquared":
        base_kernel = ExpSineSquared(
            length_scale=length_init,
            periodicity=cfg.get("periodicity", 1.0)
        )

    else:
        print(f"Unknown kernel {kernel_type}, defaulting to RBF")
        base_kernel = RBF(length_scale=length_init,
                          length_scale_bounds=length_bounds)

    # ------------------------------------------------------------
    # Add dynamic WhiteKernel
    # ------------------------------------------------------------
    if cfg.get("add_white", True):
       
    
        base_kernel += WhiteKernel(
            noise_level=noise_init,
            noise_level_bounds=noise_bounds
        )

    return base_kernel



def build_dynamic_kernel(
    X_train=None, 
    y_train=None, 
    config=None, 
    kernel_override=None,
    iteration=0,
    total_iterations=30
):
    """
    Fully dynamic kernel builder with automatic:
        - length_scale initialization
        - length_scale bounds from X geometry
        - noise level initialization from y variability
        - noise bounds that shrink over time
    """

    # --- override if explicitly given ---
    if kernel_override is not None:
        return kernel_override

    cfg = config or {}

    # Determine dimensions
    dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

    # If no data yet, fall back to defaults
    if X_train is None or y_train is None or len(X_train) < 2:
        base_kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
        white = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        return base_kernel + white

    # =====================================================
    #  Dynamic noise estimation from y
    # =====================================================
    y_std = np.std(y_train)

    # Initial noise level ~ 10% of signal std
    noise_init = max(1e-6, 0.1 * y_std)

    # Shrink upper noise bound over time
    decay = np.exp(-iteration / total_iterations)
    noise_upper = max(1e-3, y_std * decay)

    noise_bounds = (1e-8, noise_upper)

    # =====================================================
    # Dynamic length-scale estimation from geometry of X
    # =====================================================

    # average distance between points
    avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0

    # initial length-scale is around this distance
    length_init = np.ones(dim) * avg_dist  

    # bounds scale with geometry
    length_bounds = (avg_dist / 100, avg_dist * 10)

    # optionally allow override
    if "length_bounds" in cfg:
        length_bounds = cfg["length_bounds"]

    # =====================================================
    #  Choose kernel class
    # =====================================================
    kernel_type = cfg.get("kernel_type", "RBF")

    if kernel_type == "RBF":
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    elif kernel_type == "Matern":
        nu = cfg.get("nu", 2.5)
        base_kernel = Matern(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            nu=nu
        )

    elif kernel_type == "RationalQuadratic":
        base_kernel = RationalQuadratic(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            alpha=cfg.get("alpha_rq", 1.0)
        )

    else:  
        print(f"Unknown kernel type {kernel_type}, using RBF.")
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    # =====================================================
    #  Add dynamic WhiteKernel
    # =====================================================
    white = WhiteKernel(
        noise_level=noise_init,
        noise_level_bounds=noise_bounds
    )

    return base_kernel + white


def build_dynamic_kernel_broken(X_train, y_train, config=None, kernel_override=None,
                         iteration=0, total_iterations=30):
    """
    Builds a Gaussian Process kernel dynamically based on input data and config.
    Handles dynamic noise, length scales, multiple kernel types, and constant multipliers.
    Works for any input dimension.
    """

    dim = X_train.shape[1]

    if kernel_override is not None:
        return kernel_override

    cfg = config or {}
    kernel_type = cfg.get("kernel_type", "rbf").lower()
    add_white = cfg.get("add_white", True)

    # ----------------------------
    # 1️⃣ Compute y statistics for noise
    # ----------------------------
    y_std = np.std(y_train) if y_train is not None else 1.0
    noise_init = max(1e-6, 0.1 * y_std)
    decay = np.exp(-iteration / total_iterations)
    noise_upper = max(1e-3, y_std * decay)
    noise_bounds = (1e-8, noise_upper)

    # ----------------------------
    # 2️⃣ Length-scale handling
    # ----------------------------
    avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
    length_init = np.ones(dim) * avg_dist
    length_bounds = cfg.get("length_scale_bounds", (avg_dist / 100, avg_dist * 10))

    # Convert scalar bounds to arrays for multi-dimensional inputs
    if isinstance(length_bounds[0], (int, float)):
        length_bounds = (np.full(dim, length_bounds[0]), np.full(dim, length_bounds[1]))
    else:
        length_bounds = (np.array(length_bounds[0]), np.array(length_bounds[1]))
        if len(length_bounds[0]) != dim:
            length_bounds = (np.full(dim, length_bounds[0][0]), np.full(dim, length_bounds[1][0]))

    # If 1D, convert to scalars
    if dim == 1:
        length_init = float(length_init[0])
        length_bounds = (float(length_bounds[0][0]), float(length_bounds[1][0]))

    # ----------------------------
    # 3️⃣ Select kernel type
    # ----------------------------
    if kernel_type == "rbf":
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
    elif kernel_type == "matern":
        nu = cfg.get("nu", 2.5)
        base_kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
    elif kernel_type == "rationalquadratic":
        alpha_rq = cfg.get("alpha_rq", 1.0)
        base_kernel = RationalQuadratic(length_scale=length_init, length_scale_bounds=length_bounds, alpha=alpha_rq)
    else:
        print(f"Unknown kernel type '{cfg.get('kernel_type')}', using RBF.")
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    # ----------------------------
    # 4️⃣ Constant kernel multiplier
    # ----------------------------
    C_val = cfg.get("C", 1.0)
    C_bounds = cfg.get("C_bounds", (1e-3, 1e3))
    if isinstance(C_bounds[0], (list, np.ndarray)):
        C_bounds = (float(C_bounds[0][0]), float(C_bounds[1][0]))
    else:
        C_bounds = (float(C_bounds[0]), float(C_bounds[1]))
    base_kernel = C(C_val, C_bounds) * base_kernel

    # ----------------------------
    # 5️⃣ Add WhiteKernel dynamically
    # ----------------------------
    if add_white:
        if isinstance(noise_bounds[0], (list, np.ndarray)):
            noise_bounds = (float(noise_bounds[0]), float(noise_bounds[1]))
        base_kernel += WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

    return base_kernel

def build_dynamic_kernel_length_handled(
    X_train=None,
    y_train=None,
    config=None,
    kernel_override=None,
    iteration=0,
    total_iterations=30
):
    """
    Fully dynamic Gaussian Process kernel builder.

    Features:
    - Automatically estimates length_scale from X geometry
    - Supports isotropic (scalar) or anisotropic (per-dim) length scales
    - Dynamic noise from y variability
    - Shrinking noise upper bound over iterations
    - Supports RBF, Matern, RationalQuadratic
    - Adds ConstantKernel multiplier if specified
    """

    if kernel_override is not None:
        return kernel_override

    cfg = config or {}

    # ----------------------------
    # Determine dimensionality
    # ----------------------------
    dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

    # ----------------------------
    # Early return if no data yet
    # ----------------------------
    if X_train is None or y_train is None or len(X_train) < 2:
        base_kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
        white = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        return base_kernel + white

    # ----------------------------
    # 1️⃣ Dynamic noise estimation
    # ----------------------------
    y_std = np.std(y_train)
    noise_init = max(1e-6, 0.1 * y_std)
    decay = np.exp(-iteration / total_iterations)
    noise_upper = max(1e-3, y_std * decay)
    noise_bounds = (1e-8, noise_upper)

    # ----------------------------
    # 2️⃣ Dynamic length-scale estimation
    # ----------------------------
    avg_dist = np.mean(pdist(X_train)) if len(X_train) > 2 else 1.0
    length_init = np.ones(dim) * avg_dist

    # Default bounds
    length_bounds = (avg_dist / 100, avg_dist * 10)

    # Allow config override
    if "length_bounds" in cfg:
        length_bounds = cfg["length_bounds"]

    # Ensure correct shapes
    if isinstance(length_bounds[0], (int, float)):
        # scalar bounds → isotropic
        length_bounds = (float(length_bounds[0]), float(length_bounds[1]))
    else:
        # bounds as arrays → ensure length matches dim
        lb, ub = np.array(length_bounds[0]), np.array(length_bounds[1])
        if len(lb) != dim:
            lb = np.full(dim, lb[0])
        if len(ub) != dim:
            ub = np.full(dim, ub[0])
        length_bounds = (lb, ub)

    # ----------------------------
    # 3️⃣ Select kernel type
    # ----------------------------
    kernel_type = cfg.get("kernel_type", "RBF").lower()
    if kernel_type == "rbf":
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)
    elif kernel_type == "matern":
        nu = cfg.get("nu", 2.5)
        base_kernel = Matern(length_scale=length_init, length_scale_bounds=length_bounds, nu=nu)
    elif kernel_type == "rationalquadratic":
        alpha_rq = cfg.get("alpha_rq", 1.0)
        base_kernel = RationalQuadratic(length_scale=length_init, length_scale_bounds=length_bounds, alpha=alpha_rq)
    else:
        print(f"Unknown kernel type '{cfg.get('kernel_type')}', using RBF")
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    # ----------------------------
    # 4️⃣ Optional Constant kernel multiplier
    # ----------------------------
    C_val = cfg.get("C", 1.0)
    C_bounds = cfg.get("C_bounds", (1e-3, 1e3))
    if isinstance(C_bounds[0], (list, np.ndarray)):
        C_bounds = (float(C_bounds[0][0]), float(C_bounds[1][0]))
    else:
        C_bounds = (float(C_bounds[0]), float(C_bounds[1]))
    base_kernel = C(C_val, C_bounds) * base_kernel

    # ----------------------------
    # 5️⃣ Add dynamic WhiteKernel
    # ----------------------------
    add_white = cfg.get("add_white", True)
    if add_white:
        base_kernel += WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

    return base_kernel

