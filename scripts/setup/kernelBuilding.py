from sklearn.gaussian_process.kernels import (
    Matern, RBF, WhiteKernel, ConstantKernel as C, RationalQuadratic, DotProduct, ExpSineSquared
)
import importlib

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
        kernel = default_kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*input_dim))

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
    kernel_class = KERNEL_MAP.get(kernel_type)
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
    elif kernel_type == "Polynomial":
        kern = kernel_class(
            degree=settings.get("degree", 3),
            coef0=settings.get("coef0", 1)
        )

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
    if kernel_type.lower() == "poly":
        svr_model = SVR(kernel="poly", degree=degree, coef0=coef0, C=C_val,
                        epsilon=epsilon, gamma=gamma)
    elif kernel_type.lower() in ["rbf", "linear", "sigmoid"]:
        svr_model = SVR(kernel=kernel_type.lower(), C=C_val, epsilon=epsilon, gamma=gamma)
    else:
        raise ValueError(f"Unknown SVR kernel_type: {kernel_type}")

    return svr_model





