# Template 1: Smooth, low-noise surface — RBF + small noise
CONFIG_SMOOTH = {
    "dim": None,            # set later from data X shape
    "kernel_type": "rbf",   # RBF kernel: smooth, infinite differentiable :contentReference[oaicite:2]{index=2}
    "add_white": True,
    "C": 1.0,
    "C_bounds": (1e-3, 1e3),
    "length_scale_bounds": (1e-2, 1e2),
    "nu": None,             # not used for RBF
    "n_restarts_optimizer": 5,   # fewer restarts, since fewer hyperparams
    "normalize_y": False,
}

# Template 2: Rougher / possibly non‑smooth surface — Matérn kernel (nu=2.5) + moderate noise
CONFIG_ROUGH = {
    "dim": None,
    "kernel_type": "matern",
    "add_white": True,
    "C": 1.0,
    "C_bounds": (1e-3, 1e3),
    "length_scale_bounds": (1e-2, 1e2),
    "nu": 2.5,  # popular setting for moderate smoothness / flexibility :contentReference[oaicite:3]{index=3}
    "n_restarts_optimizer": 10,
    "normalize_y": False,
}

# Template 3: Flexible / uncertain surface — hybrid RBF + Matérn + noise (to capture both smooth and rough behavior)
CONFIG_HYBRID = {
    "dim": None,
    "kernel_type": "rbf+matern",  # indicates hybrid kernel (RBF + Matérn)  
    "add_white": True,
    "C": 1.0,
    "C_bounds": (1e-3, 1e3),
    "length_scale_bounds": (1e-2, 1e2),
    "nu": 2.5,  # nu for Matern part
    "n_restarts_optimizer": 15,
    "normalize_y": False,
}
# config = CONFIG_ROUGH.copy()
# config["dim"] = X_train.shape[1]
# Config candidates for GP‑BO kernel / GP hyperparameters

CONFIG_LIST = [

    # 1) Smooth, low‑noise — isotropic RBF
    {
        "name": "RBF_isotropic_smooth",
        "dim": None,  # to set: X_train.shape[1]
        "kernel_type": "rbf",
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        "length_scale_bounds": (1e-2, 1e2),
        # isotropic: uses scalar length_scale internally
        "n_restarts_optimizer": 5,
        "normalize_y": False
    },

    # 2) Rougher surface — isotropic Matern (nu = 2.5)
    {
        "name": "Matern_isotropic_nu2.5",
        "dim": None,
        "kernel_type": "matern",
        "nu": 2.5,
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        "length_scale_bounds": (1e-2, 1e2),
        "n_restarts_optimizer": 8,
        "normalize_y": False
    },

    # 3) Rough + smooth mixed — hybrid RBF + Matern (gives flexibility)
    {
        "name": "Hybrid_RBF_plus_Matern",
        "dim": None,
        "kernel_type": "rbf+matern",
        "nu": 2.5,
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        "length_scale_bounds": (1e-2, 1e2),
        "n_restarts_optimizer": 12,
        "normalize_y": False
    },

    # 4) Anisotropic — per-dimension length-scales (RBF)
    {
        "name": "RBF_anisotropic",
        "dim": None,
        "kernel_type": "rbf",
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        "length_scale_bounds": (1e-2, 1e2),
        "anisotropic": True,             # convention: instruct kernel builder to use array length_scale
        "n_restarts_optimizer": 10,
        "normalize_y": False
    },

    # 5) Noisy, uncertain function — Matern + higher noise allowed
    {
        "name": "Matern_noisy",
        "dim": None,
        "kernel_type": "matern",
        "nu": 1.5,                       # less smooth → more flexible/rough
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        "length_scale_bounds": (1e-3, 1e3),  # wide bounds to adapt to various scales
        "n_restarts_optimizer": 15,
        "normalize_y": False
    },

    # 6) Conservative + regularised — RBF + White noise + modest hyperparameter search
    {
        "name": "RBF_conservative",
        "dim": None,
        "kernel_type": "rbf",
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-2, 1e2),
        "length_scale_bounds": (1e-1, 1e1),
        "n_restarts_optimizer": 3,
        "normalize_y": True  # center outputs to zero-mean
    },

    # 7) Hybrid + regularised noise for mid‑dim problems
    {
        "name": "Hybrid_regular_noise",
        "dim": None,
        "kernel_type": "rbf+matern",
        "nu": 2.5,
        "add_white": True,
        "C": 0.5,
        "C_bounds": (1e-3, 1e2),
        "length_scale_bounds": (1e-1, 1e2),
        "n_restarts_optimizer": 10,
        "normalize_y": True
    },

    # 8) High flexibility — Matern anisotropic + wide bounds
    {
        "name": "Matern_anisotropic_flexible",
        "dim": None,
        "kernel_type": "matern",
        "nu": 2.5,
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e4),
        "length_scale_bounds": (1e-3, 1e3),
        "anisotropic": True,
        "n_restarts_optimizer": 20,
        "normalize_y": False
    },

    # 9) Smooth + safe noise floor — RBF + high noise floor (for possibly noisy data)
    {
        "name": "RBF_noisy_floor",
        "dim": None,
        "kernel_type": "rbf",
        "add_white": True,
        "C": 1.0,
        "C_bounds": (1e-3, 1e3),
        # Use relatively large min length-scale to avoid overfitting small noise
        "length_scale_bounds": (1e-0, 1e2),
        "n_restarts_optimizer": 5,
        "normalize_y": False
    },

    # 10) Basic fallback — default kernel (let GP optimize defaults)
    {
        "name": "Default_sklearn",
        "dim": None,
        # Do not specify kernel_type ⇒ let GPR use default Constant * RBF
        "kernel_type": None,
        "add_white": False,
        "C": None,
        "C_bounds": None,
        "length_scale_bounds": None,
        "n_restarts_optimizer": 5,
        "normalize_y": False
    }
]
# for cfg in CONFIG_LIST:
    # cfg["dim"] = X_train.shape[1]
