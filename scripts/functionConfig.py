FUNCTION_CONFIG = {
    1: {  # 2D Contamination
        "dim": 2,
        "acquisition": "UCB",
        "kernel_type": "Matern",
        "length_scale": [1.0, 1.0],
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e3),
        "alpha": 1e-6,
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "boundary_penalty": True
    },
    2: {  # 2D Noisy Log-Likelihood
        "dim": 2,
        "acquisition": "UCB",
        "kernel_type": "Matern",
        "length_scale": [1.0, 1.0],
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e3),
        "alpha": 1e-3,
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "boundary_penalty": True
    },
    3: {  # 3D Drug Combination
        "dim": 3,
        "acquisition": "PI",
        "kernel_type": "Matern",
        "length_scale": [1.0]*3,
        "length_scale_bounds": (1e-5, 1e8),
        "C_bounds": (1e-5, 1e5),
        "alpha": 1e-6,
        "normalize_y": True,
        "n_restarts_optimizer": 20,
        "boundary_penalty": True
    },
    4: {  # 4D Warehouse Placement
        "dim": 4,
        "acquisition": "UCB",
        "kernel_type": "RBF",  # or Polynomial if you want to experiment
        "length_scale": [1.0]*4,
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e3),
        "alpha": 1e-3,
        "normalize_y": True,
        "n_restarts_optimizer": 15,
        "boundary_penalty": True
    },
    5: {  # 4D Chemical Yield
        "dim": 4,
        "acquisition": "UCB",  # switched from PI to UCB
        "kernel_type": "RBF",
        "length_scale": [1.0]*4,
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e3),
        "alpha": 1e-3,
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "boundary_penalty": False  # optional
    },
    6: {  # 5D Cake Recipe
        "dim": 5,
        "acquisition": "UCB",
        "kernel_type": "Matern",
        "length_scale": [1.0]*5,
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e4),  # larger upper bound
        "alpha": 1e-6,
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "boundary_penalty": True
    },
    7: {  # 6D ML Hyperparameters
        "dim": 6,
        "acquisition": "UCB",
        "kernel_type": "Matern",
        "length_scale": [1.0]*6,
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e4),
        "alpha": 1e-6,
        "normalize_y": True,
        "n_restarts_optimizer": 10,
        "boundary_penalty": True
    },
    8: {  # 8D ML Hyperparameters
        "dim": 8,
        "acquisition": "UCB",
        "kernel_type": "Matern",
        "length_scale": [1.0]*8,
        "length_scale_bounds": (1e-2, 1e2),
        "C_bounds": (1e-3, 1e4),
        "alpha": 1e-6,
        "normalize_y": True,
        "n_restarts_optimizer": 5,  # fewer to save time in high dimensions
        "boundary_penalty": True
    }
}
