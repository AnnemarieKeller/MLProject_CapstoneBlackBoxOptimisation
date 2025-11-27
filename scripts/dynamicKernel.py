from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)
import numpy as np
from scipy.spatial.distance import pdist

def build_dynamic_kernel(
    X_train=None,
    y_train=None,
    config=None,
    kernel_override=None,
    iteration=0,
    total_iterations=30
):
    """
    Fully stable dynamic kernel builder.
    Avoids:
        - noise hitting lower bound warnings
        - length_scale dimension mismatch
        - degenerate kernels
    """

    # -----------------------------
    # 0. Manual override
    # -----------------------------
    if kernel_override is not None:
        return kernel_override

    cfg = config or {}

    # determine dimension
    dim = X_train.shape[1] if X_train is not None else cfg.get("dim", 1)

    # -----------------------------
    # 1. Handle case with no data
    # -----------------------------
    if X_train is None or y_train is None or len(X_train) < 2:
        return (
            C(1.0, (1e-3, 1e3)) *
            RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )

    # -----------------------------
    # 2. Noise estimation
    # -----------------------------
    y_std = np.std(y_train)

    # never allow noise to collapse, avoid bound warnings
    noise_init = max(1e-4, 0.1 * y_std)

    # decay upper noise bound over time — but never too small
    decay = np.exp(-iteration / total_iterations)
    noise_upper = max(5e-3, y_std * decay)

    # final robust noise bounds
    noise_bounds = (1e-4, noise_upper)

    # -----------------------------
    # 3. Length-scale estimation
    # -----------------------------
    avg_dist = np.mean(pdist(X_train)) if X_train.shape[0] > 2 else 1.0
    length_init = np.ones(dim) * avg_dist

    # default geometric bounds
    raw_lb, raw_ub = avg_dist / 50, avg_dist * 20

    # user override
    length_bounds_cfg = cfg.get("length_scale_bounds", (raw_lb, raw_ub))

    # --- normalize bounds to correct dimensionality ---
    if isinstance(length_bounds_cfg[0], (float, int)):
        # scalar bounds → expand for multi-dim
        if dim == 1:
            length_bounds = (float(length_bounds_cfg[0]),
                             float(length_bounds_cfg[1]))
        else:
            length_bounds = (
                np.full(dim, float(length_bounds_cfg[0])),
                np.full(dim, float(length_bounds_cfg[1])),
            )
    else:
        # already arrays
        lb = np.array(length_bounds_cfg[0])
        ub = np.array(length_bounds_cfg[1])

        # fix incorrectly sized arrays
        if lb.size != dim:
            lb = np.full(dim, lb.min())
        if ub.size != dim:
            ub = np.full(dim, ub.max())

        length_bounds = (lb, ub)

    # -----------------------------
    # 4. Kernel selection
    # -----------------------------
    kernel_type = cfg.get("kernel_type", "RBF").lower()

    if kernel_type == "rbf":
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    elif kernel_type == "matern":
        nu = cfg.get("nu", 2.5)
        base_kernel = Matern(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            nu=nu
        )

    elif kernel_type == "rationalquadratic":
        alpha_rq = cfg.get("alpha_rq", 1.0)
        base_kernel = RationalQuadratic(
            length_scale=length_init,
            length_scale_bounds=length_bounds,
            alpha=alpha_rq
        )

    else:
        base_kernel = RBF(length_scale=length_init, length_scale_bounds=length_bounds)

    # -----------------------------
    # 5. Constant kernel (C * K)
    # -----------------------------
    C_val = cfg.get("C", 1.0)
    C_bounds = cfg.get("C_bounds", (1e-3, 1e3))

    if isinstance(C_bounds[0], (list, np.ndarray)):
        C_bounds = (float(C_bounds[0][0]), float(C_bounds[1][0]))
    else:
        C_bounds = (float(C_bounds[0]), float(C_bounds[1]))

    full_kernel = C(C_val, C_bounds) * base_kernel

    # -----------------------------
    # 6. Add noise
    # -----------------------------
    full_kernel += WhiteKernel(
        noise_level=noise_init,
        noise_level_bounds=noise_bounds
    )

    return full_kernel
