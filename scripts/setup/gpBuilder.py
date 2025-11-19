from sklearn.gaussian_process import GaussianProcessRegressor
from kernel_builder import build_kernel_from_config
from defaultKernelSettings import DEFAULT_KERNEL_SETTINGS


def build_gp(config=None, X_train=None, y_train=None, kernel_override=None, use_seed = True, seed= 42):
 
    input_dim = X_train.shape[1] if X_train is not None else None
    kernel = build_kernel_from_config(config=config, input_dim=input_dim, kernel_override=kernel_override)

    alpha = config.get("alpha", 1e-6) if config else DEFAULT_KERNEL_SETTINGS.get("white_noise", 1e-6)
    normalize_y = config.get("normalize_y", True) if config else True
    n_restarts_optimizer = config.get("n_restarts_optimizer", 5) if config else 5
    if use_seed:
        gp = GaussianProcessRegressor(
             kernel=kernel,
             alpha=alpha,
             normalize_y=normalize_y,
             n_restarts_optimizer=n_restarts_optimizer
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
