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


def build_dynamic_gp(
    X_train, y_train, config=None, kernel_override=None,
    iteration=0, total_iterations=30, seed=42
):

    # Build kernel dynamically
    kernel = build_dynamic_kernel(
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

