import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

class CommitteeSurrogate:

    def __init__(self, input_dim, n_splits=3):
        self.input_dim = input_dim
        self.n_splits = n_splits

        # -------------------------
        # Define models
        # -------------------------
        kernel = 1.0 * Matern(length_scale=np.ones(input_dim), nu=2.5) + WhiteKernel(noise_level=1e-6)

        self.models = {
            "gp": GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=2,
                normalize_y=True
            ),
            "rf": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ),
            "gbm": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=64
            ),
            "svr": SVR(
                kernel="rbf",
                C=3.0,
                epsilon=0.01,
                gamma="scale"
            ),
            "nn": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=2000,
                learning_rate_init=0.001,
                random_state=42
            )
        }

        self.weights = {name: 1.0 for name in self.models}

    # ----------------------------------------------------
    # Cross-validation error → adaptive model weight
    # ----------------------------------------------------
    def _compute_weights(self, X, y):
        errors = {}
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for name, model in self.models.items():
            fold_errs = []
            for train_idx, val_idx in kf.split(X):
                Xtr, Xval = X[train_idx], X[val_idx]
                ytr, yval = y[train_idx], y[val_idx]
                try:
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xval)
                    fold_errs.append(np.mean((pred - yval) ** 2))
                except Exception as e:
                    print(f"Skipping fold for {name} due to error: {e}")
                    fold_errs.append(np.inf)

            errors[name] = np.mean(fold_errs)

        # invert MSE → weight
        inv = {name: 1 / (err + 1e-9) for name, err in errors.items()}
        total = sum(inv.values())
        self.weights = {name: v / total for name, v in inv.items()}

    # ----------------------------------------------------
    # Fit all models with adaptive weighting
    # ----------------------------------------------------
    def fit(self, X, y):
        self._compute_weights(X, y)

        for name, model in self.models.items():
            try:
                # Skip models that fail due to small dataset
                if name in ["gbm", "nn"] and X.shape[0] < 20:
                    print(f"Skipping {name} due to small dataset")
                    continue
                model.fit(X, y)
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")
                continue

    # ----------------------------------------------------
    # Weighted prediction + disagreement uncertainty
    # ----------------------------------------------------
    def predict(self, X, return_std=True):
        mus = []
        sigmas = []

        N = X.shape[0]

        for name, model in self.models.items():
            try:
                if name == "gp":
                    mu, std = model.predict(X, return_std=True)
                elif name == "rf":
                    # pseudo-variance from trees
                    preds = np.array([tree.predict(X) for tree in model.estimators_])
                    mu = preds.mean(axis=0)
                    std = preds.std(axis=0)
                elif name in ["svr", "nn", "gbm"]:
                    mu = model.predict(X)
                    # small pseudo-variance
                    std = np.ones_like(mu) * 1e-3
                else:
                    raise ValueError(f"Unknown model {name}")

                mus.append(mu.ravel())
                sigmas.append(std.ravel())

            except Exception as e:
                print(f"Skipping {name} in predict due to error: {e}")
                continue

        if not mus:  # no valid models
            raise RuntimeError("No models available for prediction.")

        # stack (n_models, N)
        mus = np.vstack(mus)
        sigmas = np.vstack(sigmas)

        # compute weighted mean
        weights = np.array([self.weights[name] for name in self.models if name in self.weights]).reshape(-1, 1)
        mu_ensemble = np.sum(weights * mus, axis=0) / np.sum(weights)
        sigma_ensemble = np.sqrt(np.sum(weights * (sigmas**2 + (mus - mu_ensemble)**2), axis=0) / np.sum(weights))

        if return_std:
            return mu_ensemble, sigma_ensemble
        else:
            return mu_ensemble
