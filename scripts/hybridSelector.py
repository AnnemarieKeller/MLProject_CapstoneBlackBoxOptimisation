import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor


class HybridGP_RF:
    def __init__(self, input_dim):
        self.input_dim = input_dim

        kernel = 1.0 * Matern(length_scale=np.ones(input_dim), nu=2.5) \
                 + 1e-6 * WhiteKernel()

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True
        )

        self.rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, X, y):
        # Fit GP first
        self.gp.fit(X, y)

        # Compute residuals
        gp_mu = self.gp.predict(X)
        residuals = y - gp_mu

        # Fit RF on residuals
        self.rf.fit(X, residuals)

    def predict(self, X, return_std=True):
        mu_gp, sigma = self.gp.predict(X, return_std=True)
        mu_rf = self.rf.predict(X)

        # Combined prediction
        mu = mu_gp + mu_rf

        return mu, sigma
