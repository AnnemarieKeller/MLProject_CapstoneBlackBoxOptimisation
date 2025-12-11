import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, pairwise_distances, davies_bouldin_score
from scripts.utils.generateX_Y import *
from scripts.BBOloop import *
import scripts.configs.functionConfig as funcConfig





import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def analyze_blackbox_space(X, y, gp_model=None, highlight_sparse=False, n_neighbors=5):
    n_samples, n_features = X.shape
    
    # Reduce dimensionality for visualisation if required
    if n_features > 3:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        xlabel, ylabel = "PC1", "PC2"
    else:
        X_vis = X.copy()
        xlabel, ylabel = "x0", "x1"
    
    # Make sure y_vis aligns with X_vis
    # If you truncated/filtered X into X_vis, do the same for y
    y_vis = y[: X_vis.shape[0]]
    
    # Compute local density if highlighting sparse regions
    if highlight_sparse:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_vis)
        distances, _ = nbrs.kneighbors(X_vis)
        local_density = distances.mean(axis=1)
    else:
        local_density = np.zeros(X_vis.shape[0])
    
    # Plot scatter
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_vis[:, 0],
        X_vis[:, 1],
        c=y_vis,         # use y_vis, not full y
        cmap='viridis',
        s=80,
        edgecolor='k'
    )
    plt.colorbar(scatter, label='Output (y)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Black-box Function Samples')
    
    # Highlight sparse points
    if highlight_sparse:
        thresh = np.percentile(local_density, 75)
        sparse_mask = local_density >= thresh
        plt.scatter(
            X_vis[sparse_mask, 0],
            X_vis[sparse_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=120,
            linewidths=2,
            label='Sparse regions'
        )
        plt.legend()
    
    # If GP model and plotting in 2D
    if gp_model is not None and n_features <= 3:
        grid_size = 50
        x0 = np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), grid_size)
        x1 = np.linspace(X_vis[:, 1].min(), X_vis[:, 1].max(), grid_size)
        xx0, xx1 = np.meshgrid(x0, x1)
        grid_vis = np.column_stack([xx0.ravel(), xx1.ravel()])
        
        # Map grid_vis back to original space if PCA used
        if n_features > 2:  
            grid_orig = pca.inverse_transform(grid_vis)
        else:
            grid_orig = grid_vis
        
        y_pred, sigma = gp_model.predict(grid_orig, return_std=True)
        y_pred = y_pred.reshape(grid_size, grid_size)
        
        plt.contour(xx0, xx1, y_pred, levels=15, cmap='coolwarm', alpha=0.5)
        plt.title('Samples + GP Prediction Contours')
    
    plt.show()



def recommend_bo_strategy(X, y, noise_threshold=0.05, sparse_percentile=25):
    """
    Recommend Bayesian Optimization strategy based on observed inputs/outputs.

    Parameters:
    -----------
    X : np.ndarray
        Input samples, shape (n_samples, n_features)
    y : np.ndarray
        Corresponding outputs, shape (n_samples,)
    noise_threshold : float
        Fraction of output variance considered "noisy"
    sparse_percentile : int
        Percentile of local density to consider "sparse"

    Returns:
    --------
    strategy : dict
        Suggested BO strategy
    """
    strategy = {}

    n_samples, n_features = X.shape
    y_var = np.var(y)
    
    # --- 1. Noise estimation ---
    # Compute local output variation for nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(5, n_samples)).fit(X)
    distances, indices = nbrs.kneighbors(X)
    local_output_var = np.array([
        np.var(y[indices[i]]) for i in range(n_samples)
    ])
    avg_local_var = local_output_var.mean()
    
    noise_ratio = avg_local_var / (y_var + 1e-12)
    is_noisy = noise_ratio > noise_threshold
    
    strategy['is_noisy'] = is_noisy
    strategy['noise_ratio'] = noise_ratio
    
    # --- 2. Sparsity / coverage ---
    avg_distances = distances.mean(axis=1)
    sparse_threshold = np.percentile(avg_distances, 100 - sparse_percentile)
    sparse_ratio = (avg_distances > sparse_threshold).mean()
    
    strategy['sparse_ratio'] = sparse_ratio
    strategy['sparse_threshold'] = sparse_threshold

    # --- 3. Local optima estimate ---
    # Count points that are local maxima among neighbors
    local_maxima = 0
    for i in range(n_samples):
        if y[i] >= y[indices[i]].max():
            local_maxima += 1
    local_maxima_ratio = local_maxima / n_samples
    strategy['local_maxima_ratio'] = local_maxima_ratio

    # --- 4. Recommend kernel & acquisition ---
    if is_noisy:
        strategy['kernel'] = 'Matern'
        strategy['add_white'] = True
        strategy['alpha'] = 1e-3
    else:
        strategy['kernel'] = 'RBF'
        strategy['add_white'] = False
        strategy['alpha'] = 1e-6
    
    # Acquisition function
    if sparse_ratio > 0.2:  # many sparse regions → encourage exploration
        strategy['acquisition'] = 'UCB'
        strategy['ucb_beta'] = 2.0  # higher beta → more exploration
    else:
        strategy['acquisition'] = 'EI'  # exploitation-oriented
    
    # Suggest number of restarts for hyperparameter optimization
    strategy['n_restarts_optimizer'] = min(max(n_features*3, 5), 20)
    
    return strategy
# strategy = recommend_bo_strategy(X, y)
# print("Recommended BO Strategy:")
# for k,v in strategy.items():
#     print(f"{k}: {v}")



def analyze_function(X, Y, top_frac=0.1):
    """
    Analyze X,Y data for BO configuration suggestions.
    Returns a diagnostics dict and recommended config dict.
    
    Parameters:
        X : np.ndarray, shape (n_samples, n_features)
        Y : np.ndarray, shape (n_samples,)
        top_frac : float, fraction of top outputs to check for multimodality
    
    Returns:
        diagnostics : dict
        bo_config : dict
    """
    n_samples, dim = X.shape
    
    # -----------------
    # 1. Noise estimation (local variability)
    # -----------------
    dists = squareform(pdist(X))
    np.fill_diagonal(dists, np.nan)
    
    # Find nearest neighbors for each point
    nn_idx = np.nanargmin(dists, axis=1)
    Y_diff = np.abs(Y - Y[nn_idx])
    noise_estimate = np.median(Y_diff)
    
    # -----------------
    # 2. Output scale
    # -----------------
    Y_range = Y.max() - Y.min()
    
    # -----------------
    # 3. Smoothness estimate (Spearman correlation of input distances vs output distances)
    # -----------------
    smooth_corr = spearmanr_correlation(X, Y)
    
    # -----------------
    # 4. Dimensional relevance via RandomForest
    # -----------------
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, Y)
    feature_importances = rf.feature_importances_
    
    # -----------------
    # 5. Multi-modality estimate (clustering top outputs)
    # -----------------
    top_k = max(int(top_frac * n_samples), 2)
    top_idx = np.argsort(Y)[-top_k:]
    top_X = X[top_idx]
    
    n_clusters = min(3, top_k)  # max 3 clusters to avoid overfitting
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = kmeans.fit_predict(top_X)
    multimodal_score = len(np.unique(labels)) / n_clusters  # 1 = multimodal, 0 = unimodal
    
    # -----------------
    # Diagnostics summary
    # -----------------
    diagnostics = {
        "dim": dim,
        "noise_estimate": noise_estimate,
        "output_range": Y_range,
        "smoothness_corr": smooth_corr,
        "feature_importances": feature_importances,
        "multimodal_score": multimodal_score
    }
    
    # -----------------
    # BO config recommendation
    # -----------------
    # Determine kernel type
    if noise_estimate > 0.01 * Y_range:
        kernel_type = "Matern"
        alpha = max(noise_estimate**2, 1e-6)
        nu = 1.5
    else:
        kernel_type = "RBF"
        alpha = 1e-6
        nu = 2.5
    
    # Acquisition
    if multimodal_score > 0.5 or smooth_corr < 0.5:
        acquisition = "UCB"
        beta = 2.0
    else:
        acquisition = "EI"
        beta = None
    
    # Normalize outputs if range is large
    normalize_y = True if Y_range > 1 else False
    
    # Suggest length scale
    length_scale = np.maximum(0.1, np.std(X, axis=0))
    
    # Number of restarts
    n_restarts_optimizer = 10 if dim <= 5 else 20
    
    bo_config = {
        "kernel_type": kernel_type,
        "alpha": alpha,
        "nu": nu if kernel_type=="Matern" else None,
        "acquisition": acquisition,
        "beta": beta,
        "normalize_y": normalize_y,
        "length_scale": length_scale.tolist(),
        "length_scale_bounds": (1e-2, 1e2),
        "n_restarts_optimizer": n_restarts_optimizer,
        "add_white": True if alpha>0 else False,
        "boundary_penalty": True
    }
    
    return diagnostics, bo_config


def strategy_based_on_data( weekno):  

    results = defaultdict(dict)
    for iter_num, cfg in funcConfig.FUNCTION_CONFIG.items():
        X, y = generate_data(iter_num, weekno)
        strategy = recommend_bo_strategy(X, y)
        print(f"+++++++++++++Recommended BO Strategy:+++++++++++++ function no{iter_num}")
        for k,v in strategy.items():
            print(f"{k}: {v}")


def estimate_noisy(X, Y, lvr_threshold=0.5, corr_threshold=0.85, n_neighbors=2):
    """
    Estimate if a function is noisy based on observed inputs and outputs.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input points.
    Y : ndarray, shape (n_samples,)
        Function outputs.
    lvr_threshold : float
        Threshold for local variation ratio. Above this, function considered noisy.
    corr_threshold : float
        Threshold for Spearman correlation between input distances and output differences. Below this → noisy.
    n_neighbors : int
        Number of neighbors for local variation computation.

    Returns
    -------
    is_noisy : bool
        True if function is estimated to be noisy.
    lvr : float
        Mean local variation ratio.
    spearman_rho : float
        Spearman correlation between input distances and output differences.
    """

    X = np.array(X)
    Y = np.array(Y)

    # ---- Local Variation Ratio (LVR) ----
    

    lvr = find_lvr(X, Y, n_neighbors)

    # ---- Spearman correlation between input distances and output differences ----
   
    rho = spearmanr_correlation(X, Y)

    # ---- Decision ----
    is_noisy = (lvr > lvr_threshold) or (rho < corr_threshold)

    return is_noisy, lvr, rho   


def update_function_config(function_data, FUNCTION_CONFIG):
    """
    Analyze all functions and update FUNCTION_CONFIG automatically.
    
    Parameters:
        function_data : dict
            Keys are function IDs, values are tuples (X, Y)
        FUNCTION_CONFIG : dict
            Original config dict to update
    
    Returns:
        updated_config : dict
            FUNCTION_CONFIG with updated BO parameters
        diagnostics_summary : dict
            Diagnostics for each function
    """
    updated_config = FUNCTION_CONFIG.copy()
    diagnostics_summary = {}

    for fid, (X, Y) in function_data.items():
      

        n_samples, dim = X.shape

        # 1. Noise estimation
        dists = squareform(pdist(X))
        np.fill_diagonal(dists, np.nan)
        nn_idx = np.nanargmin(dists, axis=1)
        Y_diff = np.abs(Y - Y[nn_idx])
        noise_estimate = np.median(Y_diff)
        Y_range = Y.max() - Y.min()

        # 2. Smoothness
        X_dist_flat = pdist(X)
        Y_dist_flat = pdist(Y.reshape(-1,1))
        smooth_corr, _ = spearmanr(X_dist_flat, Y_dist_flat)

        # 3. Dim relevance
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X, Y)
        feature_importances = rf.feature_importances_

        # 4. Multimodality
        top_k = max(int(0.1 * n_samples), 2)
        top_idx = np.argsort(Y)[-top_k:]
        top_X = X[top_idx]
        n_clusters = min(3, top_k)
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = kmeans.fit_predict(top_X)
        multimodal_score = len(np.unique(labels)) / n_clusters

        # -----------------
        # BO config recommendation
        # -----------------
        if noise_estimate > 0.01 * Y_range:
            kernel_type = "Matern"
            alpha = max(noise_estimate**2, 1e-6)
            nu = 1.5
        else:
            kernel_type = "RBF"
            alpha = 1e-6
            nu = 2.5

        if multimodal_score > 0.5 or smooth_corr < 0.5:
            acquisition = "UCB"
            beta = 2.0
        else:
            acquisition = "EI"
            beta = None

        normalize_y = True if Y_range > 1 else False
        length_scale = np.maximum(0.1, np.std(X, axis=0))
        n_restarts_optimizer = 10 if dim <= 5 else 20
        updated_config.setdefault(fid, {}).update({
            "kernel_type": kernel_type,
            "alpha": alpha,
            "nu": nu if kernel_type=="Matern" else None,
            "acquisition": acquisition,
            "beta": beta,
            "normalize_y": normalize_y,
            "length_scale": length_scale.tolist(),
            "length_scale_bounds": (1e-2, 1e2),
            "n_restarts_optimizer": n_restarts_optimizer,
            "add_white": True if alpha>0 else False,
        })

        diagnostics_summary[fid] = {
            "dim": dim,
            "noise_estimate": noise_estimate,
            "smooth_corr": smooth_corr,
            "feature_importances": feature_importances,
            "multimodal_score": multimodal_score
        }

    return updated_config, diagnostics_summary

         
def save_function_config_and_diagnostics(updated_config, diagnostics_summary, filename="function_analysis.txt"):
    """
    Save FUNCTION_CONFIG and diagnostics to a text file in readable JSON format.

    Parameters:
        updated_config : dict
            Updated FUNCTION_CONFIG dictionary
        diagnostics_summary : dict
            Dictionary with diagnostics for each function
        filename : str
            Filepath to save the text file
    """
    with open(filename, "w") as f:
        f.write("===== Updated FUNCTION_CONFIG =====\n")
        f.write(json.dumps(updated_config, indent=4))
        f.write("\n\n===== Diagnostics Summary =====\n")
        f.write(json.dumps(diagnostics_summary, indent=4))
    print(f"Configuration and diagnostics saved to {filename}")

def find_lvr(X,y, n_neighbors=2):
# Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)

    local_ratios = []
    for i in range(len(X)):
        j = indices[i, 1]  # nearest neighbor (skip itself)
        dx = np.linalg.norm(X[i] - X[j])
        dy = abs(y[i] - y[j])
        local_ratios.append(dy / (dx + 1e-8))

    lvr = np.mean(local_ratios)
    print("Estimated local variation ratio:", lvr)
    return lvr

def spearmanr_correlation (X, y):
    D_x = squareform(pdist(X))
    D_y = squareform(pdist(y[:, None]))
    # Y_dist_flat = pdist(Y.reshape(-1,1))
    rho, _ = spearmanr(D_x.flatten(), D_y.flatten())
    print("Spearman correlation (smoothness):", rho)
    return rho

# def analyze_density_clusters(X, y, eps=0.1, min_samples=5):
#     # DBSCAN (or other density-based) clustering in input space
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#     labels = clustering.labels_  # -1 = noise
#     unique = set(labels)
#     for lbl in unique:
#         mask = (labels == lbl)
#         print("Cluster", lbl, "size", mask.sum(), 
#               "y_mean", np.mean(y[mask]), "y_std", np.std(y[mask]))
#     return labels




def gp_cv_evaluate(X, y,
                   kernel=None,
                   n_splits=5,
                   random_state=None,
                   normalize_y=True,
                   n_restarts_optimizer=5):
    """
    Perform k‑fold cross validation for a GP regressor on data (X, y).
    Returns per‑fold metrics and average.
    """
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state
        )
        gp.fit(X_train, y_train)
        
        y_pred, y_std = gp.predict(X_test, return_std=True)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        inferred = gp.kernel_
        noise = None
        try:
            # If kernel has WhiteKernel component, extract noise_level
            noise = gp.kernel_.k2.noise_level
        except Exception:
            pass
        
        fold_results.append({
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "rmse": rmse,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_std": y_std,
            "inferred_kernel": inferred,
            "noise_level": noise
        })
    
  
    rmses = [fr["rmse"] for fr in fold_results]
    print("CV: mean RMSE = %.4f ± %.4f" % (np.mean(rmses), np.std(rmses)))
    return fold_results

    # diagnostics.py



def fitness_distance_correlation(X, y, x_best=None):
    """
    Compute correlation between distance-to-best and fitness (y).
    Returns correlation coefficient (Pearson).
    """
    if x_best is None:
        x_best = X[np.argmax(y)]
    dists = np.linalg.norm(X - x_best, axis=1)
    return np.corrcoef(dists, y)[0, 1] # correlation between distance and fitness

def density_clustering_analysis(X, y, eps=0.1, min_samples=5):
    """
    Run density-based clustering (DBSCAN) in input space.
    Print cluster stats (size, mean & std of y). Return labels.
    Also compute Davies–Bouldin index for clustering quality.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    clusters = set(labels)
    summary = {}
    for lbl in clusters:
        mask = (labels == lbl)
        summary[lbl] = {
            "count": mask.sum(),
            "y_mean": float(np.mean(y[mask])),
            "y_std": float(np.std(y[mask]))
        }
    # compute DBI only on non-noise clusters
    good = labels >= 0
    if good.sum() >= 2:
        try:
            dbi = davies_bouldin_score(X[good], labels[good])
        except Exception:
            dbi = None
    else:
        dbi = None
    return labels, summary, dbi

def gp_cv_uncertainty_analysis(X, y, kernel=None, n_splits=5, random_state=0):
    """
    Fit GP with cross-validation; for each fold compute prediction error and predicted uncertainty.
    Return mean squared error, mean predicted variance, and correlation between error and variance.
    """
    if kernel is None:
        kernel = 1.0 * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-6)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    errors = []
    variances = []
    for train_idx, test_idx in kf.split(X):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X[train_idx], y[train_idx])
        mu, sigma = gp.predict(X[test_idx], return_std=True)
        errs = (y[test_idx] - mu)**2
        errors.extend(errs.tolist())
        variances.extend((sigma**2).tolist())
    errors = np.array(errors)
    variances = np.array(variances)
    corr = np.corrcoef(errors, variances)[0, 1] if len(errors) > 1 else None
    return float(np.mean(errors)), float(np.mean(variances)), corr

def surrogate_peak_detection(X, y, gp, n_candidates=2000, top_k=5):
    """
    Generate a large pool of candidate inputs (random),
    predict mean and std with gp, and return top_k candidates with highest predicted mean.
    Useful to detect possible peaks / optima in surrogate landscape.
    """
    dim = X.shape[1]
    X_cand = np.random.rand(n_candidates, dim)
    mu, sigma = gp.predict(X_cand, return_std=True)
    idx = np.argsort(mu)[-top_k:]
    return X_cand[idx], mu[idx], sigma[idx]

def analyze_landscape(X, y, gp_kernel=None, cluster_eps=0.1, cluster_min_samples=5, cv_splits=5):
    """
    Run all diagnostics and print a summary.
    Returns dictionary of results.
    """
    results = {}
    # 1. Clustering / density analysis
    labels, cluster_info, dbi = density_clustering_analysis(X, y,
                                                            eps=cluster_eps,
                                                            min_samples=cluster_min_samples)
    results['clusters'] = cluster_info
    results['dbi'] = dbi

    # 2. Fitness-distance correlation
    fdc = fitness_distance_correlation(X, y)
    results['fitness_distance_correlation'] = fdc

    # 3. GP cross-validation uncertainty vs error
    mse, var_pred, err_var_corr = gp_cv_uncertainty_analysis(X, y,
                                                             kernel=gp_kernel, n_splits=cv_splits)
    results['cv_mse'] = mse
    results['cv_pred_variance_mean'] = var_pred
    results['cv_error_variance_corr'] = err_var_corr

    # 4. Surrogate peak detection (if gp_kernel provided)
    if gp_kernel is not None:
        gp_full = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True).fit(X, y)
        peaks, peak_mus, peak_sigmas = surrogate_peak_detection(X, y, gp_full)
        results['peak_candidates'] = {
            "X": peaks,
            "mu": peak_mus.tolist(),
            "sigma": peak_sigmas.tolist()
        }
    return results


def estimate_noise_and_min_prob(X, y, gp=None):
    """
    Estimate if a function is smooth or noisy based on observed data (X, y)
    Optionally uses a GP surrogate to refine the estimate.
    
    Returns:
        noise_level: estimated normalized noise (0 = very smooth, 1 = very noisy)
        suggested_min_prob: recommended min_prob for global candidate filtering
    """
    X = np.array(X)
    y = np.array(y)

    if len(y) < 2:
        # Not enough points to judge
        return 0.0, 0.5

    # ----- Method 1: look at residuals -----
    if gp is not None:
        # Predict with GP
        mu, sigma = gp.predict(X, return_std=True)
        residuals = np.abs(mu - y)
    else:
        # Use differences between neighboring points
        residuals = np.abs(np.diff(y))
        if residuals.size == 0:
            residuals = np.array([0.0])

    # Normalize residuals
    y_range = y.max() - y.min()
    if y_range == 0:
        y_range = 1e-8
    noise_level = np.mean(residuals) / y_range
    noise_level = np.clip(noise_level, 0.0, 1.0)

    # ----- Map noise level to min_prob -----
    # More noise → lower min_prob (allow exploration)
    # Less noise → higher min_prob (exploit promising points)
    suggested_min_prob = 0.7 - 0.4 * noise_level  # range ~0.3 → 0.7
    suggested_min_prob = np.clip(suggested_min_prob, 0.2, 0.7)

    return noise_level, suggested_min_prob







