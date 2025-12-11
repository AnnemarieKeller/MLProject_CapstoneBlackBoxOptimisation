import numpy as np 
from scipy.stats import normaltest

def get_gp_health(gp, X_train_full, y_train_full):
# --- A) Kernel conditioning ---
    K = gp.kernel_(X_train_full)
    cond = np.linalg.cond(K)
    cond_norm = np.clip(1 - np.log10(cond) / 12, 0, 1)  
# cond = 1e12 → cond_norm=0 (bad)
# cond = 1e4 → cond_norm≈0.67
# cond = 1e2 → cond_norm≈0.83

# --- B) Predictive uncertainty ---
    mu_t, sigma_t = gp.predict(X_train_full, return_std=True)
    avg_sigma = float(np.mean(sigma_t))
    sigma_norm = np.exp(-avg_sigma)  
# high uncertainty → low score

# --- C) GP log-marginal likelihood ---
    loglike = gp.log_marginal_likelihood()
    loglike_norm = 1 / (1 + np.exp(-loglike / 100))  
# smooth normalization

# --- Final GP health score (0=bad, 1=excellent) ---
    gp_health = 0.4 * cond_norm + 0.3 * sigma_norm + 0.3 * loglike_norm
    return gp_health, loglike, avg_sigma, cond


def get_gp_health_score(gp, X_train, y_train):
    """
    Returns a single numeric GP health score [0,1], where 1 = perfectly healthy.
    Combines anomalies and average sigma.
    """
    anomalies = detect_gp_anomalies(gp, X_train, y_train)

    if not anomalies:
        anomaly_penalty = 0.0
    else:
        # Severity-weighted penalty
        anomaly_penalty = np.mean([a["severity"] for a in anomalies])

    mu, sigma = gp.predict(X_train, return_std=True)
    avg_sigma = float(np.mean(sigma))
     # ----------------------
    # Kernel condition
    # ----------------------
    try:
        K = gp.kernel_(X_train)
        cond_val = float(np.linalg.cond(K))
    except Exception:
        cond_val = float('nan')  # fallback if something fails

    # ----------------------
    # GP health score
    # ----------------------
    health_score = np.exp(-anomaly_penalty) * np.exp(-avg_sigma)  # 0~1 scale

    return float(health_score), cond_val, avg_sigma, anomalies


def gp_health_gate(gp, X, y, 
                   min_health=0.25,
                   max_condition=1e12,
                   logger=None):
    """
    Wrapper that uses your existing get_gp_health() and decides whether
    the BO loop should proceed with this GP or fallback.
    
    Returns
    -------
    proceed : bool
    gp_health : float
    cond : float
    avg_sigma : float
    loglike : float
    """

    gp_health, cond, avg_sigma, loglike = get_gp_health(gp, X, y)

    if logger:
        logger.info(
            f"[GP STATE] health={gp_health:.3f} | cond={cond:.2e} | "
            f"avg_sigma={avg_sigma:.5f} | loglike={loglike:.3f}"
        )

    # ---- HEALTH CHECK THRESHOLDS ----
    if gp_health < min_health:
        if logger:
            logger.warning(f" GP health too low ({gp_health:.3f}) — unsafe to optimize.")
        return False, gp_health, cond, avg_sigma, loglike

    if cond > max_condition:
        if logger:
            logger.warning(f" GP kernel ill-conditioned (cond={cond:.2e}).")
        return False, gp_health, cond, avg_sigma, loglike

    return True, gp_health, cond, avg_sigma, loglike


def gp_residuals(gp, X_train, y_train):
    mu, sigma = gp.predict(X_train, return_std=True)
    residuals = y_train - mu
    return residuals, sigma

def check_calibration(residuals, sigma):
    # Standardized residuals
    z = residuals / (sigma + 1e-8)

    # Well-calibrated GP → z ~ N(0,1)
    stat, p = normaltest(z)   # low p ⇒ bad calibration
    calibration_score = np.clip(p, 0, 1)
    return calibration_score, p

def check_residual_patterns(residuals):
    # Ideally residuals have no pattern
    # Using autocorrelation as a simple pattern detector
    if len(residuals) < 3:
        return 1.0

    ac = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    # High correlation ⇒ bad
    return float(1 - abs(ac))

def check_hyperparameter_stability(kernel):
    params = kernel.get_params()
    score = 1.0

    # Example checks
    if "length_scale" in params:
        ls = np.array(params["length_scale"])
        if np.any(ls < 1e-4) or np.any(ls > 1e4):
            score *= 0.4  # penalize extreme values

    if "alpha" in params:  # RQ kernel
        a = params["alpha"]
        if a < 0.1 or a > 10:
            score *= 0.4

    return float(score)

def gp_health_score(gp, X_train, y_train):
    residuals, sigma = gp_residuals(gp, X_train, y_train)

    calibration, p_cal = check_calibration(residuals, sigma)
    residual_pattern = check_residual_patterns(residuals)
    stability = check_hyperparameter_stability(gp.kernel_)

    # Weighted score
    score = (
        0.5 * calibration +
        0.3 * stability +
        0.2 * residual_pattern
    )

    return {
        "score": float(score),
        "calibration": float(calibration),
        "hyper_stability": float(stability),
        "residual_pattern": float(residual_pattern),
        "p_calibration": float(p_cal),
    }

def detect_gp_anomalies(gp, X, y):
    """
    Automatically detect common GP failures (lengthscale collapse,
    noise collapse, singular kernel matrix, etc.)

    Returns a list of anomaly dictionaries:
    [
      {"type": "lengthscale_collapse", "severity": 0.9, "details": {...}},
      {"type": "kernel_singular", "severity": 1.0, "details": {...}}
    ]
    """
    anomalies = []

    # ------------------------
    # Extract kernel parameters safely
    # ------------------------
    kernel = gp.kernel_
    params = kernel.get_params()

    # Extract lengthscales
    lengthscales = []
    for key, value in params.items():
        if "length_scale" in key:
            if isinstance(value, (float, int)):
                lengthscales = np.array([value])
            elif isinstance(value, (tuple, list, np.ndarray)):
                # value could be (param, bounds)
                lengthscales = np.array(value[:1])  # take first element
    if len(lengthscales) == 0:
        lengthscales = np.array([1.0])  # default fallback

    # Extract noise
    noise = None
    for key, value in params.items():
        if "noise_level" in key:
            if isinstance(value, (float, int)):
                noise = float(value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                noise = float(value[0])  # first element = actual value

    # ------------------------
    # Compute health stats
    # ------------------------
    try:
        K = gp.kernel_(X)
        cond = np.linalg.cond(K)
    except np.linalg.LinAlgError:
        cond = np.inf

    mu, sigma = gp.predict(X, return_std=True)
    avg_sigma = float(np.mean(sigma))

    residuals = np.abs(mu - y.flatten())
    corr = np.corrcoef(residuals, y.flatten())[0, 1] if residuals.std() > 0 else 0

    # ------------------------
    # Lengthscale anomalies
    # ------------------------
    if np.any(lengthscales < 1e-3):
        anomalies.append({
            "type": "lengthscale_collapse",
            "severity": 0.9,
            "details": {"lengthscales": lengthscales.tolist()}
        })

    if np.any(lengthscales > 100):
        anomalies.append({
            "type": "lengthscale_explosion",
            "severity": 0.7,
            "details": {"lengthscales": lengthscales.tolist()}
        })

    # ------------------------
    # Noise anomalies
    # ------------------------
    if noise is not None:
        if noise < 1e-7:
            anomalies.append({
                "type": "noise_collapse",
                "severity": 0.8,
                "details": {"noise_level": noise}
            })

    # ------------------------
    # Kernel singularity / numerical issues
    # ------------------------
    if cond > 1e12:
        anomalies.append({
            "type": "kernel_singular",
            "severity": 1.0,
            "details": {"cond": cond}
        })

    # ------------------------
    # Calibration anomalies
    # ------------------------
    if avg_sigma < 1e-5:
        anomalies.append({
            "type": "overconfidence",
            "severity": 0.7,
            "details": {"avg_sigma": avg_sigma}
        })

    if avg_sigma > 1.0:
        anomalies.append({
            "type": "too_uncertain",
            "severity": 0.5,
            "details": {"avg_sigma": avg_sigma}
        })

    # ------------------------
    # Heteroscedasticity
    # ------------------------
    if abs(corr) > 0.6:
        anomalies.append({
            "type": "heteroscedasticity",
            "severity": 0.6,
            "details": {"corr_residuals": corr}
        })

    return anomalies


