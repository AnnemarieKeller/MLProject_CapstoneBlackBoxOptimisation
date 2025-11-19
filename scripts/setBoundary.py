import numpy as np

def apply_boundary_penalty(X_candidates, softness=0.15):
  """
    Apply a penalty for points close to the edges of [0,1]^d.
    Returns values in [0,1] as penalty.
    """
    X_candidates = np.atleast_2d(X_candidates)  # ensure 2D: n_samples x n_features
    dist = np.minimum(X_candidates, 1 - X_candidates)
    min_dist = np.min(dist, axis=1)
    return np.clip(min_d / softness, 0, 1)




