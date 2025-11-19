import numpy as np

def apply_boundary_penalty(X_train, softness=0.15):
    dist = np.minimum(X, 1 - X_train)
    d = np.min(dist, axis=1)
    return np.clip(d / softness, 0, 1)
