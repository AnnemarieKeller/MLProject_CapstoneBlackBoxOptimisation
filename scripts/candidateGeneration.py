import numpy as np

def random_candidates(n_points, dim):
    return np.random.rand(n_points, dim)

def grid_candidates(resolution, dim):
    axes = [np.linspace(0, 1, resolution)] * dim
    mesh = np.meshgrid(*axes)
    return np.column_stack([m.ravel() for m in mesh])
