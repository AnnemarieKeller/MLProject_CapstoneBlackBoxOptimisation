import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import qmc


def random_candidates(n_points, dim):
    return np.random.rand(n_points, dim)

def grid_candidates(resolution, dim):
    axes = [np.linspace(0, 1, resolution)] * dim
    mesh = np.meshgrid(*axes)
    return np.column_stack([m.ravel() for m in mesh])

def generate_candidates(input_dim, n_candidates=500, method="random"):

    if method =="lhs":
        sampler = qmc.LatinHypercube(d=input_dim)
        return  sampler.random(n_candidates)
    if method == "random":
        return np.random.rand(n_candidates, input_dim)
    elif method == "grid":
        # only works for small dim <= 3
        lin = [np.linspace(0,1,int(np.ceil(n_candidates**(1/input_dim)))) for _ in range(input_dim)]
        mesh = np.meshgrid(*lin)
        X = np.column_stack([m.ravel() for m in mesh])
        if X.shape[0] > n_candidates:
            X = X[:n_candidates] 
        return X
    elif method == "sobol":
        try:
            sampler = qmc.Sobol(d=input_dim, scramble=True)
            m = int(np.ceil(np.log2(n_candidates)))
            X= sampler.random_base2(m)[:n_candidates]
            return X
        except ImportError:
            print("Sobol requires scipy >= 1.7. Falling back to random.")
            return np.random.rand(n_candidates, input_dim)
    else:
        raise ValueError(f"Unknown candidate generation method: {method}")
    
def determine_candidate_generation_method(input_dim):

    # ---- Convert input_dim to integer safely ----
    if isinstance(input_dim, tuple):
        # e.g. (20, 8) -> 8
        if len(input_dim) == 1:
            input_dim = input_dim[0]
        else:
            input_dim = input_dim[-1]

    elif isinstance(input_dim, (list, np.ndarray)):
        if len(input_dim) == 1:
            input_dim = int(input_dim[0])
        else:
            raise ValueError(f"Ambiguous input_dim: {input_dim}")

    elif isinstance(input_dim, int):
        pass  # already good

    else:
        raise TypeError(f"input_dim must be int/tuple/list/array, got {type(input_dim)}")


    # ---- Validate ----
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive integer, got {input_dim}")

    # ---- Select method ----
    if input_dim <= 3:
       return "random"
    elif 4 <= input_dim <= 6:
        return "lhs"
    elif input_dim >= 7:
        return "sobol"
    else:
        raise ValueError(f"Unhandled input_dim: {input_dim}")

    method = ''
    if input_dim <= 3:
       return "random"
    elif 4 <= input_dim <= 6:
        return "lhs"      # Latin Hypercube Sampling
    elif input_dim >= 7:
        return "sobol" 
    else:
        raise ValueError(f"input_dim not mapped to candiate method generation: {input_dim}")
   
   
   
   
   
    # alternative sobol generation method
    # elif method == "sobol":
    #     # Hybrid approach for high dimension
    #     n_sobol = int(n_candidates * 0.6)
    #     n_local = n_candidates - n_sobol

    #     # Sobol points (must be base2)
    #     m = int(np.ceil(np.log2(n_sobol)))
    #     sampler = qmc.Sobol(d=input_dim, scramble=True)
    #     X_sobol = sampler.random_base2(m=m)
    #     if X_sobol.shape[0] > n_sobol:
    #         X_sobol = X_sobol[:n_sobol]

    #     # Local perturbations
    #     if initial_points is not None and len(initial_points) > 0:
    #         best_idx = np.random.choice(len(initial_points), n_local)
    #         best_points = initial_points[best_idx]
    #         sigma = 0.05
    #         X_local = best_points + np.random.normal(0, sigma, size=(n_local, input_dim))
    #         X_local = np.clip(X_local, 0, 1)
    #     else:
    #         X_local = np.random.rand(n_local, input_dim)

    #     return np.vstack([X_sobol, X_local])