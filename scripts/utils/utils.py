
import numpy as np 
def set_alpha(y_train):
    """Automatically determine alpha from y scale
    """
# Automatically determine alpha from y scale
    y_range = np.max(y_train) - np.min(y_train)
    alpha = max(1e-6, 1e-4 * y_range)  # small fraction of y range
