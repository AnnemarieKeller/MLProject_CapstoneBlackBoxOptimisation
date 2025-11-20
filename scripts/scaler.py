from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
def scale_data(data, scaler=None, scaler_type='minmax'):
    """
    Scale data using a given scaler or create a new one.

    Parameters:
        data : array-like, shape (n_samples, n_features)
        scaler : fitted scaler (optional)
        scaler_type : 'minmax' or 'standard' (used only if scaler is None)

    Returns:
        scaled_data : np.array, scaled version of data
        scaler : the fitted scaler (can be reused)
    """
    data = np.array(data)

    # Create a new scaler if none provided
    if scaler is None:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        scaled_data = scaler.fit_transform(data)
    else:
        # Reuse the provided scaler
        scaled_data = scaler.transform(data)

    return scaled_data, scaler