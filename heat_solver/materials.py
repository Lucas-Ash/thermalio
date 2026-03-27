import numpy as np


def process_alpha(alpha, x, y, temperature=None):
    """
    Evaluates alpha (scalar, array, callable) at coordinates (x, y).
    Returns an array of shape (..., 2, 2) representing the tensor field,
    where ... is the shape of x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = x.shape
    
    if callable(alpha):
        if temperature is not None:
            try:
                a_val = alpha(x, y, temperature)
            except TypeError:
                a_val = alpha(x, y)
        else:
            a_val = alpha(x, y)
    else:
        a_val = alpha
        
    a_val = np.asarray(a_val, dtype=float)
    
    # Broadcast to expected shape
    if a_val.ndim == 0 or (a_val.ndim == 1 and a_val.size == 1):
        # Scalar
        res = np.zeros(shape + (2, 2))
        res[..., 0, 0] = a_val
        res[..., 1, 1] = a_val
        return res
    elif a_val.shape == (2, 2):
        # Constant tensor
        res = np.zeros(shape + (2, 2))
        res[...] = a_val
        return res
    elif a_val.shape == shape:
        # Spatially varying scalar
        res = np.zeros(shape + (2, 2))
        res[..., 0, 0] = a_val
        res[..., 1, 1] = a_val
        return res
    elif a_val.shape == shape + (2, 2):
        # Spatially varying tensor
        return a_val
    else:
        raise ValueError(f"Invalid alpha shape. Expected (), (2,2), {shape}, or {shape + (2,2)}, got {a_val.shape}")
