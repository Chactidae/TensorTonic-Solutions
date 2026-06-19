import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False


def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
 
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    x_feat = x.shape[0] if x.ndim == 1 else x.shape[1]
    h_feat = h_prev.shape[0] if h_prev.ndim == 1 else h_prev.shape[1]
    
    x2d, x_was_1d = _as2d(x, x_feat)
    h2d, h_was_1d = _as2d(h_prev, h_feat)

    # Parameters
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    # Gates
    z = _sigmoid(x @ Wz + h_prev @ Uz + bz)
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)
    h_next = (1 - z) * h_prev + z * h_tilde

    if x_was_1d and h_was_1d:
        return h_next.reshape(-1)
    return h_next
    