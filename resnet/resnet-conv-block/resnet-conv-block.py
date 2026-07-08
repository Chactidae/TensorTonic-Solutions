import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    shortcut = np.matmul(x, Ws)
    out = np.matmul(x, W1)
    out = np.maximum(0, out)

    out = np.matmul(out, W2)

    out = out + shortcut
    
    out = np.maximum(0, out)

    return out
    
