import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    
    # Write code here

    s = beta * s + (1 - beta) * (g ** 2)
    w = w - lr * g / (np.sqrt(s) + eps)

    return tuple(np.round(w, 6)), tuple(np.round(s, 6))