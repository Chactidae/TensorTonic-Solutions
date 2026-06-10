import numpy as np


def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here

    x = np.array(x)
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1).")

    rng

    mask = (rng.random(x.shape) > p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask