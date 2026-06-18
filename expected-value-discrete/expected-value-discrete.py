import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    
    p = np.asarray(p, dtype=float)
    if np.sum(p) == 1.0:
        x = np.asarray(x, dtype=float)
        md = np.sum(x * p)
        return md
    else:
        raise ValueError("sum(p) != 1")
