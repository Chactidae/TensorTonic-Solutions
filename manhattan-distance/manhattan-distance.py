import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x_size = len(x)
    y_size = len(y)
    if x_size == y_size:
        res = np.sum(np.asarray([[abs(x[i] - y[i]) for i in range(x_size)]], dtype=float))
        return res
    else:
        raise ValueError("out of size!")