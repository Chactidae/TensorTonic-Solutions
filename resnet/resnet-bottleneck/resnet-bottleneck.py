import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """

    shortcut = np.matmul(x, Ws)
    y = np.matmul(x, W1)

    layer1 = np.maximum(0, y)

    y2 = np.matmul(layer1, W2)

    layer2 = np.maximum(0, y2)

    y3 = np.matmul(layer2, W3)

    y3_shr = y3 + shortcut
    
    output = np.maximum(0, y3_shr)

    return output
    
