import numpy as np


def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    x_hat = (x - mean) / np.sqrt(var + eps)

    return gamma * x_hat + beta

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    shortcut = x

    if mode == "post":
        out = np.matmul(x, W1)

        out = batch_norm(out, gamma1, beta1)

        out = np.maximum(0, out)

        out = np.matmul(out, W2)

        out = batch_norm(out, gamma2, beta2)

        out = out + shortcut

        out = np.maximum(0, out)
        out = np.round(out, decimals=4)

        out = {
            "output": out,
            "mode": mode
        }
        
        return out
    elif mode == "pre":
        out = batch_norm(x, gamma1, beta1)
        out = np.maximum(0, out)
        out = np.matmul(out, W1)

        out = batch_norm(out, gamma2, beta2)

        out = np.maximum(0, out)

        out = np.matmul(out, W2)

        out = out  +shortcut

        out = np.round(out, decimals=4)

        out = {
            "output": out,
            "mode": mode
        }

        return out
    else:
        raise ValueError("wrong mode")
        
