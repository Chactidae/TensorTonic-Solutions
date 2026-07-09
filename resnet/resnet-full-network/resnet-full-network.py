import numpy as np

def identity_block(x, W1, W2):
    identity = x

    out = np.matmul(x, W1)
    out = np.maximum(0, out)

    out = np.matmul(out, W2)

    out = out + identity

    out = np.maximum(0, out)

    return out

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

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.array(x)
    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)
    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    fc = np.array(fc)
    out = np.matmul(x, conv1)
    out = np.maximum(0, out)
    print("after conv1:", out.shape)

    out = identity_block(out, W1_b1, W2_b1)
    print("after block1:", out.shape)

    out = conv_block(out, W1_b2, W2_b2, Ws_b2)
    print("after block2:", out.shape)

    out = np.matmul(out, fc)
    print("fc:", fc.shape)
    
    return out
    
    
