import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x_mean = np.mean(x, axis=-1, keepdims=True)

    disp = np.var(x, axis=-1, keepdims=True)

    layer_norm = (gamma * ((x - x_mean) / np.sqrt(disp + eps))) + beta

    return layer_norm

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    project_Q = np.dot(Q, W_q) 
    project_k = np.dot(K, W_k) 
    project_V = np.dot(V, W_v)

    batch, seq, d_model = project_Q.shape
    d_k = d_model // num_heads

    project_Q = project_Q.reshape(batch, seq, num_heads, d_k)
    project_k = project_k.reshape(batch, seq, num_heads, d_k)
    project_V = project_V.reshape(batch, seq, num_heads, d_k)

    project_Q = project_Q.transpose(0, 2, 1, 3)
    project_k = project_k.transpose(0, 2, 1, 3)
    project_V = project_V.transpose(0, 2, 1, 3)

    attention_scores = np.matmul(project_Q, project_k.transpose(0, 1, 3, 2))/ np.sqrt(d_k)
    W = softmax(attention_scores)

    head_output = np.matmul(W, project_V)

    head_output = head_output.transpose(0, 2, 1, 3)
    head_output = head_output.reshape(batch, seq, d_model)
    output = np.matmul(head_output, W_o)

    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    assert x is not None
    project = np.dot(x, W1) + b1

    relu_out = np.maximum(0, project)
    
    relu_out = np.dot(relu_out, W2) + b2

    return relu_out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn = multi_head_attention(
        x, x, x,
        W_q, W_k, W_v,
        W_o,
        num_heads
    )

    # Residual Connection
    x = x + attn

    #Layer norm
    x = layer_norm(x, gamma1, beta1)

    ffn = feed_forward(x, W1, b1, W2, b2)

    # Second residual connection
    x = x + ffn

    # Second layerNorm
    x = layer_norm(x, gamma2, beta2)

    return x
    