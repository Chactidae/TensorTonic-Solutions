import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
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