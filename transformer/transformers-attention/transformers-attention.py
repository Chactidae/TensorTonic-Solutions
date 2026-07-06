import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))

    d_k = K.shape[-1]


    attention_scores = attention_scores / math.sqrt(d_k)

    W = torch.softmax(attention_scores, dim=-1)

    output = torch.matmul(W, V)

    return output
    

    