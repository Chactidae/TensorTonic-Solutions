import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    i = np.arange((d_model) // 2 + (d_model % 2 == 1))

    freq = base ** (2*i / d_model)
    pos = np.arange(seq_len)[:, np.newaxis]
    angles = pos / freq

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angles)
    
    if d_model // 2 > 0:
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe