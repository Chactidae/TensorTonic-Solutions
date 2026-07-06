import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_vector = np.arange(seq_length).reshape(-1, 1)

    enconing_matrix = np.zeros((seq_length, d_model))
    
    div_term = np.exp(np.arange(0, d_model,2) * (-np.log(10000.0)/d_model))

    # Sin
    enconing_matrix[:, 0::2] = np.sin(pos_vector * div_term)

    enconing_matrix[:, 1::2] = np.cos(pos_vector * div_term)

    return enconing_matrix
    