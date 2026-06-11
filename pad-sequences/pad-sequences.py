import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = 0
        for seq in seqs:
            if len(seq) > max_len:
                max_len = len(seq)

    for seq in seqs:
        while len(seq) < max_len:
            seq.append(pad_value)

        while len(seq) > max_len:
            seq.pop()
            
    return seqs
            