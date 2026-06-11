import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    
    rows = len(A)
    cols = len(A[0])
    
    A_t = np.array([[A[i][j] for i in range(rows)] for j in range(cols)])
    
    return A_t
