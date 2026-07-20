import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    P = patch_size
    
    assert H % P == 0
    assert W % P == 0

    patch_dim = P * P * C

    N = (H // P) * (W // P)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    patches = image.reshape(B, (H // P), P,  (W // P), P, C)

    # (B, H/P, W/P, P, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    # (B, N, embed_dim)
    patches = patches.reshape(B, N, patch_dim)
   
    embeddings = patches @ W_proj

    return embeddings