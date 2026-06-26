import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    print(x.ndim)
    if x.ndim == 2:
        # Write code here
        m_b = np.mean(x, axis=0)
        
        disp_B = np.mean((x - m_b)**2, axis=0)  
    
        # Normalize
        norm_x = (x - m_b) / np.sqrt(disp_B + eps)   
    
        # Scale and shift
        y = gamma * norm_x + beta

    if x.ndim == 4:
        m_b = np.mean(x, axis=(0, 2, 3), keepdims=True)
        disp_B = np.mean((x - m_b)**2, axis=(0,2,3), keepdims=True) 
        # Normalize
        norm_x = (x - m_b) / np.sqrt(disp_B + eps)    
        # Scale and shift
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        y = gamma * norm_x + beta

    return y