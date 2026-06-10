import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m_new = m
    v_new = v
    beta = [beta1, beta2]
    param_new = param
    for i in range(len(param)):
            
        m_new[i] = beta[0] * m_new[i] + (1 - beta[0]) * grad[i]
        v_new[i] = beta[1] * v_new[i] + (1 - beta[1]) * (grad[i])**2
    
        m_1 = m_new[i] / (1 - beta[0]**t)
        v_1 = v_new[i] / (1 - beta[1]**t)
    
        param_new[i] = param_new[i] - lr * (m_1 / (np.sqrt(v_1) + eps))
    
    return (param_new, m_new, v_new)
    