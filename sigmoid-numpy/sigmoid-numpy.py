import numpy as np


def solve_sigm(x):
    return ((1) / (1 + np.exp(-x)))

def solves_sigm(array_x):
    res = []
    for x_i in array_x:
        res.append(((1) / (1 + np.exp(-x))))
    return res
    
def many_solve(array_x):
    if array_x.ndim <= 1:
        return solve_sigm(array_x)
    
    result = []
    for row in array_x:
        result.append(many_solve(np.array(row)))
    
    return np.array(result)
        

def sigmoid(x):
    array_x = np.array(x)
    return many_solve(array_x)