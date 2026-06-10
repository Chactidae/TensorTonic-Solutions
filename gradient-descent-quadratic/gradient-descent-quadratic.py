def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code her

    
    x1 = 0
    min_func = -(b / (2*a))
    for i in range(steps):
        x1 = x0 - lr*(2*a*x0 + b)
        x0 = x1

    return x1