import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b =0.0
    
    for _ in range(n_iters):
        z = X @ w + b
        y_pred = 1.0 / (1.0 + np.exp(-z))
        error = y_pred - y
        dw = (1.0/n) * (X.T @ error)
        db = (1.0/n) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db

    return w.tolist(), float(b)
    
    pass
