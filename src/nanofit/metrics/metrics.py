import numpy as np

def mse(y_true, y_preds):
    y_true = np.ravel(y_true)
    y_preds = np.ravel(y_preds)
    return np.mean((y_true - y_preds) ** 2)