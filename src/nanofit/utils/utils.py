import numpy as np

def add_dummy_feature(X: np.array):
    dummy = np.c_[np.ones(len(X)), X]
    return dummy