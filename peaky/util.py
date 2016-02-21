import numpy as np

def sort_fst_col(X):
    i = np.argsort(X[:, 0])
    return X[i]
