from numba import njit
import numpy as np

@njit
def bmat_2d(m):
    out = np.hstack(m[0])
    for row in m[1:]:
        x = np.hstack(row)
        out = np.vstack((out, x))

    return out