from numba import vectorize, float64
from math import sqrt


@vectorize([float64(float64)])
def v_pinv_square(val):
    if abs(val) < 1e-5:
        return 0.0
    else:
        return 1.0 / val ** 2


@vectorize([float64(float64)])
def v_pinv(val):
    if abs(val) < 1e-5:
        return 0.0
    else:
        return 1.0 / val


@vectorize([float64(float64)])
def v_pinv_sqrt(val):
    if val < 1e-5:
        return 0.0
    else:
        return 1.0 / sqrt(val)
