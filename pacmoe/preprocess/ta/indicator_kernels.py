"""Loop kernels of two indicators (OBV, CCI mean absolute deviation). numba is optional; without it the
functions run as plain Python (the reported features were computed this way).
"""
from __future__ import annotations
import numpy as np
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):

        def _decorator(fn):
            return fn
        return _decorator if args and callable(args[0]) else _decorator

@njit(cache=True)
def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    obv = np.empty(n, dtype=np.float64)
    obv[0] = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

@njit(cache=True)
def compute_cci_mad(x: np.ndarray) -> float:
    n = len(x)
    mean_x = 0.0
    for v in x:
        mean_x += v
    mean_x /= n
    mad = 0.0
    for v in x:
        diff = v - mean_x
        if diff < 0.0:
            diff = -diff
        mad += diff
    return mad / n
