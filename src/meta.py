from functools import partial
import scipy.stats as stats
import numpy as np
from numpy.typing import NDArray


def alpha_zero(alpha:float, k:int):
    return alpha/k

def _q(n,k,alpha):
    K = np.arange(1,k+1)
    return stats.beta.ppf(1-alpha, n-K+1, K)

def _bins(q:NDArray):
    return np.concat([[np.inf], q, [-np.inf]])

def _assignments(data, bins):
    return np.digitize(data.T, bins)

def assignments(data, bins):
    return np.apply_along_axis(partial(_assignments, bins=bins), 0, data)

def _vals(data, k):
    for j in range(1,k+1):
        if np.count_nonzero(data <= j) >= j :
            return k
    return 0

def get_vals(data, k):
    return np.apply_along_axis(partial(_vals, k=k), 1, data)

def get_alarm(vals):
    return np.where(vals > 0, 1, 0)

def meta(data:NDArray, n:int, k:int, alpha:int) -> tuple[NDArray, NDArray]:
    q = _q(n,k,alpha_zero(alpha,k))
    bins = _bins(q)
    assignments = _assignments(data, bins)
    vals = get_vals(assignments, k)
    alarms = get_alarm(vals)
    return assignments, alarms    