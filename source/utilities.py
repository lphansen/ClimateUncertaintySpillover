"""
Functions that facilitate computation.

"""
import numpy as np
from numba import njit


@njit
def find_nearest_value(array, value):
    r"""Return the index of the element in ``array`` that is closest to ``value``.
    """
    loc = np.abs(array - value).argmin()
    return loc


@njit
def compute_derivatives(data, order, dlt):
    r"""Compute the derivatives of a function.
    
    Parameters
    ----------
    data : (N, ) ndarrays
        The function whose derivatives to be computed.
    order : int::
        1: first order derivative,
        
        2: second order derivative.
    dlt : float
        Grid step.
        
    Returns
    -------
    res : (N, ) ndarrays
        Deriative array. 
    
    Notes
    -----
    First order derivatives are computed one-sidedly.
    Specifically,
    
    .. math::
        \frac{d f(x_0)}{dx} \approx & \frac{f(x_1) - f(x_0)}{\Delta x} \\
        \frac{d f(x_i)}{dx} \approx & \frac{f(x_i) - f(x_{i-1})}{\Delta x}, i = 1,2,\dots,N
    """
    res = np.zeros_like(data)
    if order == 1:
        res[1:] = (1 / dlt) * (data[1:] - data[:-1])
        res[0] = (1 / dlt) * (data[1] - data[0])      
    elif order == 2:
        res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
        res[-1] = (1 / dlt ** 2) * (data[-1] + data[-3] - 2 * data[-2])
        res[0] = (1 / dlt ** 2) * (data[2] + data[0] - 2 * data[1])
    return res
