# -*- coding: utf-8 -*-
"""
Utility functions to facilitate computation.

"""
import numpy as np
from numba import njit


# @njit
def find_nearest_value(array, value):
    """
    Find nearest value for

    """
    loc = np.abs(array - value).argmin()
    return loc

def compute_h_hat(emission, γ, ξ, arg = (1.75/1000, 1.2)):
    """
    compute h hat

    Parameters
    ----------
    emission: array
        simulated emission sequence
    γ: float
        damage model parameter
    ξ: float
        model misspecification parameter;
        smaller the value, greater the concern for model misspecification

    Returns
    -------
    h_hat, or drift distortion
    """
    median, σ_n = arg
    gamma = γ
    xi = ξ
    h_hat = emission*median*gamma*σ_n/xi
    h_hat = h_hat*median*1000*σ_n
    return h_hat

def compute_std(emission, time, arg = (1.75/1000, 1.2)):
    """
    compute standard deviation in table 1

    Parameters
    ----------
    emission: array
        simulated emission path
    time: int
        time span during which the standard deviation is considered

    Returns
    -------
    implied standard deviation
    """
    median, σ_n = arg
    emission_selected = emission[:time]
    std = np.sqrt(np.sum(emission_selected**2))/emission_selected.sum()*σ_n*median*1000
    return std
