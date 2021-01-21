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

def compute_h_hat(e, gamma_e, xi_e):
    """
    compute h hat
    
    """
    sigma_n = 1.2
    median = 1.75/1000
    eta = .032
    gamma = gamma_e
    xi = xi_e
    h_hat = e*median*gamma*sigma_n/xi
    h_hat = h_hat*median*1000*sigma_n/(1-eta)
    
    return h_hat