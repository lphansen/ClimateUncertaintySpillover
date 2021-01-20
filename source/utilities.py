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