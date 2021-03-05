"""
module for cumputing derivatives
includes:
- compute_dphidr
- compted_dphidz

"""

import numpy as np
from global_parameters import *
# define tau
tau = MEDIAN*GAMMA_BASE
# start of fuction def
# compute dphidr
def compute_dphidr(phi, r, z = np.linspace(1e-5, 2, 20)):
    """computed dphi/dr and return corresponding emission
    Parameters
    ----------
    phi: sorted phi grid
    r: sort r grid
    z: original z grid used in computinng solution
        (Default np.linspace(1e-5, 2, 20))
    Returns
    -------
    r_new, phi_new, ems
    """
    dphi = (phi[1:] - phi[:-1])/(r[1:] - r[:-1])
    r_new = (r[1:] + r[:-1])/2
    phi_new = (phi[1:] + phi[:-1])/2
    ems = DELTA*ETA/(tau*z + dphi)
    return r_new, phi_new, ems

# compute dphidz
def compute_dphidz(phi, z=np.linspace(1e-5, 2, 20)):
    """compute dphi/dz, and modify the corresponding z grid
    Parameters
    ----------
    phi: array
        orgininal sorted phi grid
        IMPORTANT: do not use the one returned by computed_dphidr
    z: array
        original z grid used in computing solution
    Returns
    -------
    z_new: new z grid after partial derivatives are computed
        NOTE: dimension reduce by 1 b/c of modification
    dphi_dz: partial derivatives
    """
    dphi_dz = (phi[:, 1:] - phi[:, :-1])/(z[1:] - z[:-1])
    z_new = (z[1:] + z[:-1])/2
    return z_new, dphi_dz

def derivatives_2d(data, dim, order, step, onesided=True):
    """compute derivative matrix for a fuction space

    :data: TODO
    :dim: TODO
    :order: TODO
    :step: TODO
    :returns: TODO

    """
    num_x, num_y = data.shape
    derivative_spec = (dim, order)
    return {
        (0,1): deriv01(data, dim, order, step),
        (0,2): deriv02(data, dim, order, step),
        (1,1): deriv11(data, dim, order, step),
        (1,2): deriv12(data, dim, order, step),
    }.get(derivative_spec, "error")    # 9 is default if x not found


def deriv01(data, dim, order, step, onesided):
    num_x, num_y = data.shape
    ddatadx = np.zeros(data.shape)
    for i in range(num_x):
        if i == 0:
            ddatadx[i] = (ddatadx[i+1] - ddatadx[i])/step
        elif i == num_x -1:
            if one
            ddatadx[i] = (ddatadx[i] - data[i-1])/step
        
