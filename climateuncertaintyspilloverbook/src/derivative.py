# -*- coding: utf-8 -*-
"""
module for cumputing derivatives
includes:
- compute_dphidr
- compted_dphidz

"""
import numpy as np
import src.global_parameters as gp
from numba import njit
# define tau
tau = gp.MEDIAN*gp.GAMMA_BASE


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
        (0,1): deriv01(data, step, onesided),
        (0,2): deriv02(data, step, onesided),
        (1,1): deriv11(data, step, onesided),
        (1,2): deriv12(data, step, onesided),
    }.get(derivative_spec, "error") 


# +
@njit
def deriv01(data, step, onesided):
    num_x, _ = data.shape
    ddatadx = np.zeros(data.shape)
    for i in range(num_x):
        if i == 0:
            ddatadx[i] = (data[i+1, :] - data[i, :])/step
        elif i == num_x -1:
            ddatadx[i] = (data[i,:] - data[i-1,:])/step
        else:
            if onesided == True:
                ddatadx[i] = (data[i, :] - data[i-1,:])/step
            else:
                ddatadx[i] = (data[i+1, :] - data[i-1,:])/(2*step)
    return ddatadx

@njit
def deriv02(data, step, onesided):
    num_x, _ = data.shape
    ddatadxx = np.zeros(data.shape)
    for i in range(num_x):
        if i == 0:
            ddatadxx[i] = (data[i+2,:] -2*data[i+1,:] + data[i,:])/(step**2)
        elif i == num_x -1:
            ddatadxx[i] = (data[i,:] - 2*data[i-1,:] + data[i-2,:])/(step**2)
        else:
            ddatadxx[i] = (data[i+1,:] - 2*data[i,:] + data[i-1,:])/(step**2)
    return ddatadxx
@njit
def deriv11(data, step, onesided):
    _, num_y = data.shape
    ddatady = np.zeros(data.shape)
    for j in range(num_y):
        if j == 0:
            ddatady[:,j] = (data[:,j+1] - data[:,j])/step
        elif j == num_y -1:
            ddatady[:,j] = (data[:,j] - data[:,j-1])/step
        else:
            if onesided == True:
                ddatady[:,j] = (data[:,j] - data[:,j-1])/step
            else:
                ddatady[:,j] = (data[:,j+1] - data[:,j-1])/(2*step)
    return ddatady
@njit
def deriv12(data, step, onesided):
    _, num_y = data.shape
    ddatadyy = np.zeros(data.shape)
    for j in range(num_y):
        if j == 0:
            ddatadyy[:,j] = (data[:,j+2] -2*data[:,j+1] + data[:,j])/(step**2)
        elif j == num_y -1:
            ddatadyy[:,j] = (data[:,j] -2*data[:,j-1] + data[:,j-2])/(step**2)
        else:
            ddatadyy[:,j] = (data[:,j+1] -2*data[:,j] + data[:,j-1])/(step**2)
    return ddatadyy


# -

# derivative 1d
@njit
def derivative_1d(ϕ, order, dx, upwind=False):
    num, = ϕ.shape
    dϕ = np.zeros_like(ϕ)
    if order == 1:
        for i in range(num):
            if i == 0:
                dϕ[i] = (ϕ[i+1]-ϕ[i])/dx
            elif i == num-1:
                dϕ[i] = (ϕ[i]-ϕ[i-1])/dx
            else: 
                if upwind == True:
                    dϕ[i] = (ϕ[i]-ϕ[i-1])/dx
                else:
                    dϕ[i] = (ϕ[i+1]-ϕ[i-1])/(2*dx)
    elif order == 2:
        for i in range(num):
            if i == 0:
                dϕ[i] = (ϕ[i+2]-2*ϕ[i+1] + ϕ[i])/(dx**2)
            elif i == num -1:
                dϕ[i] = (ϕ[i]-2*ϕ[i-1] + ϕ[i-2])/(dx**2)
            else:
                dϕ[i] = (ϕ[i+1]- 2*ϕ[i] + ϕ[i-1])/(dx**2)
    
    return dϕ
