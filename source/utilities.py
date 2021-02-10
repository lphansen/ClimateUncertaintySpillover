"""
Utility functions to facilitate computation.

"""
import numpy as np
from numba import njit

@njit
def find_nearest_value(array, value):
    loc = np.abs(array - value).argmin()
    return loc

@njit
def compute_derivatives_3d(data, dim, order, dlt):
    res = np.zeros_like(data)
    if order == 1:                    # first order derivatives
        if dim == 0:                  # to first dimension
            res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
            res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
            res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])
        elif dim == 1:                # to second dimension
            res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
            res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
            res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])
        elif dim == 2:                # to third dimension
            res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
            res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
            res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])
    elif order == 2:
        if dim == 0:                  # to first dimension
            res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
            res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
            res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])
        elif dim == 1:                # to second dimension
            res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
            res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
            res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])
        elif dim == 2:                # to third dimension
            res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
            res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
            res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])    
    return res


@njit
def compute_derivatives_2d(data, dim, order, dlt):
    res = np.zeros_like(data)
    if order == 1:                    # first order derivatives
        if dim == 0:                  # to first dimension
            res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
            res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
            res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])
        elif dim == 1:                # to second dimension
            res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
            res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
            res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])        
    elif order == 2:
        if dim == 0:                  # to first dimension
            res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
            res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
            res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])
        elif dim == 1:                # to second dimension
            res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
            res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
            res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])    
    return res
