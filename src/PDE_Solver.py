#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np

from numba import njit


# %%

def PDESolver_nd(stateSpace, A, B, C, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1  , order = 'F')
        B = B.reshape(-1,1  , order = 'F')
        C = C.reshape(-1,1  , order = 'F')
        D = D.reshape(-1,1  , order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':

        if smartguess:
            iters = 1
        else:
            iters = 400000

        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)
        return out


# %%
def derivatives_nd(data, dim, order, step, onesided=True):
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




# %%
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

# %%
