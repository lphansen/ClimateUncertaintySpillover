import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm
import pdb
import warnings
import time
from scipy.interpolate import RegularGridInterpolator, CubicSpline
import pickle
import SolveLinSys

c_temp = 24
k_temp = 5.35
sigma_T = 8.4478e-7 ** 0.5
gamma_bar = 2
Q_0 = 342.5
T_b = 273.15 + 13.9
T0 = 284.15
mu = 1
Tc = 269
sigma_ZG = 5.6697e-8

Tb = T_b
μ = 1
σ = sigma_ZG

def Ri(κ, Tb, T,μ = 1, c1 = 0.15, c2 = 0.7, Tc = 273):
    α = c1 + c2 / 2 * (1 - np.tanh(κ * (T + Tb - Tc)))
    return 342.5 * μ * (1 - α)

def North(T,Tb, B = 2.09):
    return -B * (T-Tb)

def RO(Tb,T,m):
    return σ * (T + Tb) ** 4 * ( 1- m * np.tanh(( (T+Tb)) ** 6 * 1.9e-15))

def Ri_1(κ, Tb, T,μ = 1, c2 = 0.7, Tc = 273):
    return -342.5 * μ * (- c2 / 2 * (1 - np.tanh(κ * (T + Tb - Tc)) ** 2 )) * κ

def RO_1(Tb, T,m):
    return 4 * σ * (T + Tb) ** 3 * ( 1- m * np.tanh(( (T+Tb) / T0) ** 6)) + σ * (T + Tb) ** 4 * - m * (1 - np.tanh(( (T+Tb) / T0) ** 6) ** 2) * 6 * ( (T+Tb) / T0) ** 5 / T0


def diff_NorthtoGhil(p, Tb, T, μ,c1,c2,Tc,B=2.09):
    κ, m = p
    return (Ri(κ, Tb, T, μ,c1,c2, Tc) - RO(Tb, T, m), Ri_1(κ, Tb, T, μ, c2, Tc) - RO_1(Tb, T, m) + B)

def finiteDiff(data, dim, order, dlt, cap = None):
    """ compute the central difference derivatives for given input and dimensions;
    dim is the number of state varibles;
    order: order of the derivative;
    dlt: delta of variables"""

    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 4:
        if order == 1:                    # first order derivatives

            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / (2 * dlt)) * (data[2:,:,:,:] - data[:-2,:,:,:])
                res[-1,:,:,:] = (1 / dlt) * (data[-1,:,:,:] - data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt) * (data[1,:,:,:] - data[0,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / (2 * dlt)) * (data[:,2:,:,:] - data[:,:-2,:,:])
                res[:,-1,:,:] = (1 / dlt) * (data[:,-1,:,:] - data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt) * (data[:,1,:,:] - data[:,0,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / (2 * dlt)) * (data[:,:,2:,:] - data[:,:,:-2,:])
                res[:,:,-1,:] = (1 / dlt) * (data[:,:,-1,:] - data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt) * (data[:,:,1,:] - data[:,:,0,:])

            elif dim == 3:                # to forth dimension

                res[:,:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,:,2:] - data[:,:,:,:-2])
                res[:,:,:,-1] = (1 / dlt) * (data[:,:,:,-1] - data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt) * (data[:,:,:,1] - data[:,:,:,0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:

            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / dlt ** 2) * (data[2:,:,:,:] + data[:-2,:,:,:] - 2 * data[1:-1,:,:,:])
                res[-1,:,:,:] = (1 / dlt ** 2) * (data[-1,:,:,:] + data[-3,:,:,:] - 2 * data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt ** 2) * (data[2,:,:,:] + data[0,:,:,:] - 2 * data[1,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / dlt ** 2) * (data[:,2:,:,:] + data[:,:-2,:,:] - 2 * data[:,1:-1,:,:])
                res[:,-1,:,:] = (1 / dlt ** 2) * (data[:,-1,:,:] + data[:,-3,:,:] - 2 * data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt ** 2) * (data[:,2,:,:] + data[:,0,:,:] - 2 * data[:,1,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / dlt ** 2) * (data[:,:,2:,:] + data[:,:,:-2,:] - 2 * data[:,:,1:-1,:])
                res[:,:,-1,:] = (1 / dlt ** 2) * (data[:,:,-1,:] + data[:,:,-3,:] - 2 * data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt ** 2) * (data[:,:,2,:] + data[:,:,0,:] - 2 * data[:,:,1,:])

            elif dim == 3:                # to third dimension

                res[:,:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,:,2:] + data[:,:,:,:-2] - 2 * data[:,:,:,1:-1])
                res[:,:,:,-1] = (1 / dlt ** 2) * (data[:,:,:,-1] + data[:,:,:,-3] - 2 * data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt ** 2) * (data[:,:,:,2] + data[:,:,:,0] - 2 * data[:,:,:,1])

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    elif l == 3:
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

            else:
                raise ValueError('wrong dim')

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

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives

            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:

            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    else:
        if order == 1:                    # first order derivatives

            res[1:-1] = (1 / (2 * dlt)) * (data[2:] - data[:-2])
            res[-1] = (1 / dlt) * (data[-1] - data[-2])
            res[0] = (1 / dlt) * (data[1] - data[0])

        elif order == 2:

            res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
            res[-1] = (1 / dlt ** 2) * (data[-1] + data[-3] - 2 * data[-2])
            res[0] = (1 / dlt ** 2) * (data[2] + data[0] - 2 * data[1])
    if cap is not None:
        res[res < cap] = cap
    return res

def quad_points_legendre(n):
    u = np.sqrt(1 / (4 - 1 / np.linspace(1,n-1,n-1)**2))  # upper diag
    [lambda0,V] = np.linalg.eig(np.diagflat(u,1) + np.diagflat(u,-1))  # V's column vectors are the main d
    i = np.argsort(lambda0)
    Vtop = V[0,:]
    Vtop = Vtop[i]
    w = 2 * Vtop ** 2
    return (lambda0[i],w)

def quad_points_hermite(n):
    i = np.linspace(1,n-1,n-1)
    a = np.sqrt(i / 2.0)
    [lambda0,V] = np.linalg.eig(np.diagflat(a,1) + np.diagflat(a,-1))
    i = np.argsort(lambda0)
    Vtop = V[0,:]
    Vtop = Vtop[i]
    w = np.sqrt(np.pi) * Vtop ** 2
    return (lambda0[i],w)


def quad_int(f,a,b,n,method):
    """
    This function takes a function f to integrate from the multidimensional
    interval specified by the row vectors a and b. N different points are used
    in the quadrature method. Legendre and Hermite basis functions are
    currently supported. In the case of Hermite methodology b is the normal
    density and a is the normal mean.

    Created by John Wilson (johnrwilson@uchicago.edu) & Updaed by Jiaming Wang (Jiamingwang@uchicago.edu)
    """
    if method == 'legendre':

        (xs,ws) = quad_points_legendre(n)
        g = lambda x: f((b-a) * 0.5  * x + (a + b) * 0.5)
        s = np.prod((b-a) * 0.5)                ######## Why using prod here?

    elif method == 'hermite':

        (xs,ws) = quad_points_hermite(n)
        g = lambda x: f(np.sqrt(2) * b * x + a)
        s = 1 / np.sqrt(np.pi)

    else:
        raise TypeError('Wrong polynomials specification')

    """
    tp = type(a)
    if tp is np.float64 or tp is int or tp is np.double:
        res = 0
        for i in range(n):
    #             pdb.set_trace()
            res += ws[i] * g(xs[i])
    else:
        raise ValueError('dimension is not 1')

    return s * res
    """
    res = 0
    for i in range(n):
        res += ws[i] * g(xs[i])
    return s * res


def cap(x, lb, ub):
    if x <= ub or x >= lb:
        return x
    else:
        if x > ub:
            return ub
        else:
            return lb


def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':

        if smartguess:
            iters = 1
        else:
            iters = 400000

        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out