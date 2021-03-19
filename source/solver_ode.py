# -*- coding: utf-8 -*-
"""
module for solving ode
"""
import numpy as np
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg


@njit
def derivative_1d(data, order, h_data, side="up"):
    num, = data.shape
    ddata = np.zeros_like(data)
    if order == 1:
        for i in range(num):
            if i == 0:
                ddata[i] = (data[i+1]-data[i])/h_data
            elif i == num-1:
                ddata[i] = (data[i]-data[i-1])/h_data
            else: 
                if side == "up":
                    ddata[i] = (data[i]-data[i-1])/h_data
                elif side == 'center':
                    ddata[i] = (data[i+1]-data[i-1])/(2*h_data)
                elif side == "down":
                    ddata[i] = (data[i+1]-data[i])/h_data
                else:
                    raise ValueError
    elif order == 2:
        for i in range(num):
            if i == 0:
                ddata[i] = (data[i+2]-2*data[i+1] + data[i])/(h_data**2)
            elif i == num -1:
                ddata[i] = (data[i]-2*data[i-1] + data[i-2])/(h_data**2)
            else:
                ddata[i] = (data[i+1]- 2*data[i] + data[i-1])/(h_data**2)
    
    return ddata


@njit
def get_coeff(A, Bx, Cxx, D, x_grid, ϕ_prev, ϵ, boundspec):
    dx = x_grid[1] - x_grid[0]
    numx = len(x_grid)
    LHS = np.zeros((numx, numx))
    RHS = -1/ϵ*ϕ_prev - D
    for i in range(numx):
        LHS[i,i] += - 1/ϵ + A[i]
        if i == 0:
            LHS[i,i] += - 1/dx*Bx[i] + Cxx[i]/(dx**2)
            LHS[i,i+1] += 1/dx*Bx[i] - 2*Cxx[i]/(dx**2)
            LHS[i,i+2] += Cxx[i]/(dx**2)
        elif i == numx-1:
            if boundspec[0] == True:
                LHS[i,i] = 1
                RHS[i] = boundspec[1]
            else:
                LHS[i,i] += 1/dx*Bx[i] + Cxx[i]/(dx**2)
                LHS[i,i-1] += -1/dx*Bx[i] - 2*Cxx[i]/(dx**2)
                LHS[i,i-2] += Cxx[i]/(dx**2)
        else:
            LHS[i,i+1] += Bx[i]*(1./dx)*(Bx[i]>0) + Cxx[i]/(dx**2)
            LHS[i,i] += Bx[i]*((-1/dx)*(Bx[i]>0) + (1/dx)*(Bx[i]<0)) - 2*Cxx[i]/(dx**2)
            LHS[i,i-1] += Bx[i]*(-1/dx)*(Bx[i]<0) + Cxx[i]/(dx**2)
    return LHS, RHS


@njit
def get_coeff_one(A, Bx, Cxx, D, x_grid, boundspec):
    dx = x_grid[1] - x_grid[0]
    numx = len(x_grid)
    LHS = np.zeros((numx, numx))
    RHS =  - D
    for i in range(numx):
        LHS[i,i] +=  A[i]
        if i == 0:
            LHS[i,i] += - 1/dx*Bx[i] + Cxx[i]/(dx**2)
            LHS[i,i+1] += 1/dx*Bx[i] - 2*Cxx[i]/(dx**2)
            LHS[i, i+1] += + Cxx[i]/(dx**2)
        elif i == numx-1:
            if boundspec[0] == True:
                LHS[i,i] = 1
                RHS[i] = boundspec[1]
            else:
                LHS[i,i] += 1/dx*Bx[i] + Cxx[i]/(dx**2)
                LHS[i,i-1] += -1/dx*Bx[i]  - 2*Cxx[i]/(dx**2)
                LHS[i, i+1] += Cxx[i]/(dx**2)
        else:
            LHS[i,i+1] += Bx[i]*(1./dx)*(Bx[i]>0) + Cxx[i]/(dx**2)
            LHS[i,i] += Bx[i]*((-1/dx)*(Bx[i]>0) + (1/dx)*(Bx[i]<0)) - 2*Cxx[i]/(dx**2)
            LHS[i,i-1] += Bx[i]*(-1/dx)*(Bx[i]<0) + Cxx[i]/(dx**2)
    return LHS, RHS


def solve_ode_one( A, By, Cyy, D, y_grid,  boundspec):
    LHS, RHS = get_coeff_one( A, By, Cyy, D, y_grid, boundspec)
    phi_grid, exit_code = bicg(csc_matrix(LHS), RHS)
#     phi_grid = np.linalg.solve(LHS, RHS)
    return phi_grid


def solve_ode( A, By, Cyy, D, y_grid, ϕ_prev, ϵ, boundspec):
    LHS, RHS = get_coeff( A, By, Cyy, D, y_grid, ϕ_prev, ϵ, boundspec)
    phi_grid, exit_code = bicg(csc_matrix(LHS), RHS)
#     phi_grid = np.linalg.solve(LHS, RHS)
    return phi_grid
