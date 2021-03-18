"""
Functions to numerically solve a nonlinear ODE via false transient scheme.

"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from numba import njit


@njit
def compute_coefficient(LHS, A, B, C, i, dx, ϵ):
    """
    Compute the coefficient of the equation at v(i).

    Parameters
    ----------
    LHS : (I, I) ndarray
        LHS matrix of the linear system.
    A, B, C : (I,) ndarrays
    i : int
    dx : float
    ϵ : float
        False transient step size.

    Returns
    -------
    LHS : (I, I) ndarray
        Updated LHS matrix of the linear system.

    """
    I = len(A)
    LHS[i, i] += A[i] - 1./ϵ
    if i == 0:
        LHS[i, i] += B[i] * (-1./dx) + C[i] * (1./dx**2)
        LHS[i, i+1] += B[i] * (1./dx) + C[i] * (-2./dx**2)
        LHS[i, i+2] += C[i] * (1./dx**2)
    elif i == I-1:
        LHS[i, i] += B[i] * (1./dx) + C[i] * (1./dx**2)
        LHS[i, i-1] += B[i] * (-1./dx) + C[i] * (-2./dx**2)
        LHS[i, i-2] += C[i] * (1./dx**2)
    else:
        LHS[i, i] += B[i] * ((-1./dx) * (B[i]>0) + (1./dx) * (B[i]<=0))\
                    + C[i] * (-2./dx**2)
        LHS[i, i-1] += B[i] * (-1./dx) * (B[i]<=0) + C[i] * (1./dx**2)
        LHS[i, i+1] += B[i] * (1./dx) * (B[i]>0) + C[i] * (1./dx**2)
    return LHS


@njit
def linearize(A, B, C, D, v0, ϵ, dx, bc, impose_bc):
    """
    Construct coefficient matrix of the linear system.
    
    Parameters
    ----------
    A, B, C, D : (I,) ndarrays
    v0 : (I,) ndarray
        Value function from last iteration.
    ϵ : False transient step size
    dx : float  
    bc : tuple of ndarrays
        Impose v=bc[k] at boundaries.
        Order: lower boundary of x, upper boundary of x,
    impose_bc : tuple of bools
        Order: lower boundary of x, upper boundary of x,

    Returns
    -------
    LHS : (I, I) ndarray
        LHS of the linear system.
    RHS : (I,) ndarray
        RHS of the linear system.

    """
    I = len(A)
    LHS = np.zeros((I, I))
    RHS = - D - 1./ϵ*v0
    for i in range(I):
        if i == 0 and impose_bc[0]:
            LHS[i, i] = 1.
            RHS[i] = bc[0]
        elif i == I-1 and impose_bc[1]:
            LHS[i, i] = 1.
            RHS[i] = bc[1]
        else:
            LHS = compute_coefficient(LHS, A, B, C, i, dx, ϵ)
    return LHS, RHS


def false_transient(A, B, C, D, v0, ϵ, dx, bc, impose_bc):
    """
    Implement false transient scheme.
    
    """
    LHS, RHS = linearize(A, B, C, D, v0, ϵ, dx, bc, impose_bc)
    v, exit_code = bicg(csc_matrix(LHS), RHS)
    return v
