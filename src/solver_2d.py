"""
Functions to numerically solve 2d HJB.
"""
import numpy as np
import scipy
import SolveLinSys
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from numba import njit


@njit
def upwind_2d(LHS, A, B1, B2, C1, C2, i, j, dx, dy, ϵ):
    """
    Compute the coefficient of the equation at v(i, j).
    Parameters
    ----------
    LHS : (I*J, I*J) ndarray
        LHS matrix of the linear system.
    A, B1, B2, C1, C2 : (I, J) ndarrays
    i, j : ints
    dx, dy : floats
    ϵ : float
        False transient step size.
    Returns
    -------
    LHS : (I*J, I*J) ndarray
        Updated LHS matrix of the linear system.
    """
    I, J = A.shape
    loc_eq = i*J + j
    loc_xy = loc_eq
    loc_xp1y = (i+1)*J + j
    loc_xp2y = (i+2)*J + j
    loc_xm1y = (i-1)*J + j
    loc_xm2y = (i-2)*J + j
    loc_xyp1 = i*J + j+1
    loc_xyp2 = i*J + j+2
    loc_xym1 = i*J + j-1
    loc_xym2 = i*J + j-2
    LHS[loc_xy, loc_xy] += A[i, j] - 1./ϵ
    if i == 0:
        LHS[loc_eq, loc_xy] += B1[i, j] * (-1./dx) + C1[i, j] * (1./dx**2)
        LHS[loc_eq, loc_xp1y] += B1[i, j] * (1./dx) + C1[i, j] * (-2./dx**2)
        LHS[loc_eq, loc_xp2y] += C1[i, j] * (1./dx**2)
    elif i == I-1:
        LHS[loc_eq, loc_xy] += B1[i, j] * (1./dx) + C1[i, j] * (1./dx**2)
        LHS[loc_eq, loc_xm1y] += B1[i, j] * (-1./dx) + C1[i, j] * (-2./dx**2)
        LHS[loc_eq, loc_xm2y] += C1[i, j] * (1./dx**2)
    else:
        LHS[loc_eq, loc_xy] += B1[i, j] * ((-1./dx) * (B1[i, j]>0) + (1./dx) * (B1[i, j]<=0))\
                    + C1[i, j] * (-2./dx**2)
        LHS[loc_eq, loc_xm1y] += B1[i, j] * (-1./dx) * (B1[i, j]<=0) + C1[i, j] * (1./dx**2)
        LHS[loc_eq, loc_xp1y] += B1[i, j] * (1./dx) * (B1[i, j]>0) + C1[i, j] * (1./dx**2)
    if j == 0:
        LHS[loc_eq, loc_xy] += B2[i, j] * (-1./dy) + C2[i, j] * (1./dy**2)
        LHS[loc_eq, loc_xyp1] += B2[i, j] * (1./dy) + C2[i, j] * (-2./dy**2)
        LHS[loc_eq, loc_xyp2] += C2[i, j] * (1./dy**2)
    elif j == J-1:
        LHS[loc_eq, loc_xy] += B2[i, j] * (1./dy) + C2[i, j] * (1./dy**2)
        LHS[loc_eq, loc_xym1] += B2[i, j] * (-1./dy) + C2[i, j] * (-2./dy**2)
        LHS[loc_eq, loc_xym2] += C2[i, j] * (1./dy**2)
    else:
        LHS[loc_eq, loc_xy] += B2[i, j] * ((-1./dy) * (B2[i, j]>0) + (1./dy) * (B2[i, j]<=0))\
                    + C2[i, j] * (-2./dy**2)
        LHS[loc_eq, loc_xym1] += B2[i, j] * (-1./dy) * (B2[i, j]<=0) + C2[i, j] * (1./dy**2)
        LHS[loc_eq, loc_xyp1] += B2[i, j] * (1./dy) * (B2[i, j]>0) + C2[i, j] * (1./dy**2)
    return LHS


@njit
def construct_matrix_2d(A, B1, B2, C1, C2, D, v0, ϵ, dx, dy, bc, impose_bc):
    """
    Construct coefficient matrix of the linear system.
    
    Parameters
    ----------
    A, B1, B2, C1, C2, D : (I, J) ndarrays
    v0 : (I, J) ndarray
        Value function from last iteration.
    ϵ : False transient step size
    dx, dy : floats    
    bc : tuple of ndarrays
        Impose v=bc[k] at boundaries.
        Order: lower boundary of x, upper boundary of x,
               lower boundary of y, upper boundary of y.
    impose_bc : tuple of bools
        Order: lower boundary of x, upper boundary of x,
               lower boundary of y, upper boundary of y.
    Returns
    -------
    LHS : (I*J, I*J) ndarray
        LHS of the linear system.
    RHS : (I*J) ndarray
        RHS of the linear system.
    """
    I, J = A.shape
    LHS = np.zeros((I*J, I*J))
    RHS = - D.reshape(-1) - 1./ϵ*v0.reshape(-1)
    for i in range(I):
        for j in range(J):
            loc_eq = i*J + j
            if i == 0 and impose_bc[0]:
                LHS[loc_eq, loc_eq] = 1.
                RHS[loc_eq] = bc[0][j]
            elif i == I-1 and impose_bc[1]:
                LHS[loc_eq, loc_eq] = 1.
                RHS[loc_eq] = bc[1][j]
            elif j == 0 and impose_bc[2]:
                LHS[loc_eq, loc_eq] = 1.
                RHS[loc_eq] = bc[2][i]
            elif j == J-1 and impose_bc[3]:
                LHS[loc_eq, loc_eq] = 1.
                RHS[loc_eq] = bc[3][i]
            else:
                LHS = upwind_2d(LHS, A, B1, B2, C1, C2, i, j, dx, dy, ϵ)
    return LHS, RHS


def false_transient_one_iteration_python(A, B1, B2, C1, C2, D, v0, ϵ, dx, dy, bc, impose_bc):
    I, J = A.shape
    LHS, RHS = construct_matrix_2d(A, B1, B2, C1, C2, D, v0, ϵ, dx, dy, bc, impose_bc)
    v, exit_code = bicg(csc_matrix(LHS), RHS)
    return v.reshape((I, J))


def false_transient_one_iteration_cpp(stateSpace, A, B1, B2, C1, C2, D, v0, ε):
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F')])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F')])
    D = D.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0.reshape(-1, 1, order='F'), ε, -10)
    return out[2].reshape(v0.shape, order = "F")