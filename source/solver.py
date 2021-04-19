# -*- coding: utf-8 -*-
"""
Functions to numerically solve a nonlinear ODE via false transient scheme.

.. seealso::
   For more details on false transient:
   refer to :doc:`~app.appendices` (TODO:fix to ref).


The ODE we solve is as follows:

The state space descretize into :math:`y_1, y_2, \\dots, y_N` with equal interval :math:`\\Delta y`.
For each :math:`y_n \\in \{y_1, y_2, \\dots, y_N\}` in the grid with , the value function satifies 
(we use upwinding first order derivative for concern of, 
and at :math:`y_1` we use forward derivative instead):

.. math::
   :label: ODE
   
   \\begin{align}
   \\frac{\\phi_{i+1}(y_n) - \\phi_{i}(y)}{\\epsilon} =& \\quad A_n \\phi_{i+1}(y) \\cr
   & + B_{n} \\frac{\\phi_{i+1}(y_n) - \\phi_{i+1}(y_{n-1})}{\Delta y}  \\cr
   & + C_n \\frac{\\phi_{i+1}(y_{n+1}) - 2 \\phi_{i+1}(y_{n}) + \\phi_{i+1}(y_{n-1})}{\Delta y^2} \\cr
   & + D_n
   \\end{align}
   
   

where :math:`A_n`, :math:`B_n`, :math:`C_n` and :math:`D_n` are coefficients.
The exact values are given by the HJB to be solved.
Therefore, we construct the following linear system:

.. math::
   :label: FT
   
   LHS \\cdot \\phi_{i+1}(Y) = - D - \\frac{1}{\\epsilon} \\phi_{i}(Y)
   
where :math:`LHS` is a :math:`N\\times N` matrix of coefficients, :math:`Y = (y_1, y_2, \\dots, y_N)'`, 
and :math:`D = (D_1, D_2, \\dots, D_n)'`. 
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from numba import njit


@njit
def compute_coefficient(LHS, A, B, C, i, dx, ϵ):
    r"""
    Compute the coefficient of the equation at :math:`y_i`.
    Values are computed according to :math:numref:`ODE`.

    Parameters
    ----------
    LHS : (I, I) ndarray
        LHS matrix of the linear system.
    A : (I,) ndarrays
        Coefficient arrays for value function :math:`\phi(y_i)`.
    B : (I,) ndarrays
        Coefficient arrays for first order derivative.
    C : (I,) ndarrays
        Coefficient arrays for second order derivative.
    i : int
        Index for the grid point.
    dx : float
        Grid step, :math:`\Delta y`.
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
    r"""
    Construct coefficient matrix of the linear system.
     
    Parameters
    ----------
    A, B, C, D : (I,) ndarrays
        see :func:`~source.solver.compute_coefficient` for further detail.
    v0 : (I,) ndarray
        Value function from last iteration, :math:`\phi_i(y)`.
    ϵ : float
        False transient step size.
    dx : float
        Grid step size.
    bc : tuple of ndarrays::
        Impose `v=bc[k]` at boundaries.
        
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
    r"""
    Implement false transient scheme of one iteration,
    and update from :math:`\phi_i(y)` to :math:`\phi_{i+1}(y)`
    according to :math:numref:`FT`.
    
    See appendix B(TODO: cross-ref link) for further detail.
    
    Parameters
    ----------
    A, B, C, D: 
        Same as those in :func:`~source.solver.compute_coefficient`
    v0 : (I, ) ndarrays
        Value function from last iteration, :math:`\phi_i(y)`
    ϵ : float
        Step size of false transient
    dx : float
        Step size of Grid
    bc : 
        See :func:`~source.solver.linearize`.
    impose_bc:
        See :func:`~source.solver.linearize`.
        
    Returns
    -------
    v :  (I, ) ndarrays
        Updated value function, :math:`\phi_{i+1}(y)` according to :math:numref:`FT`.
    """
    LHS, RHS = linearize(A, B, C, D, v0, ϵ, dx, bc, impose_bc)
    v, exit_code = bicg(csc_matrix(LHS), RHS)
    return v
