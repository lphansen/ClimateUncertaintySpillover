"""
Functions to numerically solve ψ and ϕ.

"""

import numpy as np
from numba import njit


@njit(parallel=True)
def solve_ψ(ell=1., b_step=1e-2, args=(0.01, 0.032, 0.00175 * 0.012, 0)):
    r"""
    Given :math:`\ell`, numerically solve for ψ on a grid of b from 0 to 1.

    Parameters
    ----------
    ell : float
        Value of variable :math:`\ell`.
    b_step : float
        Step size of b grid. b is evenly spaced from b_step to 1.
    args : list/tuple of floats
        Values for δ, η, τ_1, τ_2 respectively.

    Returns
    -------
    ψ_grid : (ell/b_step, ) ndarray
        ψ values on the grid of b, given :math:`\ell`. 

    """
    δ, η, τ_1, τ_2 = args
    b_size = int(1./b_step)
    b_grid = np.linspace(b_step, 1., b_size)

    # Construct a grid for the linear system Aψ+B=0
    A = np.zeros((b_size, b_size))
    B = np.zeros(b_size)
    for i in range(b_size):
        b = b_grid[i]
        # Calculate e_star
        if τ_2 == 0:
            e_star = b*δ*η/(b*τ_1+ell)
        else:
            e_star = (-b*τ_1-ell+np.sqrt((b*τ_1+ell)**2 + 4*b**2*τ_2*δ*η))/(2*b*τ_2)
        # Construct coefficient B
        B[i] = b*(δ*η*np.log(e_star)-τ_1*e_star-τ_2/2*e_star**2) - ell*e_star
        # Construct coefficient matrix A
        if i == 0:
            # Impose boundary condition ψ(0;ell)=0
            A[i, i+1] = -δ*b/(2*b_step) 
        elif i == b_size-1:
            A[i, i-1] = δ*b/b_step
            A[i, i] = - δ*b/b_step
        else:
            A[i, i-1] = δ*b/(2*b_step)
            A[i, i+1] = -δ*b/(2*b_step)
    ψ_grid = np.linalg.solve(A, -B)
    return ψ_grid


@njit
def compute_ϕ_r(ell=1., d_step=1e-9, b_step=1e-2, args=(0.01, 0.032, 0.00175 * 0.012, 0)):
    r"""
    Given :math:`\ell`, compute a pair of ϕ and r.
    
    Parameters
    ----------
    ell : float
        Value of variable :math:`\ell`.
    d_step : float
        Step size of :math:`\ell`, used to calculate numerical derivative of ψ.
    b_step : float
        Step size of b grid. b is evenly spaced from b_step to 1.
    args : list/tuple of floats
        Values for δ, η, τ_1, τ_2 respectively.

    Returns
    -------
    r : float
        Value of r evaluated at given :math:`\ell`.
    ϕ : float
        Value of ϕ evaluated at given :math:`\ell`.

    """
    ψ_ip1 = solve_ψ(ell+d_step, b_step, args)[-1]
    ψ_i = solve_ψ(ell, b_step, args)[-1]
    dψ_i = (ψ_ip1-ψ_i)/d_step  # One-sided derivative
    r = - dψ_i
    ϕ = ψ_i + ell * r
    return r, ϕ


@njit(parallel=True)
def trace_ϕ_r(log_ell_min=-20, log_ell_max=10, grid_size=1000,
            d_step=1e-9, b_step=1e-2,
            args=(0.01, 0.032, 0.00175 * 0.012, 0)):
    r"""
    Compute pairs of ϕ and r based on a grid of :math:`\log \ell`.

    Parameters
    ----------
    log_ell_min : float
        Minimum value of the grid for :math:`\log \ell`.
    log_ell_max : float
        Maximum value of the grid for :math:`\log \ell`.
    grid_size : float
        Number of points of the grid for :math:`\log \ell`.
        The grid is evenly spaced.
    d_step : float
        Step size of :math:`\ell`, used to calculate numerical derivative of ψ.
    b_step : float
        Step size of b grid. b is evenly spaced from b_step to 1.
    args : list/tuple of floats
        Values for δ, η, τ_1, τ_2 respectively.

    Returns
    -------
    r_grid_sorted : (grid_size, ) ndarray
        Grid of r sorted from low to high.
    ϕ_grid_sorted : (grid_size, ) ndarray
        Grid of ϕ in the same order as r_grid_sorted.  

    """
    log_ell_grid = np.linspace(log_ell_min, log_ell_max, grid_size)
    ell_grid = np.exp(log_ell_grid)
    r_grid = np.zeros_like(ell_grid)
    ϕ_grid = np.zeros_like(ell_grid)
    for i in range(grid_size):
        ell = ell_grid[i]
        r_grid[i], ϕ_grid[i] = compute_ϕ_r(ell, d_step, b_step, args)
    sort_indices = np.argsort(r_grid) # Set r from low to high
    r_grid_sorted = r_grid[sort_indices]
    ϕ_grid_sorted = ϕ_grid[sort_indices]
    return r_grid_sorted, ϕ_grid_sorted
