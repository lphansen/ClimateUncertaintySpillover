"""
Functions to simulate deterministic or stochastic paths for variables of interest.

"""
import warnings
import numpy as np
from numba import njit
from utilities import find_nearest_value
from solver import trace_ϕ_r


@njit
def simulate_log_damage(exp_avg_response, σ_n, Et, Ws):
    """
    Simulate log damage.

    Parameters
    ----------
    exp_avg_response, σ_n : float
        Model parameters.
    Et : (T, ) ndarray
        Emission trajectory.
    Ws : (N, T) ndarray
        Iid normal shocks for N paths.

    Returns
    -------
    Ys : (T, ) ndarray
        Simulated log damages.

    """
    Ys = np.zeros_like(Ws)
    for path in range(Ws.shape[0]):
        Y = 0.
        for i in range(Ws.shape[1]):
            dY = exp_avg_response * Et[i] * (1+σ_n*Ws[path,i])
            Ys[path, i] = dY + Y
            Y = Ys[path, i]
    return Ys


def simulate_emission_quadratic(δ, η, τ_1, ξ, σ_n,
                                args_trace_ϕ = (-20, -5, 1000, 1e-9, 1e-3),
                                r_start=9000, T=100):
    """
    Simulate emission assuming the quadratic structure in Section 5.7.

    Parameters
    ----------
    δ, η, τ_1, ξ, σ_n : float
        Model parameters
    args_trace_ϕ : list/tuple of floats
        Values for log_ell_min, log_ell_max, grid_size, d_step,
        b_step respectively.
    r_start : float
        Initial reserve.
    T : int
        Time length for the simulation.

    Returns
    -------
    Et : (T, ) ndarray
        Emission trajectory.

    """
    τ_2 = (τ_1**2) * (σ_n**2) / (2*ξ)
    args = (δ, η, τ_1, τ_2)

    r_grid, ϕ_grid = trace_ϕ_r(*args_trace_ϕ, args)

    # Calculate dϕ/dr, e_star and Et
    r_grid, indices = np.unique(r_grid, return_index=True)
    ϕ_grid = ϕ_grid[indices]
    ϕ_der_grid = (ϕ_grid[1:]-ϕ_grid[:-1])/(r_grid[1:]-r_grid[:-1])
    r_grid = (r_grid[1:]+r_grid[:-1])/2
    ϕ_grid = (ϕ_grid[1:]+ϕ_grid[:-1])/2
    e_star = δ*η/(τ_1 + ϕ_der_grid)
    if np.max(r_grid) < r_start:
        warnings.warn("r_start exceeds the maximum value of the grid of r. Try changing the grid of log ell or decreasing r_start.")
    Et = _simulate_emission(e_star, r_grid, r_start=r_start, T=T)

    return Et, r_grid, ϕ_grid, e_star


@njit(parallel=True)
def _simulate_emission(e_grid, r_grid, r_start=9000, T=100):
    """
    Simulate emission trajectory baesd on grids of emission and reserve.

    Parameters
    ----------
    e_grid : (N, ) ndarray
        Grid of emission.
    r_grid : (N, ) ndarray
        Grid of reserve.

    Returns
    -------
    Et : (T, ) ndarray
        Emission trajectory.

    """
    Et = np.zeros(T)
    r_remain = r_start
    for i in range(T):
        loc = find_nearest_value(r_grid, r_remain)
        Et[i] = e_grid[loc]
        r_remain = r_remain - Et[i]
    return Et
