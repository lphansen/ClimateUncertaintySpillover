# -*- coding: utf-8 -*-
"""
module for simulation
"""
import numpy as np
from scipy import interpolate
# function claim
def simulate_jump(model_res, θ_list, ME=None,  y_start=1,  T=100, dt=1):
    """
    Simulate temperature anomaly, emission, distorted probabilities of climate models, 
    distorted probabilities of damage functions, and drift distortion.
    When ME is asigned value, it will also simulate paths for marginal value of emission
    
    Parameters
    ----------
    model_res : dict::
        A dictionary storing solution with misspecified jump process. See :func:`~source.model.solve_hjb_y_jump` 
        for detail.
    θ_list : (N,) ndarray::
        A list of matthew coefficients. Unit: celsius/gigaton of carbon.
    ME : (N,) ndarray
        Marginal value of emission as a function of y.
    y_start : float, default=1
        Initial value of y.
    T : int, default=100
        Time span of simulation.
    dt : float, default=1
        Time interval of simulation.
        
    Returns
    -------
    simulation_res: dict of ndarrays
        dict: {
            yt : (T,) ndarray
                Temperature anomaly trajectories.
            et : (T,) ndarray
                Emission trajectories.
            πct : (T, L) ndarray
                Trajectories for distorted probabilities of climate models.
            πdt : (T, M) ndarray
                Trajectories for distorted probabilities of damage functions.
            ht : (T,) ndarray
                Trajectories for drift distortion.
            if ME is not None, the dictionary will also include
                me_t : (T,) ndarray
                    Trajectories for marginal value of emission.
        }
    """
    y_grid = model_res["y"]
    ems = model_res["e_tilde"]
    πc = model_res["πc"]
    πd = model_res["πd"]
    h = model_res["h"]
    periods = int(T/dt)
    et = np.zeros(periods)
    yt = np.zeros(periods)
    πct = np.zeros((periods, len(θ_list)))
    πdt = np.zeros((periods, len(πd)))
    ht = np.zeros(periods)
    if ME is not None:
        me_t = np.zeros(periods)
    # interpolate
    get_πd = interpolate.interp1d(y_grid, πd)
    get_πc = interpolate.interp1d(y_grid, πc)
#     y = np.mean(θ_list)*290
    y = y_start
    for t in range(periods):
        if y > np.max(y_grid):
            break
        else:
            ems_point = np.interp(y, y_grid, ems)
            πᵈ_list = get_πᵈ(y)
            πᶜ_list = get_πᶜ(y)
            h_point = np.interp(y, y_grid, h)
            if ME is not None:
                me_point = np.interp(y, y_grid, ME)
                me_t[t] = me_point
            et[t] = ems_point
            πᵈt[t] = πᵈ_list
            πᶜt[t] = πᶜ_list
            ht[t] = h_point
            yt[t] = y
            dy = ems_point*np.mean(θ_list)*dt
            y = dy + y
    if ME is not None:
        simulation_res = dict(yt=yt, et=et, πct=πct, πdt=πdt, ht=ht, me_t=me_t)
    else:
        simulation_res = dict(yt=yt, et=et, πct=πct, πdt=πdt, ht=ht)
    return simulation_res


def simulate_me(y_grid, e_grid, ratio_grid, θ=1.86/1000., y_start=1, T=100, dt=1):
    """
    simulate trajectories of uncertainty decomposition
    
    .. math::

        \\log(\\frac{ME_{new}}{ME_{baseline}})\\times 1000.
    
    Parameters
    ----------
    y_grid : (N, ) ndarray
        Grid of y.
    e_grid : (N, ) ndarray
        Corresponding :math:`\\tilde{e}` on the grid of y.
    ratio_grid : (N, ) ndarray::
        Corresponding :math:`\\log(\\frac{ME_{new}}{ME_{baseline}})\\times 1000` on the grid of y.
    θ : float, default=1.86/1000
        Coefficient used for simulation.
    y_start : float
        Initial value of y.
    T : int, default=100
        Time span of simulation.
    dt : float, default=1
        Time interval of simulation. Default=1 indicates yearly simulation.
    
    Returns
    -------
    Et : (T, ) ndarray
        Emission trajectory.
    yt : (T, ) ndarray
        Temperature anomaly trajectories.
    ratio_t : (T, ) ndarray
        Uncertainty decomposition ratio trajectories.
    """
    periods = int(T/dt)
    Et = np.zeros(periods+1)
    yt = np.zeros(periods+1)
    ratio_t = np.zeros(periods+1)
    for i in range(periods+1):
        Et[i] = np.interp(y_start, y_grid, e_grid)
        ratio_t[i] = np.interp(y_start, y_grid, ratio_grid)
        yt[i] = y_start
        y_start = y_start + Et[i]*θ
    return Et, yt, ratio_t
