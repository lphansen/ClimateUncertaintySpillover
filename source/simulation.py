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
    model_res: a dictionary 
        A dictionary storing solution with misspecified jump process.
        v : (N,) ndarray
            Value function
        dvdy : (N,) ndarray
            First order derivative of the value function.
        dvddy : (N,) ndarray
            Second order derivative of the value function.
        e_tilde : (N,) ndarray
            :math:`\tilde{e}` on the grid of y.
        h : (N,) ndarray
            Implied drift distortion.
        πc : (M, N) ndarray
            Distorted probabilities of θ.
        g : (K, N) ndarray
            Change in damage probability and intensity.
        πd : (K, N) ndarray
            Distorted probabilities of damage functions.
        bc : float
            The boundary condition that we impose on the HJB.
        y : (N,) ndarray
            Grid of y.
        model_args : tuple
            Model parameters.
    θ_list: (N,) ndarray
        A list of matthew coefficients. Unit: celsius/gigaton of carbon.
    ME: (N,) ndarray
        Marginal value of emission as a function of y.
    y_start: float (Default: 1)
        Initial value of y.
    T: int (Default: 100)
        Time span of simulation.
    dt: float (Default: 1)
        Time interval of simulation.
        
    Returns
    -------
    simulation_res: a dictionary of ndarrays
        yt: (T,) ndarray
            Temperature trajectories.
        et: (T,) ndarray
            Emission trajectories.
        πct: (T, L) ndarray
            Trajectories for distorted probabilities of climate models.
        πdt: (T, M) ndarray
            Trajectories for distorted probabilities of damage functions.
        ht: (T,) ndarray
            Trajectories for drift distortion.
        if ME is not None, the dictionary will also include
            me_t: (T,) ndarray
                Trajectories for marginal value of emission.
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


def simulate_me(y_grid, e_grid, ratio_grid, θ=1.86/1000., y_start=1, T=100):
    """
    simulate log(ME_new/ME_baseline)*1000.
    
    Parameters
    ----------
    y_grid:
    e_grid:
    ratio_grid:
    θ:
    y_start:
    T
    
    Returns
    -------
    Et:
    yt:
    ratio_t:
    """
    Et = np.zeros(T+1)
    yt = np.zeros(T+1)
    ratio_t = np.zeros(T+1)
    for i in range(T+1):
        Et[i] = np.interp(y_start, y_grid, e_grid)
        ratio_t[i] = np.interp(y_start, y_grid, ratio_grid)
        yt[i] = y_start
        y_start = y_start + Et[i]*θ
    return Et, yt, ratio_t
