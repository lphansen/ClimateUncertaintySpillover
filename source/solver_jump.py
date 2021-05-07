# -*- coding: utf-8 -*-
"""
module for jump model and uncertainty decomposition
"""
# packages
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
from utilities import dLambda
from supportfunctions import PDESolver_2d, finiteDiff
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from derivative import derivatives_2d, derivative_1d
import SolveLinSys
from solver_ode import solve_ode, solve_ode_one
from scipy import interpolate
# function
# intensity
def get_intensity(y_grid, ς, ȳ=2):
    temp = 1/(np.sqrt(np.pi*2)*ς)*np.exp(-(y_grid - ȳ)**2/(2*ς**2))
#     temp *= v_bar - v_new
    return temp

# compute proportional and distorted probability
def get_ι(πᵈo, g_list):
    ι = πᵈo@g_list
    n_dmg = len(πᵈo)
    π_star = np.array([g_list[i]*πᵈo[i]/ι for i in range(n_dmg)])
    return ι, π_star

# solve for approach two, step one, for individual ϕⱼ
def solve_smooth(y_grid, args, max_iter, tol, ϵ,):
    """
    solve for step one, ϕⱼ for individual damage function
    
    Parameter
    ---
    args: δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw
    """
    δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw = args
    dy = y_grid[1] - y_grid[0]
    n_y = len(y_grid)
    dΛ = γ1 + γ2*y_grid + γ3*(y_grid-ȳ)*(y_grid>ȳ)
    ddΛ = γ2 + γ3*(y_grid>ȳ)
    ϕ = - δ*η*y_grid**2
    ems = δ*η
    ems_old = ems
    πo = np.ones((len(θ_list), n_y))/len(θ_list)
    lhs_error = 1
    episode = 0
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, "up")
        dϕdyy = derivative_1d(ϕ, 2, dy, "up")
        temp = dϕdy + (η-1)*dΛ
        # update belief
        weight = np.array([ - 1/ξa*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        π = πo*np.exp(weight)
        π[π <= 1e-15] = 1e-15
        π = π/np.sum(π, axis=0)
        # update control
        a = (dϕdyy - 1/ξw*temp**2 + (η-1)*ddΛ)*σy**2
        b = (θ_list@π)*temp
        c = δ*η
        Δ = b**2 - 4*a*c
        Δ[Δ < 0] = 0
        root1 = (-b - np.sqrt(Δ))/(2*a)
        root2 = (-b + np.sqrt(Δ))/(2*a)
        if root1.all() > 0:
            ems_new = root1
        else:
            ems_new = root2
        ems_new[ems_new < 1e-15] = 1e-15
        ems = ems_new*0.5 + ems_old*0.5
        # solve for ode
        A = - δ*np.ones(y_grid.shape)
        B = (θ_list@π)*ems
#         C = np.zeros(y_grid.shape)
        C = ems**2*σy**2/2
        D = δ*η*np.log(ems) + (θ_list@π)*(η-1)*dΛ*ems + ξa*np.sum(π*(np.log(π) - np.log(πo)), axis=0)\
        - 1/(2*ξw)*temp**2*ems**2*σy**2\
        + 1/2*(η-1)*ddΛ*ems**2*σy**2
        ϕ_new = solve_ode(A, B, C, D, y_grid, ϕ, ϵ, (False, 0))
        rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        ems_old = ems
        episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    h = - temp*ems*σy/ξw
    return ϕ, ems, π, h


# + ξₘ*get_intensity(y_grid_cap,ς)*(πᵈo@(1 - g_list))\
# solve for approach 2, step two
def solve_jump(y_grid, numy_bar, ϕ_list, args, ϵ, tol, max_iter):
    """
    compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ς, ξp, ξa, ξw, σy = args
    # solve for HJB with jump function
    bound = np.average(np.exp(-1/ξp*ϕ_list), axis=0, weights=dmg_weight)
    bound = -ξp*np.log(bound)
    y_grid_cap = y_grid[:numy_bar+1]
    dΛ = γ1 + γ2*y_grid_cap
    ddΛ = γ2
    ϕ = np.average(ϕ_list, axis=0, weights=dmg_weight)[:numy_bar+1]
    dy = y_grid_cap[1] - y_grid_cap[0]
    ems = δ*η
    ems_old = ems
    episode = 0
    lhs_error = 1
    πᵈo = dmg_weight
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, "up")
        dϕdyy = derivative_1d(ϕ, 2, dy, "up")
        # update control
        temp = dϕdy + (η-1)*dΛ 
        weight = np.array([ - 1/ξa*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πᶜ = πᶜo*np.exp(weight)
        πᶜ[πᶜ <= 1e-15] = 1e-15
        πᶜ = πᶜ/np.sum(πᶜ, axis=0)
        # update control
        a = (dϕdyy  - 1/ξw*temp**2 + (η-1)*ddΛ)*σy**2
        b = (θ_list@πᶜ)*temp
        c = δ*η
        Δ = b**2 - 4*a*c
        Δ[Δ < 0] = 0
        root1 = (-b - np.sqrt(Δ))/(2*a)
        root2 = (-b + np.sqrt(Δ))/(2*a)
        if root1.all() > 0:
            ems_new = root1
        else:
            ems_new = root2
        ems_new[ems_new < 1e-15] = 1e-15
        ems = ems_new*0.5 + ems_old*0.5
        g_list = np.array([np.exp(1/ξp*(ϕ - ϕ_list[i][:numy_bar+1])) for i in range(len(γ3_list))])
        # coefficients
        A = -δ*np.ones(y_grid_cap.shape)
        By = (θ_list@πᶜ)*ems
        Cyy = ems**2*σy**2/2
        D = δ*η*np.log(ems) + θ_list@πᶜ*(η-1)*dΛ*ems\
        + ξa*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0) \
        - 1/(2*ξw)*temp**2*ems**2*σy**2\
        + 1/2*(η-1)*ddΛ*ems**2*σy**2
        # solver
        ϕ_new = solve_ode(A, By, Cyy, D, y_grid_cap, ϕ, ϵ, (True, bound[numy_bar]))
        rhs = -δ*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
        ems_old = ems
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    dϕdy = derivative_1d(ϕ, 1, dy, "up")
    dϕdyy = derivative_1d(ϕ, 2, dy, "up")
    temp = dϕdy + (η-1)*dΛ 
    h =  - temp*ems*σy/ξw
    ι, πᵈ = get_ι(πᵈo, g_list)
    penalty = ξp*get_intensity(y_grid_cap, ς)*(πᵈo@(1 - g_list))
    solution = dict(ϕ=ϕ, ems=ems, πc=πᶜ, ι=ι, πd=πᵈ, h=h)
    return solution


# -

# solve for approach one
def approach_one(y_grid, numy_bar, args, ϵ=0.3, tol=1e-8, max_iter=10_000):
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ς, ξp, ξa, ξw, σy = args
    ϕ_list = list()
    for γ3 in γ3_list:
        args_post = (δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw)
        ϕ, _, _ , _ = solve_smooth(y_grid, args_post, max_iter, tol, ϵ)
        ϕ_list.append(ϕ)
    ϕ_list = np.array(ϕ_list)
    solution = solve_jump(y_grid, numy_bar, ϕ_list, args, ϵ, tol, max_iter)
    return solution, ϕ_list


# solve for approach two, step one, for individual ϕⱼ
def solve_smooth_100(y_grid, args, max_iter, tol, ϵ,):
    """
    solve for step one, ϕⱼ for individual damage function
    
    Parameter
    ---
    args: δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw
    """
    δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw = args
    dy = y_grid[1] - y_grid[0]
    n_y = len(y_grid)
    dΛ = γ1 + γ2*y_grid + γ3*(y_grid-ȳ)*(y_grid>ȳ)
    ddΛ = γ2 + γ3*(y_grid>ȳ)
    ϕ = - η*y_grid**2
    ems = η
    ems_old = ems
    πo = np.ones((len(θ_list), n_y))/len(θ_list)
    lhs_error = 1
    episode = 0
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, "up")
        dϕdyy = derivative_1d(ϕ, 2, dy, "up")
        temp = dϕdy + (η-1)/δ*dΛ
        # update belief
        weight = np.array([ - 1/ξa*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        π = πo*np.exp(weight)
        π[π <= 1e-15] = 1e-15
        π = π/np.sum(π, axis=0)
        # update control
        a = (dϕdyy - 1/ξw*temp**2 + (η-1)/δ*ddΛ)*σy**2
        b = (θ_list@π)*temp
        c = η
        Δ = b**2 - 4*a*c
        Δ[Δ < 0] = 0
        root1 = (-b - np.sqrt(Δ))/(2*a)
        root2 = (-b + np.sqrt(Δ))/(2*a)
        if root1.all() > 0:
            ems_new = root1
        else:
            ems_new = root2
        ems_new[ems_new < 1e-15] = 1e-15
        ems = ems_new*0.5 + ems_old*0.5
        # solve for ode
        A = - δ*np.ones(y_grid.shape)
        B = (θ_list@π)*ems
#         C = np.zeros(y_grid.shape)
        C = ems**2*σy**2/2
        D = η*np.log(ems) + (θ_list@π)*(η-1)/δ*dΛ*ems + ξa*np.sum(π*(np.log(π) - np.log(πo)), axis=0)\
        - 1/(2*ξw)*temp**2*ems**2*σy**2\
        + 1/2*(η-1)/δ*ddΛ*ems**2*σy**2
        ϕ_new = solve_ode(A, B, C, D, y_grid, ϕ, ϵ, (False, 0))
        rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        ems_old = ems
        episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    h = - temp*ems*σy/ξw
    solution = dict(ϕ=ϕ, ems=ems, πc=π)
    return solution


def solve_jump_100(y_grid, numy_bar, ϕ_list, args, ϵ, tol, max_iter):
    """
    compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ς, ξp, ξa, ξw, σy = args
    # solve for HJB with jump function
    bound = np.average(np.exp(-1/ξp*ϕ_list), axis=0, weights=dmg_weight)
    bound = -ξp*np.log(bound)
    y_grid_cap = y_grid[:numy_bar+1]
    dΛ = γ1 + γ2*y_grid_cap
    ddΛ = γ2
    ϕ = np.average(ϕ_list, axis=0, weights=dmg_weight)[:numy_bar+1]
    dy = y_grid_cap[1] - y_grid_cap[0]
    ems = η
    ems_old = ems
    episode = 0
    lhs_error = 1
    πᵈo = dmg_weight
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, "up")
        dϕdyy = derivative_1d(ϕ, 2, dy, "up")
        # update control
        temp = dϕdy + (η-1)/δ*dΛ 
        weight = np.array([ - 1/ξa*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πᶜ = πᶜo*np.exp(weight)
        πᶜ[πᶜ <= 1e-15] = 1e-15
        πᶜ = πᶜ/np.sum(πᶜ, axis=0)
        # update control
        a = (dϕdyy  - 1/ξw*temp**2 + (η-1)/δ*ddΛ)*σy**2
        b = (θ_list@πᶜ)*temp
        c = η
        Δ = b**2 - 4*a*c
        Δ[Δ < 0] = 0
        root1 = (-b - np.sqrt(Δ))/(2*a)
        root2 = (-b + np.sqrt(Δ))/(2*a)
        if root1.all() > 0:
            ems_new = root1
        else:
            ems_new = root2
        ems_new[ems_new < 1e-15] = 1e-15
        ems = ems_new*0.5 + ems_old*0.5
        g_list = np.array([np.exp(1/ξp*(ϕ - ϕ_list[i][:numy_bar+1])) for i in range(len(γ3_list))])
        # coefficients
        A = -δ*np.ones(y_grid_cap.shape)
        By = (θ_list@πᶜ)*ems
        Cyy = ems**2*σy**2/2
        D = η*np.log(ems) + θ_list@πᶜ*(η-1)/δ*dΛ*ems\
        + ξa*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0) \
        - 1/(2*ξw)*temp**2*ems**2*σy**2\
        + 1/2*(η-1)/δ*ddΛ*ems**2*σy**2
        # solver
        ϕ_new = solve_ode(A, By, Cyy, D, y_grid_cap, ϕ, ϵ, (True, bound[numy_bar]))
        rhs = -δ*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
        ems_old = ems
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    dϕdy = derivative_1d(ϕ, 1, dy, "up")
    dϕdyy = derivative_1d(ϕ, 2, dy, "up")
    temp = dϕdy + (η-1)/δ*dΛ 
    h =  - temp*ems*σy/ξw
    ι, πᵈ = get_ι(πᵈo, g_list)
    penalty = ξp*get_intensity(y_grid_cap, ς)*(πᵈo@(1 - g_list))
    solution = dict(ϕ=ϕ, ems=ems, πc=πᶜ, ι=ι, πd=πᵈ, h=h, bound=bound)
    return solution


# solve for approach one
def approach_one_100(y_grid, numy_bar, args, report_π=False, ϵ=0.3, tol=1e-8, max_iter=10_000):
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ς, ξp, ξa, ξw, σy = args
    ϕ_list = list()
    π_list = list()
    ems_list = list()
    for γ3 in γ3_list:
        args_post = (δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw)
        ϕ, ems, π , _ = solve_smooth_100(y_grid, args_post, max_iter, tol, ϵ)
        ϕ_list.append(ϕ)
        π_list.append(π)
        ems_list.append(ems)
    ϕ_list = np.array(ϕ_list)
    π_list = np.array(π_list)
    solution = solve_jump_100(y_grid, numy_bar, ϕ_list, args, ϵ, tol, max_iter)
    if report_π == True:
        return solution, ϕ_list, π_list
    else:
        return solution, ϕ_list


# solve for approach one
def value_for_match(y_grid, args, report_π=False, ϵ=0.3, tol=1e-8, max_iter=5_000):
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ς, ξp, ξa, ξw, σy = args
    ϕ_list = list()
    π_list = list()
    ems_list = list()
    for γ3 in γ3_list:
        args_post = (δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw)
        ϕ, ems, π , _ = solve_smooth_100(y_grid, args_post, max_iter, tol, ϵ)
        ϕ_list.append(ϕ)
        π_list.append(π)
        ems_list.append(ems)
    ϕ_list = np.array(ϕ_list)
    π_list = np.array(π_list)
    ems_list = np.array(ems_list)
    if report_π == True:
        return ϕ_list, π_list
    else:
        return ϕ_list, ems_list


def generate_weight(params, ems, dϕdy, y_grid, args_weight=()):
    δ, η, γ1, γ2, ȳ, σy, ξa = args_weight
    num, _ = params.shape
    n_y = len(y_grid)
    weight =  np.zeros((num, n_y))
    for i in range(num):
        θ, γ3 = params[i]
        dΛ = γ1 + γ2*y_grid + γ3*(y_grid-ȳ)*(y_grid>ȳ)
        ddΛ = γ2 + γ3*(y_grid>ȳ)
        temp = dϕdy*ems*θ + (η-1)/δ*dΛ*ems*θ + 1/2*(η-1)/δ*ddΛ*σy**2*ems**2
        weight[i] = -1/ξa*temp
    weight = weight - np.max(weight, axis=0)
    return weight


def solve_smooth_unreveal(y_grid, args, max_iter, tol, ϵ,):
    """
    solve for step one, ϕⱼ for individual damage function
    
    Parameter
    ---
    args: δ, η, θ_list, σy, γ1, γ2, γ3, ȳ, ξa, ξw
    """
    δ, η, θ_list, σy, γ1, γ2, γ3_list, ȳ, ξa, ξw = args
    params = list()
    for θ_i in θ_list:
        for γ3 in γ3_list:
            params.append([θ_i, γ3])
    params = np.array(params)
    θ_long = params[:,0]
    γ3_long = params[:,1]
    args_weight = (δ, η, γ1, γ2, ȳ, σy, ξa)
    dy = y_grid[1] - y_grid[0]
    n_y = len(y_grid)
#     dΛ = γ1 + γ2*y_grid + γ3*(y_grid-ȳ)*(y_grid>ȳ)
#     ddΛ = γ2 + γ3*(y_grid>ȳ)
    ϕ = - η*y_grid**2
    ems = η
    ems_old = ems
    πo = np.ones((len(params), n_y))/len(params)
    lhs_error = 1
    episode = 0
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, "up")
        dϕdyy = derivative_1d(ϕ, 2, dy, "up")
#         temp1 = dϕdy + (η-1)/δ*dΛ
#         temp2 = dϕdyy + (de)
        # update belief
        weight = generate_weight(params, ems, dϕdy, y_grid, args_weight)
        π = πo*np.exp(weight)
        π[π <= 1e-15] = 1e-15
        π = π/np.sum(π, axis=0)
        # update control
        temp = dϕdy + (η-1)/δ*(γ1 + γ2*y_grid)
        ddΛ = γ2 + γ3_long@π*(y_grid>ȳ)
        a = (dϕdyy  + (η-1)/δ*ddΛ)*σy**2
        b = (θ_long@π)*temp + (η-1)/δ*((θ_long*γ3_long)@π)*(y_grid - ȳ)*(y_grid>ȳ)
        c = η
        Δ = b**2 - 4*a*c
        Δ[Δ < 0] = 0
        root1 = (-b - np.sqrt(Δ))/(2*a)
        root2 = (-b + np.sqrt(Δ))/(2*a)
        if root1.all() > 0:
            ems_new = root1
        else:
            ems_new = root2
        ems_new[ems_new < 1e-15] = 1e-15
        ems = ems_new*0.5 + ems_old*0.5
        # solve for ode
        dmg_drift = θ_long@π*(γ1 + γ2*y_grid)*ems + (θ_long*γ3_long)@π*(y_grid-ȳ)*(y_grid>ȳ)*ems\
        + 1/2*(γ3_long@π)*(y_grid>ȳ)*σy**2*ems**2 + 1/2*γ2*σy**2*ems**2
        A = - δ*np.ones(y_grid.shape)
        B = (θ_long@π)*ems
#         C = np.zeros(y_grid.shape)
        C = ems**2*σy**2/2
        D = η*np.log(ems) + ξa*np.sum(π*(np.log(π) - np.log(πo)), axis=0)\
            + (η-1)/δ*dmg_drift
#         - 1/(2*ξw)*temp**2*ems**2*σy**2\
# + 1/2*(η-1)/δ*ddΛ*ems**2*σy**2
        ϕ_new = solve_ode(A, B, C, D, y_grid, ϕ, ϵ, (False, 0))
        rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        ems_old = ems
        episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    h = - temp*ems*σy/ξw
    return ϕ, ems, π
