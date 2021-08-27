#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from src.derivative import derivatives_2d
from src.solver_2d import false_transient_one_iteration_cpp
import SolveLinSys

def pde_2d(stateSpace, A, B_r, B_f, C_rr, C_ff, D, v0, ε = 1, tol = -10, smartguess = False):

    A = A.reshape(-1,1,order = 'F')
    B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F')])
    C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F')])
    D = D.reshape(-1,1,order = 'F')
    v0 = v0.reshape(-1,1,order = 'F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

    return out


def solve_prep(y1_grid, y2_grid, γ3, θ_list, args=(), tol=1e-7, ϵ=1, max_iter=2000, fraction=0.05):
    δ, η, γ1, γ2, y_bar, λ, ξa = args
    # grid information
    (y1_mat, y2_mat) = np.meshgrid(y1_grid, y2_grid, indexing = 'ij')
    stateSpace = np.hstack([y1_mat.reshape(-1,1, order='F'), y2_mat.reshape(-1,1,order='F')])
    n_y1, n_y2 = y1_mat.shape
    hy1 = y1_grid[1] - y1_grid[0]
    hy2 = y2_grid[1] - y2_grid[0]
    stateSpace = np.hstack([y1_mat.reshape(-1,1, order='F'), y2_mat.reshape(-1,1,order='F')])
    # climate model prior
    πa_o = np.ones((len(θ_list), n_y1, n_y2))/len(θ_list)
    θ_mat = np.zeros((len(θ_list), n_y1, n_y2))
    for i in range(len(θ_list)):
        θ_mat[i] = θ_list[i]
    πa = πa_o
    dΛ1 = γ1 + γ2*y1_mat + γ3*(y1_mat - y_bar)*(y1_mat > y_bar)
    ems_new = η*np.ones(y1_mat.shape)
    ems_old = ems_new
    episode = 0
    lhs_error = 0.5
    while lhs_error > tol and episode  < max_iter:
        if episode ==0:
            v0 =  - η*((y1_mat+y2_mat) + (y1_mat+y2_mat)**2)
        else:
            vold = v0.copy()
        v0_dy1 = derivatives_2d(v0,0,1,hy1)
        v0_dy2 = derivatives_2d(v0,1,1,hy2)
        # updating controls
        ems_new =  - η/(v0_dy2*λ*np.sum(θ_mat*πa, axis=0))
        ems_new[ems_new <= 1e-15] = 1e-15
        ems = ems_new*fraction + ems_old*(1 - fraction)
        weight = np.array([-1/ξa*v0_dy2*λ*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πa = πa_o*np.exp(weight)
        πa[πa<1e-15] = 1e-15
        πa = πa/np.sum(πa, axis=0)
        # HJB coefficient
        A =  -δ*np.ones(y1_mat.shape)
        B_y1 =  y2_mat
        B_y2 = λ*( - y2_mat + ems*np.sum(θ_mat*πa, axis=0))
        C_yy1 = np.zeros(y1_mat.shape)
        C_yy2 = np.zeros(y1_mat.shape)
        D = η*np.log(ems) +  (η-1)/δ*dΛ1*y2_mat + ξa*np.sum(πa*(np.log(πa) - np.log(πa_o)), axis=0) 
        # PDE solver
        out = pde_2d(stateSpace, A, B_y1, B_y2, C_yy1, C_yy2, D, v0, ϵ)
        out_comp = out[2].reshape(v0.shape,order = "F")
        rhs = A*v0 + B_y1*v0_dy1 + B_y2*v0_dy2  + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((out_comp - v0)/ϵ))
        episode += 1
        v0 = out_comp
        ems_old = ems
        print("Episode {:d}: PDE Error: {:.12f}; False Transient Error: {:.12f}; Iterations: {:d}; CG Error: {:.12f}".format(episode, rhs_error, lhs_error, out[0], out[1]))
    result = dict(v0=v0, ems=ems, πa=πa, y1=y1_grid, y2=y2_grid, λ=λ)
    return result

def solve_pre_jump_2state(res_list, args=(), tol=1e-6, ε=1., max_iter=10_000, fraction=0.05):
    δ, η, θ_list,  γ1, γ2, γ3_list, ξa, ξp = args
    # get grid info
    res = res_list[0]
    λ = res["λ"]
    y1_grid = res["y1"]
    y2_grid = res["y2"]
    
    y1_step = y1_grid[1] - y1_grid[0]
    y2_step = y2_grid[1] - y2_grid[0]
    # get value function list
    φ_list = np.zeros((len(γ3_list), len(y1_grid), len(y2_grid)))
    for i in range(len(γ3_list)):
        φ_list[i] = res_list[i]["v0"]
        
    y1_grid_cap = np.arange(0., 2.1 + y1_step, y1_step)
    loc_2 = np.abs(y1_grid - 2.).argmin()
    # terminal value
    dmg_weight = np.ones(len(γ3_list)) / len(γ3_list)
    ϕ_weight = np.average(np.exp(-1 / ξp * φ_list), axis=0, weights=dmg_weight)
    ϕ_equiv = -ξp * np.log(ϕ_weight)

    (y1_mat_cap, y2_mat_cap) = np.meshgrid(y1_grid_cap, y2_grid, indexing='ij')
    stateSpace = np.hstack([y1_mat_cap.reshape(-1,1,order = 'F'),y2_mat_cap.reshape(-1,1,order = 'F')])

    num_y1 = len(y1_grid_cap)
    num_y2 = len(y2_grid)
    πd_o = np.ones((len(γ3_list), num_y1, num_y2)) / len(γ3_list)
    πa_o = np.ones((len(θ_list), num_y1, num_y2)) / len(θ_list)
    θ_mat = np.zeros((len(θ_list), num_y1, num_y2))
    for i in range(len(θ_list)):
        θ_mat[i] = θ_list[i]
    dΛ1 = γ1 + γ2 * y1_mat_cap

    r1 = 1.5
    r2 = 2.5
    y_lower = 1.5
    Intensity = r1 * (np.exp(r2 / 2 * (y1_mat_cap - y_lower)**2) -
                    1) * (y1_mat_cap >= y_lower)

    # initiate v and control
    ems = η
    ems_old = ems
    lhs_error = 1
    episode = 0
    v0 = ϕ_equiv[:num_y1]
    v_m = np.zeros(πd_o.shape)
    for i in range(len(γ3_list)):
        v_m[i] = φ_list[i][loc_2]

    while lhs_error > tol and episode < max_iter:
        v0_old = v0.copy()
        v0_dy1 = derivatives_2d(v0, 0, 1, y1_step)
        v0_dy2 = derivatives_2d(v0, 1, 1, y2_step)
        # updating controls
        weight = np.array([-1 / ξa * v0_dy2 * λ * ems_old * θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πa = πa_o * np.exp(weight)
        πa[πa < 1e-15] = 1e-15
        πa = πa / np.sum(πa, axis=0)
        ems_new = -η / (v0_dy2 * λ * np.sum(θ_mat * πa, axis=0))
        ems_new[ems_new <= 1e-15] = 1e-15
        ems = ems_new * 0.05 + ems_old * 0.95
        g_m = np.exp(1 / ξp * (v0 - v_m))
        g_m[g_m < 1e-15] = 1e-15
        # HJB coefficient
        A = -δ * np.ones(y1_mat_cap.shape) - Intensity * np.sum(πd_o * g_m, axis=0)
        B_y1 = y2_mat_cap
        B_y2 = λ * (-y2_mat_cap + ems * np.sum(θ_mat * πa, axis=0))
        C_yy1 = np.zeros(y1_mat_cap.shape)
        C_yy2 = np.zeros(y1_mat_cap.shape)
        D = η * np.log(ems) + (η - 1) / δ * dΛ1 * y2_mat_cap + ξa * np.sum(
            πa * (np.log(πa) - np.log(πa_o)), axis=0) + Intensity * np.sum(
                πd_o * g_m * v_m, axis=0) + ξp * Intensity * np.sum(
                    πd_o * (1 - g_m + g_m * np.log(g_m)), axis=0)
        phi_mat = false_transient_one_iteration_cpp(stateSpace,
            A,
            B_y1,
            B_y2,
            C_yy1,
            C_yy2,
            D,
            v0,
            ε)

        rhs = A * phi_mat + B_y1 * v0_dy1 + B_y2 * v0_dy2 + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((phi_mat - v0_old) / ε))
        v0 = phi_mat
        ems_old = ems
        episode += 1
        print('Episode: {:d}\t lhs error: {:.12f}\t rhs error: {:.12f}'.format(episode, lhs_error, rhs_error))
    πd = g_m / np.sum( πd_o * g_m, axis=0)
    res = {
        "v0": v0,
        "ems": ems,
        "y1": y1_grid_cap,
        "y2": y2_grid,
        "λ": λ,
        "πd": πd,
        "πa": πa
    }
    return res
