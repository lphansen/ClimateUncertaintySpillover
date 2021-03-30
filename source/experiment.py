import numpy as np
import pandas as pd
from model import ode_y, ode_y_jump_approach_one_boundary, uncertainty_decomposition
from utilities import find_nearest_value


def solve_value_function(ξ_w, ξ_p, ξ_a, damage_setting={'γ_2p': np.array([0, .0197*2*2, .3853*2]),
                                                        'πd_o': np.array([1./3, 1./3, 1./3])}):
    η = .032
    δ = .01

    θ = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    πc_o = np.ones_like(θ)/len(θ)
    σ_y = 1.2*np.mean(θ)

    y_bar = 2.
    γ_1 = 1.7675/10000
    γ_2 = .0022*2
    γ_2p = damage_setting['γ_2p']
    πd_o = damage_setting['πd_o']

    y_step = .02
    y_grid_long = np.arange(0., 4., y_step)
    y_grid_short = np.arange(0., 2+y_step, y_step)
    n_bar = find_nearest_value(y_grid_long, y_bar) + 1

    # Prepare ϕ_i
    model_res_list = []
    for γ_2p_i in γ_2p:
        model_paras = (η, δ, θ, πc_o, σ_y, ξ_w, ξ_a, γ_1, γ_2, γ_2p_i, y_bar) 
        model_res = ode_y(y_grid_long, model_paras, v0=None, ϵ=5.,
                           tol=1e-8, max_iter=5_000, print_all=False)
        model_res_list.append(model_res)

    ϕ_list = [res['v0'] for res in model_res_list]

    ϕ_list_short = []
    for ϕ_i in ϕ_list:
        temp = ϕ_i[:n_bar]
        ϕ_list_short.append(temp)
    ϕ_i = np.array(ϕ_list_short)

    # Compute ϕ
    ς = .25
    model_paras = (η, δ, θ, πc_o, σ_y, ξ_w, ξ_p, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o)
    model_res = ode_y_jump_approach_one_boundary(y_grid_short, model_paras, 
                                                 v0=np.average(ϕ_i, weights=πd_o, axis=0),
                                                 ϵ=5., tol=1e-8, max_iter=5_000, print_all=False)

    return model_res_list, model_res


def solve_alternative_ME(ξ_w, ξ_p, ξ_a, baseline_w, baseline_p, baseline_a,
                         damage_setting={'γ_2p': np.array([0, .0197*2*2, .3853*2]),
                                         'πd_o': np.array([1./3, 1./3, 1./3])}):
    η = .032
    δ = .01

    θ = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    πc_o = np.ones_like(θ)/len(θ)
    σ_y = 1.2*np.mean(θ)

    y_bar = 2.
    γ_1 = 1.7675/10000
    γ_2 = .0022*2
    γ_2p = damage_setting['γ_2p']
    πd_o = damage_setting['πd_o']

    y_step = .02
    y_grid_long = np.arange(0., 4., y_step)
    y_grid_short = np.arange(0., 2+y_step, y_step)
    n_bar = find_nearest_value(y_grid_long, y_bar) + 1

    # Prepare ϕ_i
    ϕ_list = []
    for γ_2p_i in γ_2p:
        model_paras = (η, δ, θ, πc_o, σ_y, ξ_w, ξ_a, γ_1, γ_2, γ_2p_i, y_bar) 
        model_res = ode_y(y_grid_long, model_paras, v0=None, ϵ=.5,
                           tol=1e-8, max_iter=5_000, print_all=False)
        ϕ_list.append(model_res['v0'])

    ϕ_list_short = []
    for ϕ_i in ϕ_list:
        temp = ϕ_i[:n_bar]
        ϕ_list_short.append(temp)
    ϕ_i = np.array(ϕ_list_short)

    # Compute ϕ
    ς = .25
    model_paras = (η, δ, θ, πc_o, σ_y, ξ_w, ξ_p, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o)
    model_res = ode_y_jump_approach_one_boundary(y_grid_short, model_paras, 
                                                 v0=np.average(ϕ_i, weights=πd_o, axis=0),
                                                 ϵ=.5, tol=1e-8, max_iter=5_000, print_all=False)

    ME_total = η/model_res['e_tilde']

    # Uncertainty decomposition
    model_paras_new = (η, δ, θ, πc_o, σ_y, ξ_w, ξ_p, ξ_a, γ_1, γ_2, ϕ_i, πd_o)
    
    if baseline_w:
        h = np.zeros_like(model_res['h'])
    else:
        h = None
    if baseline_p:
        bc = np.average(ϕ_i, weights=πd_o, axis=0)[-1]
    else:
        bc = None
    if baseline_a:
        πc = np.ones_like(model_res['πc'])/len(θ)
    else:
        πc = None

    model_res_new = uncertainty_decomposition(y_grid_short, model_paras_new,
                                              e_tilde=model_res['e_tilde'], 
                                              h=h, πc=πc, bc=bc,
                                              v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=False)  
    ME_part = model_res_new['ME']
    return ME_total, ME_part

