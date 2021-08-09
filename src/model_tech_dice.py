import numpy as np
import pandas as pd
from .utilities_2d import compute_derivatives
from .solver_2d import false_transient_one_iteration_cpp
from numba import njit
from multiprocessing import Pool


@njit
def _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                   δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction):

    dvdk  = compute_derivatives(v0, 0, 1, dk, central_diff=True)
    dvdkk = compute_derivatives(v0, 0, 2, dk)
    dvdy  = compute_derivatives(v0, 1, 1, dy, central_diff=True)
    dvdyy = compute_derivatives(v0, 1, 2, dy)

    temp = α - i - α * vartheta_bar * (1 - e / (α * lambda_bar * np.exp(k_mat))) ** theta
    mc = 1. / temp

    i_new = - (mc / dvdk - 1) / κ

    G = dvdy  - 1. / δ * d_Λ
    F = dvdyy - 1. / δ * dd_Λ

    temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
    a = temp / (α * lambda_bar * np.exp(k_mat)) ** 2
    b = - 2 * temp / (α * lambda_bar * np.exp(k_mat))\
        + (F - G**2/ξ_b) * σ_y ** 2
    c = temp + G * np.sum(πc * θ, axis=0)

    # Method 1 : Solve second order equation
    if vartheta_bar != 0:
        temp = b ** 2 - 4 * a * c
        temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2
    else:
        e_new = c / (-b)

#     # Method 2 : Fix a and solve
#     e_new = (a * e**2 + c) / (-b)

    e_new = e_new * (e_new > 0) + 1e-8 * (e_new <= 0)
    
    i = i_new * fraction + i * (1-fraction)
    e = e_new * fraction + e * (1-fraction)

    log_πc_ratio = - G * e * θ / ξ_a
    πc_ratio = log_πc_ratio - np.max(log_πc_ratio)
    πc = np.exp(πc_ratio) * πc_o
    πc = πc / np.sum(πc, axis=0)
    πc = (πc <= 0) * 1e-16 + (πc > 0) * πc
    entropy = np.sum(πc * (np.log(πc) - np.log(πc_o)), axis=0)

    A = np.ones_like(y_mat) * (- δ)
    B_k = μ_k + i - κ / 2. * i ** 2 - σ_k ** 2 / 2.
    B_y = np.sum(πc * θ, axis=0) * e
    C_kk = σ_k ** 2 / 2 * np.ones_like(y_mat)
    C_yy = .5 * σ_y **2 * e**2

    D = np.log(1. / mc)\
        + k_mat - 1./δ * (d_Λ * np.sum(πc * θ, axis=0) * e + .5 * dd_Λ * σ_y ** 2 * e ** 2)\
        + ξ_a * entropy - C_yy * G**2 / ξ_b

    h = - G * e * σ_y / ξ_b

    return πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h


def hjb_post_damage_post_tech(k_grid, y_grid, model_args=(), v0=None, ϵ=1., fraction=.1,
                              tol=1e-8, max_iter=10_000, print_iteration=True):

    δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, γ_1, γ_2, γ_3, τ, theta, lambda_bar, vartheta_bar = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = κ * (1. / δ)
    b_i = - (1. + α * κ) * (1. / δ)
    c_i = α * (1. / δ) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / δ * k_mat -  y_mat ** 2

    d_Λ = γ_1 + γ_2 * y_mat + γ_3 * (y_mat > τ) * (y_mat - τ)
    dd_Λ = γ_2 + γ_3 * (y_mat > τ)

    πc_o = np.array([temp * np.ones_like(y_mat) for temp in πc_o])
    θ = np.array([temp * np.ones_like(y_mat) for temp in θ])
    πc = πc_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h = \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                           δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction)

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v': v,
           'e': e,
           'i': i,
           'πc': πc,
           'h': h}

    return res


def hjb_post_damage_pre_tech(k_grid, y_grid, model_args=(), v0=None, ϵ=1., fraction=.1,
                              tol=1e-8, max_iter=10_000, print_iteration=True):

    δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g, I_g, v_g, γ_1, γ_2, γ_3, τ, theta, lambda_bar, vartheta_bar = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = κ * (1. / δ)
    b_i = - (1. + α * κ) * (1. / δ)
    c_i = α * (1. / δ) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)
    
    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / δ * k_mat -  y_mat ** 2

    d_Λ = γ_1 + γ_2 * y_mat + γ_3 * (y_mat > τ) * (y_mat - τ)
    dd_Λ = γ_2 + γ_3 * (y_mat > τ)

    πc_o = np.array([temp * np.ones_like(y_mat) for temp in πc_o])
    θ = np.array([temp * np.ones_like(y_mat) for temp in θ])
    πc = πc_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h = \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                           δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction)
        
#         # Method 1:
#         D -= ξ_g * I_g * (np.exp(- v_g / ξ_g) - np.exp(- v0 / ξ_g)) / (np.exp(- v0 / ξ_g))
        
        # Method 2:
        g_tech = np.exp(1. / ξ_g * (v0 - v_g))
        A -= I_g * g_tech
        D += I_g * g_tech * v_g + ξ_g * I_g * (1 - g_tech + g_tech * np.log(g_tech))

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g_tech = np.exp(1. / ξ_g * (v - v_g))

    res = {'v': v,
           'e': e,
           'i': i,
           'g_tech': g_tech,
           'πc': πc,
           'h': h}

    return res


def hjb_pre_damage_post_tech(k_grid, y_grid, model_args=(), v0=None, ϵ=1., fraction=.1,
                             tol=1e-8, max_iter=10_000, print_iteration=True):

    δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_p, πd_o, v_i, γ_1, γ_2,\
        theta, lambda_bar, vartheta_bar, y_bar_lower = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = κ * (1. / δ)
    b_i = - (1. + α * κ) * (1. / δ)
    c_i = α * (1. / δ) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / δ * k_mat -  y_mat ** 2

    d_Λ = γ_1 + γ_2 * y_mat
    dd_Λ = γ_2

    πc_o = np.array([temp * np.ones_like(y_mat) for temp in πc_o])
    πd_o = np.array([temp * np.ones_like(y_mat) for temp in πd_o])
    θ = np.array([temp * np.ones_like(y_mat) for temp in θ])
    πc = πc_o.copy()

    r1=1.5
    r2=2.5
    intensity = r1*(np.exp(r2/2*(y_mat - y_bar_lower)**2)-1)*(y_mat >= y_bar_lower)

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h= \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                           δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction)
        
        D -= ξ_p * intensity * (np.sum(πd_o * np.exp(- v_i / ξ_p), axis=0) - np.exp(- v0 / ξ_p)) / np.exp(- v0 / ξ_p)

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g = np.exp(1. / ξ_p * (v - v_i))

    res = {'v': v,
           'e': e,
           'i': i,
           'g': g,
           'πc': πc,
           'h': h}

    return res


def hjb_pre_damage_pre_tech(k_grid, y_grid, model_args=(), v0=None, ϵ=1., fraction=.1,
                             tol=1e-8, max_iter=10_000, print_iteration=True):

    δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g, ξ_p, πd_o, v_i, I_g, v_g,\
        γ_1, γ_2, theta, lambda_bar, vartheta_bar, y_bar_lower = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')

    a_i = κ * (1. / δ)
    b_i = - (1. + α * κ) * (1. / δ)
    c_i = α * (1. / δ) - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = 1. / δ * k_mat -  y_mat ** 2

    d_Λ = γ_1 + γ_2 * y_mat
    dd_Λ = γ_2

    πc_o = np.array([temp * np.ones_like(y_mat) for temp in πc_o])
    πd_o = np.array([temp * np.ones_like(y_mat) for temp in πd_o])
    θ = np.array([temp * np.ones_like(y_mat) for temp in θ])
    πc = πc_o.copy()

    r1=1.5
    r2=2.5
    intensity = r1*(np.exp(r2/2*(y_mat - y_bar_lower)**2)-1)*(y_mat >= y_bar_lower)

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        πc, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h = \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Λ, dd_Λ, theta, lambda_bar, vartheta_bar,
                           δ, α, κ, μ_k, σ_k, πc_o, πc, θ, σ_y, ξ_a, ξ_b, i, e, fraction)

#         # Method 1:
#         D -= ξ_g * I_g * (np.exp(- v_g / ξ_g) - np.exp(- v0 / ξ_g)) / (np.exp(- v0 / ξ_g))
        
        # Method 2:
        g_tech = np.exp(1. / ξ_g * (v0 - v_g))
        A -= I_g * g_tech
        D += I_g * g_tech * v_g + ξ_g * I_g * (1 - g_tech + g_tech * np.log(g_tech))        
        
        D -= ξ_p * intensity * (np.sum(πd_o * np.exp(- v_i / ξ_p), axis=0) - np.exp(- v0 / ξ_p)) / np.exp(- v0 / ξ_p)

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g = np.exp(1. / ξ_p * (v - v_i))
    g_tech = np.exp(1. / ξ_g * (v - v_g))

    res = {'v': v,
           'e': e,
           'i': i,
           'g': g,
           'g_tech': g_tech,
           'πc': πc,
           'h': h}

    return res


def parallel_solve(fun, args_list):
    with Pool() as p:
        res_list = p.starmap(fun, args_list)
    return res_list

