import numpy as np
from utilities import compute_derivatives_1d
from solver_1d import false_transient_one_iteration_python


def hjb_modified(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000,
                 bc=None, impose_bc=None):
    η, δ, μ_2, λ_1, λ_2, λ_bar, λ_2p = model_paras
    Δ_y = y_grid[1] - y_grid[0]
    y_mat = y_grid

    if v0 is None:
        v0 = -δ*η*y_mat

    d_Λ = λ_1 + λ_2*y_mat + λ_2p*(y_mat>λ_bar)*(y_mat-λ_bar)
    d_Λ_z = d_Λ * μ_2

    e = - δ*η / ((η-1)*d_Λ_z)
    e_old = e.copy()

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        v0_dy = compute_derivatives_1d(v0, 1, Δ_y)

        e_new = - δ*η / (v0_dy*μ_2 + (η-1)*d_Λ_z)
        e_new[e_new<=0] = 1e-12
        e = e_new * 0.5 + e_old * 0.5
        e_old = e.copy()

        A = np.ones_like(y_mat)*(-δ)
        B = μ_2*e
        C = np.zeros_like(y_mat)
        D = δ*η*np.log(e) + (η-1)*d_Λ_z*e
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, bc, impose_bc)

        rhs_error = A*v0 + B*v0_dy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e



def hjb_modified_jump(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000):
    η, δ, μ_2, λ_1, λ_2, λ_bar, σ, ϕ_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]
    y_mat = y_grid

    if v0 is None:
        v0 = -δ*η*y_mat

    d_Λ = λ_1 + λ_2*y_mat
    d_Λ_z = d_Λ * μ_2

    e = - δ*η / ((η-1)*d_Λ_z)
    e_old = e.copy()

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        v0_dy = compute_derivatives_1d(v0, 1, Δ_y)

        e_new = - δ*η / (v0_dy*μ_2 + (η-1)*d_Λ_z)
        e_new[e_new<=0] = 1e-12
        e = e_new * 0.5 + e_old * 0.5
        e_old = e.copy()
        
        density = 1./(np.sqrt(2*np.pi)*σ)*np.exp(-(λ_bar-y_mat)**2/(2*σ**2))
        A = np.ones_like(y_mat)*(-δ) - density
        B = μ_2*e
        C = np.zeros_like(y_mat)
        D = δ*η*np.log(e) + (η-1)*d_Λ_z*e + density*ϕ_bar
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e

