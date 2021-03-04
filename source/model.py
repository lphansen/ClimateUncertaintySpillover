import numpy as np
from utilities import compute_derivatives_2d
from solver import false_transient_one_iteration_cpp, false_transient_one_iteration_python


def hjb_modified(z_grid, y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000,
                                 use_python=False, bc=None, impose_bc=None):
    η, δ, η, μ_2, ρ, σ_2, λ_1, λ_2, λ_bar, λ_2p = model_paras
    Δ_z = z_grid[1] - z_grid[0]
    Δ_y = y_grid[1] - y_grid[0]
    (z_mat, y_mat) = np.meshgrid(z_grid, y_grid, indexing = 'ij')
    stateSpace = np.hstack([z_mat.reshape(-1, 1, order='F'), y_mat.reshape(-1, 1, order='F')])
    if v0 is None:
        v0 = -δ*η*y_mat

    d_Λ = λ_1 + λ_2*y_mat + λ_2p*(y_mat>λ_bar)*(y_mat-λ_bar)
    d_Λ_z = d_Λ * z_mat

    mean = - ρ*(z_mat-μ_2)
    std = np.sqrt(z_mat)*σ_2
    var = std**2/2.
    e = - δ*η / ((η-1)*d_Λ_z)
    e_old = e.copy()

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        v0_dz = compute_derivatives_2d(v0, 0, 1, Δ_z)
        v0_dzz = compute_derivatives_2d(v0, 0, 2, Δ_z)
        v0_dy = compute_derivatives_2d(v0, 1, 1, Δ_y)

        e_new = - δ*η / (v0_dy*z_mat + (η-1)*d_Λ_z)
        e_new[e_new<=0] = 1e-12
        e = e_new * 0.5 + e_old * 0.5
        e_old = e.copy()

        A = np.ones_like(z_mat)*(-δ)
        B_z = mean
        B_y = z_mat*e
        C_zz = var
        C_yy = np.zeros_like(z_mat)
        D = δ*η*np.log(e) + (η-1)*d_Λ_z*e
        if use_python:
            v0 = false_transient_one_iteration_python(A, B_z, B_y, C_zz, C_yy, D, v0, ε, Δ_z, Δ_y, bc, impose_bc)
        else:
            v0 = false_transient_one_iteration_cpp(stateSpace, A, B_z, B_y, C_zz, C_yy, D, v0, ε)

        rhs_error = A*v0 + B_z*v0_dz + B_y*v0_dy + C_zz*v0_dzz + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e


def hjb_modified_vanishing_viscosity(z_grid, y_grid, model_paras=(), v0=None, ϵ=.5, vc=.1, tol=1e-8, max_iter=10_000,
                                 use_python=False, bc=None, impose_bc=None):
    η, δ, η, μ_2, ρ, σ_2, λ_1, λ_2, λ_bar, λ_2p = model_paras
    Δ_z = z_grid[1] - z_grid[0]
    Δ_y = y_grid[1] - y_grid[0]
    (z_mat, y_mat) = np.meshgrid(z_grid, y_grid, indexing = 'ij')
    stateSpace = np.hstack([z_mat.reshape(-1, 1, order='F'), y_mat.reshape(-1, 1, order='F')])
    if v0 is None:
        v0 = -δ*η*y_mat

    d_Λ = λ_1 + λ_2*y_mat + λ_2p*(y_mat>λ_bar)*(y_mat-λ_bar)
    d_Λ_z = d_Λ * z_mat

    mean = - ρ*(z_mat-μ_2)
    std = np.sqrt(z_mat)*σ_2
    var = std**2/2.
    e = - δ*η / ((η-1)*d_Λ_z)
    e_old = e.copy()

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        v0_dz = compute_derivatives_2d(v0, 0, 1, Δ_z)
        v0_dzz = compute_derivatives_2d(v0, 0, 2, Δ_z)
        v0_dy = compute_derivatives_2d(v0, 1, 1, Δ_y)
        v0_dyy = compute_derivatives_2d(v0, 1, 2, Δ_y)

        e_new = - δ*η / (v0_dy*z_mat + (η-1)*d_Λ_z)
        e_new[e_new<=0] = 1e-12
        e = e_new * 0.5 + e_old * 0.5
        e_old = e.copy()

        A = np.ones_like(z_mat)*(-δ)
        B_z = mean
        B_y = z_mat*e
        C_zz = var
        C_yy = np.ones_like(z_mat)*vc
        D = δ*η*np.log(e) + (η-1)*d_Λ_z*e
        if use_python:
            v0 = false_transient_one_iteration_python(A, B_z, B_y, C_zz, C_yy, D, v0, ε, Δ_z, Δ_y, bc, impose_bc)
        else:
            v0 = false_transient_one_iteration_cpp(stateSpace, A, B_z, B_y, C_zz, C_yy, D, v0, ε)

        rhs_error = A*v0 + B_z*v0_dz + B_y*v0_dy + C_yy*v0_dyy + C_zz*v0_dzz + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e