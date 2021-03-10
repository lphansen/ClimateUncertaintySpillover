import numpy as np
from utilities import compute_derivatives
from solver import false_transient_one_iteration_python


def ode_y(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, σ_y, ξ_1m, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid + γ_2p*(y_grid>y_bar)*(y_grid-y_bar)

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=True)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ
        
        if σ_y == 0:
            e_tilde = -δ*η/(G*θ)
        else:
            temp = σ_y**2*(v0_dyy-G**2/ξ_1m)
            e_tilde = (-G*θ - np.sqrt(θ**2*G**2 - 4*δ*η*temp)) / (2*temp)
        e_tilde[e_tilde<=0] = 1e-12

        A = np.ones_like(y_grid)*(-δ)
        B = e_tilde * θ
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*θ
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
    return v0, e_tilde


def ode_z(z_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, ρ, μ_2, σ_2, ξ_1m = model_paras
    Δ_z = z_grid[1] - z_grid[0]

    if v0 is None:
        v0 = -δ*η*(z_grid+z_grid**2)

    h = np.zeros_like(z_grid)

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        h_old = h.copy()

        v0_dz = compute_derivatives(v0, 1, Δ_z)
        v0_dzz = compute_derivatives(v0, 2, Δ_z)

        h = -(v0_dz*np.sqrt(z_grid)*σ_2)/ξ_1m
        h = h * .5 + h_old * .5

        AA = np.zeros_like(z_grid)
        BB = -ρ*(z_grid-μ_2) + np.sqrt(z_grid)*σ_2*h
        CC = .5 * σ_2**2 * z_grid
        DD = -δ*η*np.log(z_grid) + .5*ξ_1m*h**2
        v0 = false_transient_one_iteration_python(AA, BB, CC, DD, v0, ε, Δ_z, (0, 0), (False, False))

        rhs_error = AA*v0 + BB*v0_dz + CC*v0_dzz + DD
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
    return v0


def ode_y_jump_approach_one(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, σ_y, ξ_1m, ξ_2m, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid

    π = np.ones((len(πd_o), len(y_grid)))
    for i in range(π.shape[0]):
        π[i] = πd_o[i]
    πd_o = π

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=True)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ
        
        if σ_y == 0:
            e_tilde = -δ*η/(G*θ)
        else:
            temp = σ_y**2*(v0_dyy-G**2/ξ_1m)
            e_tilde = (-G*θ - np.sqrt(θ**2*G**2 - 4*δ*η*temp)) / (2*temp)
        e_tilde[e_tilde<=0] = 1e-12

        density = 1./(np.sqrt(2*np.pi)*ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))

        A = np.ones_like(y_grid)*(-δ) - density
        B = e_tilde * θ
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*θ\
            - ξ_2m*density*np.log(np.sum(πd_o*np.exp(1./ξ_2m*(-ϕ_i)), axis=0))
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))        
    return v0, e_tilde
