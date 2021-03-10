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


def ode_y(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000):
    η, δ, θ, σ_y, ξ_m, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid + γ_2p*(y_grid>y_bar)*(y_grid-y_bar)

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives_1d(v0, 1, Δ_y, central_diff=True)
        v0_dyy = compute_derivatives_1d(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ
        
        if σ_y == 0:
            e_tilde = -δ*η/(G*θ)
        else:
            temp = σ_y**2*(v0_dyy-G**2/ξ_m)
            root = θ**2*G**2 - 4*δ*η*temp
            e_tilde = (-G*θ - np.sqrt(root)) / (2*temp)
        e_tilde[e_tilde<=0] = 1e-12

        A = np.ones_like(y_grid)*(-δ)
        B = e_tilde * θ
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_m + (η-1)*d_Λ*e_tilde*θ
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e_tilde












# NEED TO MODIFY
def ode_y_ambiguity(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000):
    η, δ, θ, σ_y, ξ_m, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid + γ_2p*(y_grid>y_bar)*(y_grid-y_bar)
    A = δ*η
    h = np.zeros_like(y_grid)
    e_tilde = np.zeros_like(y_grid)

    π_o = np.ones_like(y_grid)/len(y_grid)
    π = π_o.copy()

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        h_old = h.copy()
        e_tilde_old = e_tilde.copy()

        v0_dy = compute_derivatives_1d(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives_1d(v0, 2, Δ_y)

        temp = v0_dy + (η-1)*d_Λ
        B = temp * (θ+σ_y*h)
        C = v0_dyy*σ_y**2
        e_tilde = (abs(C)>1e-12) * (-B + np.sqrt(B**2-4*A*C)) / (2*np.where(C==0, 1e-12, C))\
                + (abs(C)<=1e-12) * (-A/B)
        e_tilde[e_tilde<=0] = 1e-12
        e_tilde = e_tilde * .5 + e_tilde_old * .5

        h = -(temp*e_tilde*σ_y)/ξ_m
        h = h * .5 + h_old * .5

        AA = np.ones_like(y_grid)*(-δ)
        BB = e_tilde * (θ+σ_y*h)
        CC = .5 * σ_y**2 * e_tilde**2
        DD = δ*η*np.log(e_tilde) + .5*ξ_m*h**2 + (η-1)*d_Λ*e_tilde*(θ+σ_y*h)
        v0 = false_transient_one_iteration_python(AA, BB, CC, DD, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = AA*v0 + BB*v0_dy + CC*v0_dyy + DD
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e_tilde


def ode_y_jump(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000):
    η, δ, θ, σ_y, ξ_m, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid
    A = δ*η
    h = np.zeros_like(y_grid)
    e_tilde = np.zeros_like(y_grid)

    π = np.ones((len(πd_o), len(y_grid)))
    for i in range(π.shape[0]):
        π[i] = πd_o[i]
    πd_o = π

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        h_old = h.copy()
        e_tilde_old = e_tilde.copy()

        v0_dy = compute_derivatives_1d(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives_1d(v0, 2, Δ_y)

        temp = v0_dy + (η-1)*d_Λ
        B = temp * (θ+σ_y*h)
        C = v0_dyy*σ_y**2
        e_tilde = (abs(C)>1e-12) * (-B + np.sqrt(B**2-4*A*C)) / (2*np.where(C==0, 1e-12, C))\
                + (abs(C)<=1e-12) * (-A/B)
        e_tilde[e_tilde<=0] = 1e-12
        e_tilde = e_tilde * .5 + e_tilde_old * .5

        h = -(temp*e_tilde*σ_y)/ξ_m
        h = h * .5 + h_old * .5

        density = 1./(np.sqrt(2*np.pi)*ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))

        AA = np.ones_like(y_grid)*(-δ) - density
        BB = e_tilde * (θ+σ_y*h)
        CC = .5 * σ_y**2 * e_tilde**2
        DD = δ*η*np.log(e_tilde) + .5*ξ_m*h**2 + (η-1)*d_Λ*e_tilde*(θ+σ_y*h)\
            - ξ_m*density*np.log(np.sum(πd_o*np.exp(1./ξ_m*(-ϕ_i)), axis=0))
        v0 = false_transient_one_iteration_python(AA, BB, CC, DD, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = AA*v0 + BB*v0_dy + CC*v0_dyy + DD
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0, e_tilde


def ode_z(z_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000):
    η, δ, ρ, μ_2, σ_2, ξ_m = model_paras
    Δ_z = z_grid[1] - z_grid[0]

    if v0 is None:
        v0 = -δ*η*(z_grid+z_grid**2)

    h = np.zeros_like(z_grid)

    count = 1
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        h_old = h.copy()

        v0_dz = compute_derivatives_1d(v0, 1, Δ_z, central_diff=False)
        v0_dzz = compute_derivatives_1d(v0, 2, Δ_z)

        h = -(v0_dz*np.sqrt(z_grid)*σ_2)/ξ_m
        h = h * .5 + h_old * .5

        AA = np.zeros_like(z_grid)
        BB = -ρ*(z_grid-μ_2) + np.sqrt(z_grid)*σ_2*h
        CC = .5 * σ_2**2 * z_grid
        DD = -δ*η*np.log(z_grid) + .5*ξ_m*h**2
        v0 = false_transient_one_iteration_python(AA, BB, CC, DD, v0, ε, Δ_z, (0, 0), (False, False))

        rhs_error = AA*v0 + BB*v0_dz + CC*v0_dzz + DD
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
        count += 1
    return v0

    
    
    
    
    
    
    
    
    
    