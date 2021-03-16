import numpy as np
from utilities import compute_derivatives
from solver import false_transient_one_iteration_python, solve_lienar_ode_python


def ode_y(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_a, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid + γ_2p*(y_grid>y_bar)*(y_grid-y_bar)
    dd_Λ = γ_2 + γ_2p*(y_grid>y_bar)

    πc = np.ones((len(πc_o), len(y_grid)))
    θ_reshape = np.ones_like(πc)
    for i in range(πc.shape[0]):
        πc[i] = πc_o[i]
        θ_reshape[i] = θ[i]
    πc_o = πc.copy()
    θ = θ_reshape

    count = 0
    error = 1.
    
    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            e_tilde = -δ*η/(G*np.sum(πc*θ, axis=0))
        else:
            temp = σ_y**2*(v0_dyy+(η-1.)*dd_Λ-G**2/ξ_1m)
            root = np.sum(πc*θ, axis=0)**2*G**2 - 4*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-G*np.sum(πc*θ, axis=0) - np.sqrt(root)) / (2*temp)
        e_tilde[e_tilde<=0] = 1e-16

        log_πc_ratio = -G*e_tilde*θ/ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)

        A = np.ones_like(y_grid)*(-δ)
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2 + ξ_a*c_entropy
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'e_tilde': e_tilde,
           'πc': πc,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}
    return res


def ode_z(z_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, ρ, μ_2, σ_2, ξ_1m = model_paras
    Δ_z = z_grid[1] - z_grid[0]

    if v0 is None:
        v0 = -δ*η*(z_grid+z_grid**2)

    h = np.zeros_like(z_grid)

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        h_old = h.copy()

        v0_dz = compute_derivatives(v0, 1, Δ_z, central_diff=True)
        v0_dzz = compute_derivatives(v0, 2, Δ_z)

        h = -(v0_dz*np.sqrt(z_grid)*σ_2)/ξ_1m
        h = h * .5 + h_old * .5

        A = np.zeros_like(z_grid)
        B = -ρ*(z_grid-μ_2) + np.sqrt(z_grid)*σ_2*h
        C = .5 * σ_2**2 * z_grid
        D = -δ*η*np.log(z_grid) + .5*ξ_1m*h**2
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_z, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dz + C*v0_dzz + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))
    return v0


def ode_y_jump_approach_one(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_2m, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid
    dd_Λ = γ_2

    πd = np.ones((len(πd_o), len(y_grid)))
    for i in range(πd.shape[0]):
        πd[i] = πd_o[i]
    πd_o = πd

    πc = np.ones((len(πc_o), len(y_grid)))
    θ_reshape = np.ones_like(πc)
    for i in range(πc.shape[0]):
        πc[i] = πc_o[i]
        θ_reshape[i] = θ[i]
    πc_o = πc.copy()
    θ = θ_reshape

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            e_tilde = -δ*η/(G*np.sum(πc*θ, axis=0))
        else:
            temp = σ_y**2*(v0_dyy+(η-1.)*dd_Λ-G**2/ξ_1m)
            root = np.sum(πc*θ, axis=0)**2*G**2 - 4*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-G*np.sum(πc*θ, axis=0) - np.sqrt(root)) / (2*temp)

        e_tilde[e_tilde<=0] = 1e-16

        log_πc_ratio = -G*e_tilde*θ/ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16        
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)

        intensity = 1./np.sqrt(ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))
        g = np.exp(1./ξ_2m*(v0-ϕ_i))

        A = np.ones_like(y_grid)*(-δ) - intensity
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + intensity*np.sum(πd_o*g*ϕ_i, axis=0) + ξ_2m*intensity*np.sum(πd_o*(1-g+g*np.log(g)), axis=0)\
            + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2 + ξ_a*c_entropy
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))     
    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'e_tilde': e_tilde,
           'πc': πc,
           'g': g,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}
    return res


def ode_y_jump_approach_one_boundary(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_2m, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid
    dd_Λ = γ_2

    πd = np.ones((len(πd_o), len(y_grid)))
    for i in range(πd.shape[0]):
        πd[i] = πd_o[i]
    πd_o = πd

    πc = np.ones((len(πc_o), len(y_grid)))
    θ_reshape = np.ones_like(πc)
    for i in range(πc.shape[0]):
        πc[i] = πc_o[i]
        θ_reshape[i] = θ[i]
    πc_o = πc.copy()
    θ = θ_reshape
    
    e_tilde = 0.
    
    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        e_tilde_old = e_tilde

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            e_tilde = -δ*η/(G*np.sum(πc*θ, axis=0))
        else:
            temp = σ_y**2*(v0_dyy+(η-1.)*dd_Λ-G**2/ξ_1m)
            root = np.sum(πc*θ, axis=0)**2*G**2 - 4*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-G*np.sum(πc*θ, axis=0) - np.sqrt(root)) / (2*temp)
        
        e_tilde[e_tilde<=0] = 1e-16
        e_tilde = e_tilde *.5 + e_tilde_old*.5

        log_πc_ratio = -G*e_tilde*θ/ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16        
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)

        intensity = 1./np.sqrt(ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))
        g = np.exp(1./ξ_2m*(v0-ϕ_i))

        A = np.ones_like(y_grid)*(-δ)
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2 + ξ_a*c_entropy\

        bc = -ξ_2m*np.log(np.sum(πd_o[:, -1]*np.exp(-1./ξ_2m*ϕ_i[:, -1])))

        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, bc), (False, True))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))     
    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'e_tilde': e_tilde,
           'πc': πc,
           'g': g,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}
    return res


def ode_y_jump_approach_two(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_2m, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid
    dd_Λ = γ_2

    πd = np.ones((len(πd_o), len(y_grid)))
    for i in range(πd.shape[0]):
        πd[i] = πd_o[i]
    πd_o = πd

    πc = np.ones((len(πc_o), len(y_grid)))
    θ_reshape = np.ones_like(πc)
    for i in range(πc.shape[0]):
        πc[i] = πc_o[i]
        θ_reshape[i] = θ[i]
    πc_o = πc.copy()
    θ = θ_reshape

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            e_tilde = -δ*η/(G*np.sum(πc*θ, axis=0))
        else:
            temp = σ_y**2*(v0_dyy+(η-1.)*dd_Λ-G**2/ξ_1m)
            root = np.sum(πc*θ, axis=0)**2*G**2 - 4*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-G*np.sum(πc*θ, axis=0) - np.sqrt(root)) / (2*temp)
        e_tilde[e_tilde<=0] = 1e-12

        log_πc_ratio = -G*e_tilde*θ/ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)

        intensity = 1./(np.sqrt(2*np.pi)*ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))
        temp = np.exp(1./ξ_2m*(v0-ϕ_i))
        g = temp/np.sum(πd_o*temp, axis=0)

        A = np.ones_like(y_grid)*(-δ) - intensity
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + intensity*np.sum(πd_o*g*ϕ_i, axis=0) + ξ_2m*intensity*np.sum(πd_o*g*np.log(g), axis=0)\
            + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2 + ξ_a*c_entropy
        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, 0), (False, False))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))    
    
    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'e_tilde': e_tilde,
           'πc': πc,
           'g': g,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}

    return res


def solve_with_emission_boundary(y_grid, model_paras=()):
    η, δ, θ, πc_o, σ_y, γ_1, γ_2, ϕ_i, πd_o, e_tilde = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    d_Λ = γ_1 + γ_2*y_grid
    dd_Λ = γ_2

    A = np.ones_like(y_grid)*(-δ)
    B = (πc_o@θ)*e_tilde
    C = .5*σ_y**2*e_tilde**2
    D = δ*η*np.log(e_tilde) + (η-1)*d_Λ*e_tilde*(πc_o@θ) + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2

    bc = np.average(ϕ_i, axis=0, weights=πd_o)
    bc = bc[-1]

    v0 = solve_lienar_ode_python(A, B, C, D, 0., Δ_y, (0, bc), (False, True))

    v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
    v0_dyy = compute_derivatives(v0, 2, Δ_y)

    error = np.max(abs(A*v0 + B*v0_dy + C*v0_dyy + D))
    
    ME = -(v0_dy+(η-1)*d_Λ)*(πc_o@θ) - (v0_dyy+(η-1)*dd_Λ)*σ_y**2*e_tilde

    print("Solved. PDE error: %s" % (error))  

    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'ME': ME}

    return res


def uncertainty_decomposition(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_w, ξ_p, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid
    dd_Λ = γ_2

    πd = np.ones((len(πd_o), len(y_grid)))
    for i in range(πd.shape[0]):
        πd[i] = πd_o[i]
    πd_o = πd

    πc = np.ones((len(πc_o), len(y_grid)))
    θ_reshape = np.ones_like(πc)
    for i in range(πc.shape[0]):
        πc[i] = πc_o[i]
        θ_reshape[i] = θ[i]
    πc_o = πc.copy()
    θ = θ_reshape
    
    e_tilde = 0.
    
    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        e_tilde_old = e_tilde

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=False)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)

        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            e_tilde = -δ*η/(G*np.sum(πc*θ, axis=0))
        else:
            temp = σ_y**2*(v0_dyy+(η-1.)*dd_Λ-G**2/ξ_w)
            root = np.sum(πc*θ, axis=0)**2*G**2 - 4*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-G*np.sum(πc*θ, axis=0) - np.sqrt(root)) / (2*temp)
        
        e_tilde[e_tilde<=0] = 1e-16
        e_tilde = e_tilde *.5 + e_tilde_old*.5

        log_πc_ratio = -G*e_tilde*θ/ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16        
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)

        intensity = 1./np.sqrt(ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))
        g = np.exp(1./ξ_p*(v0-ϕ_i))

        A = np.ones_like(y_grid)*(-δ)
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_w + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + .5*(η-1)*dd_Λ*σ_y**2*e_tilde**2 + ξ_a*c_entropy\

        bc = -ξ_p*np.log(np.sum(πd_o[:, -1]*np.exp(-1./ξ_p*ϕ_i[:, -1])))

        v0 = false_transient_one_iteration_python(A, B, C, D, v0, ε, Δ_y, (0, bc), (False, True))

        rhs_error = A*v0 + B*v0_dy + C*v0_dyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v0 - v_old)/ϵ))
        error = lhs_error
        count += 1
        if print_all:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))     
    res = {'v0': v0,
           'v0_dy': v0_dy,
           'v0_dyy': v0_dyy,
           'e_tilde': e_tilde,
           'πc': πc,
           'g': g,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}
    return res


