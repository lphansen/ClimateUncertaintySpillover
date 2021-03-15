import numpy as np
from utilities import compute_derivatives
from solver import false_transient_one_iteration_python
from solver_ode import solve_ode, derivative_1d


def ode_y(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_a, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid + γ_2p*(y_grid>y_bar)*(y_grid-y_bar)

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
            temp = σ_y**2*(v0_dyy-G**2/ξ_1m)
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
            + ξ_a*c_entropy
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


# intensity
def get_intensity(y_grid, ς, ȳ=2):
    temp = 1/(np.sqrt(np.pi*2)*ς)*np.exp(-(y_grid - ȳ)**2/(2*ς**2))
#     temp *= v_bar - v_new
    return temp


# solve for approach 2
def solve_jump(y_grid, args, ϵ=0.3, tol=1e-8):
    """compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, σy, γ1, γ2, γ3_list, dmg_weight, ς, ξₘ, ξₐ, ξ, ϕ_list = args
    # solve for HJB with jump function
    y_grid_cap = y_grid
    dΛ = γ1 + γ2*y_grid_cap
    ϕ = - δ*η*y_grid_cap**2
    dy = y_grid_cap[1] - y_grid_cap[0]
    ems = δ*η
    ems_old = ems
    episode = 0
    lhs_error = 1
    πᵈo = dmg_weight
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)
    while lhs_error > tol:
        ϕ_old = ϕ.copy()
        dϕdy = compute_derivatives(ϕ, 1, dy, central_diff=False)
        dϕdyy = compute_derivatives(ϕ, 2, dy, central_diff=False)
        # update control
        temp = dϕdy + (η-1)*dΛ
        weight = np.array([ - 1/ξₐ*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πᶜ = πᶜo*np.exp(weight)
        πᶜ[πᶜ <= 1e-15] = 1e-15
        πᶜ = πᶜ/np.sum(πᶜ, axis=0)
        # update control
        a = dϕdyy*σy**2  - 1/ξ*temp**2*σy**2
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
        g_list = np.array([np.exp(1/ξₘ*(ϕ - ϕ_list[i])) for i in range(len(γ3_list))])
        # coefficients
        A = -δ*np.ones(y_grid_cap.shape)
        By = (θ_list@πᶜ)*ems
        Cyy = ems**2*σy**2/2
        D = δ*η*np.log(ems) + θ_list@πᶜ*(η-1)*dΛ*ems\
        + ξₘ*get_intensity(y_grid_cap,ς)*(πᵈo@(1-g_list))\
        + ξₐ*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0) \
        - 1/(2*ξ)*temp**2*ems**2*σy**2
        # solver
        ϕ_new = solve_ode(A, By, Cyy, D, y_grid_cap, ϕ, ϵ, (False, 0))
        rhs = -δ*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
        ems_old = ems
        print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    
    h =  - temp*ems*σy/ξ
    ι, πᵈ = get_ι(πᵈo, g_list)
    penalty = ξₘ*get_intensity(y_grid_cap, ς)*(πᵈo@(1 - g_list))
    return ϕ, ems, πᶜ, ι, πᵈ, h, ϕ_list


def ode_y_jump_approach_one_new(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_p, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid

    temp = np.zeros((len(πd_o), len(y_grid)))
    for i in range(len(πd_o)):
        temp[i] = πd_o[i]
    πd_o = temp

    temp_1 = np.zeros((len(πc_o), len(y_grid)))
    temp_2 = np.zeros_like(temp_1)
    for i in range(len(πc_o)):
        temp_1[i] = πc_o[i]
        temp_2[i] = θ[i]
    πc_o = temp_1
    θ = temp_2

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()

        v0_dy = compute_derivatives(v0, 1, Δ_y, central_diff=True)
        v0_dyy = compute_derivatives(v0, 2, Δ_y)
        
        G = v0_dy + (η-1)*d_Λ

        if σ_y == 0:
            f_tilde = -δ*η/(G*np.sum(πc_o*θ, axis=0))
        else:
            temp_1 = σ_y**2*v0_dyy
            temp_2 = G*np.sum(πc_o*θ, axis=0)
            root = np.sum(πc_o*θ, axis=0)**2*G**2 - 4*δ*η*temp_1
            root[root<0] = 0.
            f_tilde = np.zeros_like(temp_1)
            for i in range(len(f_tilde)):
                if temp_1[i] == 0:
                    f_tilde[i] = -δ*η/temp_2[i]
                if temp_1[i] > 0:
                    f_tilde[i] = (-temp_2[i] + np.sqrt(root[i])) / (2*temp_1[i])
                else:
                    f_tilde[i] = (-temp_2[i] - np.sqrt(root[i])) / (2*temp_1[i])
        
        f_tilde[f_tilde<0] = 1e-12
        intensity = 1./np.sqrt(ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))

        A = np.ones_like(y_grid)*(-δ)
        B = f_tilde * np.sum(πc_o*θ, axis=0)
        C = .5 * σ_y**2 * f_tilde**2
        D = δ*η*np.log(f_tilde) + (η-1)*d_Λ*f_tilde*np.sum(πc_o*θ, axis=0)\
            + -ξ_p*intensity*( (np.sum(πd_o*np.exp(-1./ξ_p*ϕ_i), axis=0)-np.exp(-1./ξ_p*v_old)) / np.exp(-1./ξ_p*v_old) )
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
           'f_tilde': f_tilde,
           'd_Λ': d_Λ}
    return res


def ode_y_jump_approach_one(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_2m, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid

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
            temp = σ_y**2*(v0_dyy-G**2/ξ_1m)
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

        intensity = 1./(np.sqrt(2*np.pi)*ς)*np.exp(-(y_bar-y_grid)**2/(2*ς**2))
        g = np.exp(1./ξ_2m*(v0-ϕ_i))

        A = np.ones_like(y_grid)*(-δ) - intensity
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ*η*np.log(e_tilde) - C*G**2/ξ_1m + (η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + intensity*np.sum(πd_o*g*ϕ_i, axis=0) + ξ_2m*intensity*np.sum(πd_o*(1-g+g*np.log(g)), axis=0)\
            + ξ_a*c_entropy
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


def ode_y_jump_approach_two(y_grid, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_2m, ξ_a, ς, γ_1, γ_2, y_bar, ϕ_i, πd_o = model_paras
    Δ_y = y_grid[1] - y_grid[0]

    if v0 is None:
        v0 = -δ*η*(y_grid+y_grid**2)

    d_Λ = γ_1 + γ_2*y_grid

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
            temp = σ_y**2*(v0_dyy-G**2/ξ_1m)
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
            + ξ_a*c_entropy
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
