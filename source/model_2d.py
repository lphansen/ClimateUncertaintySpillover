import numpy as np
from utilities import compute_derivatives_2d
from solver_2d import false_transient_one_iteration_cpp, false_transient_one_iteration_python


def pde_hotelling(y_grid, b_grid, ell, model_paras=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_all=True):
    η, δ, θ, πc_o, σ_y, ξ_1m, ξ_a, γ_1, γ_2, γ_2p, y_bar = model_paras
    Δ_y = y_grid[1] - y_grid[0]
    Δ_b = b_grid[1] - b_grid[0]
    (y_mat, b_mat) = np.meshgrid(y_grid, b_grid, indexing = 'ij')
    stateSpace = np.hstack([y_mat.reshape(-1, 1, order='F'), b_mat.reshape(-1, 1, order='F')])
    if v0 is None:
        v0 = -δ*η*y_mat

    d_Λ = γ_1 + γ_2*y_mat + γ_2p*(y_mat>y_bar)*(y_mat-y_bar)
    πc = np.ones((len(πc_o), len(y_grid), len(b_grid)))
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
        v0_dy = compute_derivatives_2d(v0, 0, 1, Δ_y, central_diff=True)
        v0_dyy = compute_derivatives_2d(v0, 0, 2, Δ_y)
        v0_db = compute_derivatives_2d(v0, 1, 1, Δ_b, central_diff=True)
        
        G = v0_dy + b_mat*(η-1)*d_Λ
        if σ_y == 0:
            e_tilde = -b_mat*δ*η/(G*np.sum(πc*θ, axis=0)-ell)
        else:
            temp = σ_y**2*(v0_dyy-G**2/(b_mat*ξ_1m))
            root = (np.sum(πc*θ, axis=0)*G - ell)**2 - 4*b_mat*δ*η*temp
            root[root<0] = 0.
            e_tilde = (-(G*np.sum(πc*θ, axis=0)-ell) - np.sqrt(root)) / (2*temp)
#         e_tilde[e_tilde<=0] = 1e-16

        log_πc_ratio = -G*e_tilde*θ/(b_mat*ξ_a)
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc/np.sum(πc, axis=0)
        πc[πc<=0] = 1e-16
        c_entropy = np.sum(πc*(np.log(πc)-np.log(πc_o)), axis=0)        

        A = np.zeros_like(y_mat)
        B_y = e_tilde * np.sum(πc*θ, axis=0)
        B_b = -np.ones_like(y_mat)*δ*b_mat
        C_yy = .5 * σ_y**2 * e_tilde**2
        C_bb = np.zeros_like(y_mat)
        D = -ell*e_tilde + b_mat*δ*η*np.log(e_tilde) - C_yy*G**2/(b_mat*ξ_1m) + b_mat*(η-1)*d_Λ*e_tilde*np.sum(πc*θ, axis=0)\
            + b_mat*ξ_a*c_entropy
        v0 = false_transient_one_iteration_cpp(stateSpace, A, B_y, B_b, C_yy, C_bb, D, v0, ε)

        rhs_error = A*v0 + B_y*v0_dy + B_b*v0_db + C_yy*v0_dyy + D
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
           'v0_db': v0_db,
           'e_tilde': e_tilde,
           'πc': πc,
           'c_entropy': c_entropy,
           'd_Λ': d_Λ}     
    return res
