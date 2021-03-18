"""
Functions that solve the HJBs in the draft paper.

"""
import numpy as np
from utilities import compute_derivatives
from solver import false_transient


def solve_hjb_y(y, model_args=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_iteration=True):
    r"""
    Solve the HJB that is only related to y.
    y is the accumulative change of temperature.

    Parameters
    ----------
    y : (N,) ndarray
        An evenly spaced grid of y.
    model_args : tuple of model inputs
        η, δ, σ_y, y_bar, γ_1, γ_2, γ_2p, ξ_w, ξ_a : floats
        θ, πc_o : (M,) ndarrays
    v0 : (N,) ndarray
        Initial guess of the value function
    ϵ : float
        Step size of the false transient method.
    tol : float
        Tolerance level of the hjb solver.
    max_iter : int
        Maximum iterations of the false transient method.
    print_iteration : bool
        Print the information of each iteration if True.

    Returns
    -------
    res : dict of ndarrays
        v : (N,) ndarray
            Value function
        dvdy : (N,) ndarray
            First order derivative of the value function.
        dvddy : (N,) ndarray
            Second order derivative of the value function.
        e_tilde : (N,) ndarray
            :math:`\tilde{e}` on the grid of y.
        πc : (M, N) ndarray
            Distorted probabilities of θ.            
        h : (N,) ndarray
            Implied drift distortion.
        y : (N,) ndarray
            Grid of y.            
        model_args : tuple
            Model parameters.
    """
    η, δ, σ_y, y_bar, γ_1, γ_2, γ_2p, θ, πc_o, ξ_w, ξ_a = model_args
    dy = y[1] - y[0]

    if v0 is None:
        v0 = - δ * η * (y + y**2)

    d_Λ = γ_1 + γ_2 * y + γ_2p * (y > y_bar) * (y - y_bar)
    dd_Λ = γ_2 + γ_2p * (y > y_bar)

    πc_o = np.array([np.ones_like(y) * temp for temp in πc_o])
    πc = πc_o.copy()
    θ = np.array([np.ones_like(y) * temp for temp in θ])

    count = 0
    lhs_error = 1.

    while lhs_error > tol and count < max_iter:
        dvdy = compute_derivatives(v0, 1, dy)
        dvddy = compute_derivatives(v0, 2, dy)

        G = dvdy + (η - 1) * d_Λ

        if σ_y == 0:
            e_tilde = - δ * η / (G * np.sum(πc * θ, axis=0))
        else:
            temp = σ_y**2 * (dvddy + (η - 1.) * dd_Λ - G**2 / ξ_w)
            square = np.sum(πc*θ, axis=0)**2 * G**2 - 4 * δ * η * temp
            square[square < 0] = 0.
            e_tilde = (- G * np.sum(πc * θ, axis=0) - np.sqrt(square)) / (2 * temp)

        e_tilde[e_tilde <= 0] = 1e-16

        log_πc_ratio = - G * e_tilde * θ / ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc / np.sum(πc, axis=0)
        c_entropy = np.sum(πc * (np.log(πc) - np.log(πc_o)), axis=0)

        A = np.ones_like(y) * (- δ)
        B = e_tilde * np.sum(πc * θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ * η * np.log(e_tilde) -  C * G**2 / ξ_w\
            + (η-1) * d_Λ * e_tilde * np.sum(πc * θ, axis=0)\
            + .5 * (η - 1) * dd_Λ * σ_y**2 * e_tilde**2 + ξ_a * c_entropy

        v = false_transient(A, B, C, D, v0, ε, dy, (0, 0), (False, False))

        rhs_error = A * v + B * dvdy + C * dvddy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    h = - (dvdy + (η - 1) * d_Λ) * e_tilde * σ_y / ξ_w

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v': v,
           'dvdy': dvdy,
           'dvddy': dvddy,
           'e_tilde': e_tilde,
           'πc': πc,
           'h': h,
           'y': y,
           'model_args': model_args}
    return res


def solve_hjb_z(z, model_args=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_iteration=True):
    """
    Solve the HJB that is only related to z.

    Parameters
    ----------
    z : (N,) ndarray
        An evenly spaced grid of z.
    model_args : tuple of model inputs
        η, δ, ρ, μ_2, σ_2, ξ_1m : floats
    v0 : (N,) ndarray
        Initial guess of the value function
    ϵ : float
        Step size of the false transient method.
    tol : float
        Tolerance level of the hjb solver.
    max_iter : int
        Maximum iterations of the false transient method.
    print_iteration : bool
        Print the information of each iteration if True.

    Returns
    -------
    res : dict of ndarrays
        v : (N,) ndarray
            Value function
        dvdz : (N,) ndarray
            First order derivative of the value function.
        dvddz : (N,) ndarray
            Second order derivative of the value function.
        h : (N,) ndarray
            Implied drift distortion.              
        z : (N,) ndarray
            Grid of z.
        model_args : tuple
            Model parameters.            
    """    
    
    η, δ, ρ, μ_2, σ_2, ξ_w = model_args
    dz = z[1] - z[0]

    if v0 is None:
        v0 = - δ * η * (z + z**2)

    h0 = np.zeros_like(z)

    count = 0
    lhs_error = 1.

    while lhs_error > tol and count < max_iter:
        dvdz = compute_derivatives(v0, 1, dz)
        dvddz = compute_derivatives(v0, 2, dz)

        h = - (dvdz * np.sqrt(z) * σ_2) / ξ_w
        h = h * .5 + h0 * .5

        A = np.zeros_like(z)
        B = - ρ * (z - μ_2) + np.sqrt(z) * σ_2 * h
        C = .5 * σ_2**2 * z
        D = - δ * η * np.log(z) + .5 * ξ_w * h**2
        v = false_transient(A, B, C, D, v0, ε, dz, (0, 0), (False, False))

        rhs_error = A * v + B * dvdz + C * dvddz + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0) / ϵ))

        v0 = v
        h0 = h
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v': v,
           'dvdz': dvdz,
           'dvddz': dvddz,
           'h': h,
           'z': z,
           'model_args': model_args}
    return res


def solve_hjb_y_jump(y, model_args=(), v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_iteration=True):
    r"""
    Solve the HJB with y prior to jump. We impose certainty equivalent as a boundary condition
    to the equation instead of adding the jump intensity term.

    Parameters
    ----------
    y : (N,) ndarray
        An evenly spaced grid of y.
    model_args : tuple of model inputs
        η, δ, σ_y, y_bar, γ_1, γ_2, γ_2p, ξ_w, ξ_p, ξ_a : floats
        θ, πc_o : (M,) ndarrays
        ϕ_i, πd_o : (K,) ndarrays
    v0 : (N,) ndarray
        Initial guess of the value function
    ϵ : float
        Step size of the false transient method.
    tol : float
        Tolerance level of the hjb solver.
    max_iter : int
        Maximum iterations of the false transient method.
    print_iteration : bool
        Print the information of each iteration if True.

    Returns
    -------
    res : dict of ndarrays
        v : (N,) ndarray
            Value function
        dvdy : (N,) ndarray
            First order derivative of the value function.
        dvddy : (N,) ndarray
            Second order derivative of the value function.
        e_tilde : (N,) ndarray
            :math:`\tilde{e}` on the grid of y.
        h : (N,) ndarray
            Implied drift distortion.
        πc : (M, N) ndarray
            Distorted probabilities of θ.
        g : (K, N) ndarray
            Change in damage probability and intensity.
        bc : float
            The boundary condition that we impose on the HJB.
        y : (N,) ndarray
            Grid of y.            
        model_args : tuple
            Model parameters.
    """    
    
    η, δ, σ_y, y_bar, γ_1, γ_2, γ_2p, θ, πc_o, ϕ_i, πd_o, ξ_w, ξ_p, ξ_a = model_args
    dy = y[1] - y[0]

    if v0 is None:
        v0 = - δ * η * (y + y**2)

    d_Λ = γ_1 + γ_2 * y
    dd_Λ = γ_2

    πd_o = np.array([np.ones_like(y) * temp for temp in πd_o])
    πd = πd_o.copy()
    πc_o = np.array([np.ones_like(y) * temp for temp in πc_o])
    πc = πc_o.copy()
    θ = np.array([np.ones_like(y) * temp for temp in θ])
    
    e0_tilde = 0.

    count = 0
    lhs_error = 1.

    while lhs_error > tol and count < max_iter:
        dvdy = compute_derivatives(v0, 1, dy)
        dvddy = compute_derivatives(v0, 2, dy)

        G = dvdy + (η - 1) * d_Λ

        if σ_y == 0:
            e_tilde = - δ * η / (G * np.sum(πc * θ, axis=0))
        else:
            temp = σ_y**2*(dvddy + (η-1.) * dd_Λ - G**2 / ξ_w)
            root = np.sum(πc * θ, axis=0)**2 * G**2 - 4 * δ * η * temp
            root[root < 0] = 0.
            e_tilde = (- G * np.sum(πc * θ, axis=0) - np.sqrt(root)) / (2 * temp)

        e_tilde[e_tilde <= 0] = 1e-16
        e_tilde = e_tilde * .5 + e0_tilde * .5

        log_πc_ratio = - G * e_tilde * θ / ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc / np.sum(πc, axis=0)
        c_entropy = np.sum(πc * (np.log(πc) - np.log(πc_o)), axis=0)

        A = np.ones_like(y) * (- δ)
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = δ * η * np.log(e_tilde) - C * G**2 / ξ_w\
            + (η - 1) * d_Λ * e_tilde * np.sum(πc * θ, axis=0)\
            + .5 * (η-1) * dd_Λ * σ_y**2 * e_tilde**2 + ξ_a * c_entropy

        bc = - ξ_p * np.log(np.sum(πd_o[:, -1] * np.exp(- 1. / ξ_p * ϕ_i[:, -1])))

        v = false_transient(A, B, C, D, v0, ε, dy, (0, bc), (False, True))

        rhs_error = A * v + B * dvdy + C * dvddy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0) / ϵ))

        v0 = v
        e0_tilde = e_tilde
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    h = - (dvdy + (η - 1) * d_Λ) * e_tilde * σ_y / ξ_w
    g = np.exp(1. / ξ_p * (v - ϕ_i))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))     
    res = {'v': v,
           'dvdy': dvdy,
           'dvddy': dvddy,
           'e_tilde': e_tilde,
           'h': h,
           'πc': πc,
           'g': g,
           'bc': bc
           'y': y,
           'model_args': model_args}
    return res
