# -*- coding: utf-8 -*-
"""
Functions that solve the HJBs in the draft paper.

"""
import numpy as np
from .utilities import compute_derivatives, J
from .solver import false_transient

def solve_hjb_y(y, model_args=(), v0=None, ϵ=1., tol=1e-8, max_iter=10_000, print_iteration=True):
    r"""
    Solve the HJB that is only related to y.
    y is the accumulative change of temperature.

    Parameters
    ----------
    y : (N,) ndarray
        An evenly spaced grid of y.
    model_args : tuple of model inputs::
        The first 9 elements are floats for :math:`η, δ, σ_y, \bar{y}, γ_1, γ_2, γ_3, ξ_w, ξ_a`.

        The last two are (L,) ndarrays for :math:`\{θ_\ell\}_{\ell=1}^L, \{\pi^a_\ell\}_{\ell=1}^L`.
    v0 : (N,) ndarray, default=None
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
        ``dict``: {
            v : (N,) ndarray
                Value function, :math:`\phi(y)`.

            dvdy : (N,) ndarray
                First order derivative of the value function, :math:`\frac{d\phi(y)}{dy}`.
            dvddy : (N,) ndarray
                Second order derivative of the value function, :math:`\frac{d^2\phi(y)}{dy^2}`.
            e_tilde : (N,) ndarray
                :math:`\tilde{e}` on the grid of y.
            πc : (M, N) ndarray
                Distorted probabilities of :math:`\{θ_\ell\}_{\ell=1}^L`.
            h : (N,) ndarray
                Implied drift distortion.
            y : (N,) ndarray
                Grid of y.
            model_args : tuple
                Model parameters, see model_args above in `Parameters`.
        }
    """
    η, δ, σ_y, y_bar, γ_1, γ_2, γ_2p, θ, πc_o, ξ_w, ξ_a = model_args
    dy = y[1] - y[0]

    if v0 is None:
        v0 = - η * (y + y**2)

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

        G = dvdy + (η - 1) / δ * d_Λ

        if σ_y == 0:
            e_tilde = - η / (G * np.sum(πc * θ, axis=0))
        else:
            temp = σ_y**2 * (dvddy + (η - 1.) / δ * dd_Λ - G**2 / ξ_w)
            square = np.sum(πc*θ, axis=0)**2 * G**2 - 4 * η * temp
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
        D = η * np.log(e_tilde) -  C * G**2 / ξ_w\
            + (η-1) / δ * d_Λ * e_tilde * np.sum(πc * θ, axis=0)\
            + .5 * (η - 1) / δ * dd_Λ * σ_y**2 * e_tilde**2 + ξ_a * c_entropy

        v = false_transient(A, B, C, D, v0, ε, dy, (0, 0), (False, False))

        rhs_error = A * v + B * dvdy + C * dvddy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    h = - (dvdy + (η - 1) / δ * d_Λ) * e_tilde * σ_y / ξ_w

    # print("Converged. Total iteration: {:d};\t LHS Error: {:.10f};\t RHS Error: {:.10f} ".format(count, lhs_error, rhs_error))

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
    model_args : tuple of model inputs ::
        Elements are floats for :math:`η, δ, ρ, μ_2, σ_2, ξ_{1m}` in the HJB.
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
        ``dict``: {
            v : (N,) ndarray
                Value function, :math:`\zeta(z)`.
            dvdz : (N,) ndarray
                First order derivative of the value function, :math:`\\frac{\partial\zeta(z)}{\partial z}`.
            dvddz : (N,) ndarray
                Second order derivative of the value function, :math:`\\frac{\partial^2\zeta(z)}{\partial z\partial z'}`.
            h : (N,) ndarray
                Implied drift distortion.
            z : (N,) ndarray
                Grid of z.
            model_args : tuple
                Model parameters.
        }
    """

    η, δ, ρ, μ_2, σ_2, ξ_w = model_args
    dz = z[1] - z[0]

    if v0 is None:
        v0 = - η * (z + z**2)

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
        D = - η * np.log(z) + .5 * ξ_w * h**2
        v = false_transient(A, B, C, D, v0, ε, dz, (0, 0), (False, False))

        rhs_error = A * v + B * dvdz + C * dvddz + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0) / ϵ))

        v0 = v
        h0 = h
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    # print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

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
    model_args : tuple of model inputs::
        The ten float elements are for :math:`η, δ, σ_y, \bar{y}, γ_1, γ_2, γ_3, ξ_w, ξ_p, ξ_a`;

        Two `(L,) ndarrays` are for :math:`\{θ_\ell\}_{\ell=1}^L`, :math:`\{\pi^a_\ell\}_{\ell=1}^L`;

        Two `(M,N) ndarrays` are  for :math:`\{\phi_m(y)\}_{m=1}^M`, :math:`\{\pi_p^m\}_{m=1}^M`.
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
    res : dict of ndarrays ::

        dict: {
            v: (N,) ndarray
                Value function :math:`\phi(y)`
            dvdy: (N,) ndarray
                First order derivative of the value function, :math:`\frac{d\phi(y)}{dy}`
            dvddy: (N,) ndarray
                Second order derivative of the value function.
            e_tilde : (N,) ndarray
                :math:`\tilde{e}` on the grid of y.
            h : (N,) ndarray
                Implied drift distortion.
            πc : (M, N) ndarray
                Distorted probabilities of θ.
            g : (K, N) ndarray
                Change in damage probability and intensity.
            πd : (K, N) ndarray
                Distorted probabilities of damage functions.
            bc : float
                The boundary condition that we impose on the HJB.
            y : (N,) ndarray
                Grid of y.
            model_args : tuple
                Model parameters.
        }
    """

    η, δ, σ_y, y_underline, y_bar, γ_1, γ_2, γ_2p, θ, πc_o, ϕ_i, πd_o, ξ_w, ξ_p, ξ_a = model_args
    dy = y[1] - y[0]

    if v0 is None:
        v0 = - η * (y + y**2)

    d_Λ = γ_1 + γ_2 * y
    dd_Λ = γ_2

    πd_o = np.array([np.ones_like(y) * temp for temp in πd_o])
    πd = πd_o.copy()
    πc_o = np.array([np.ones_like(y) * temp for temp in πc_o])
    πc = πc_o.copy()
    θ = np.array([np.ones_like(y) * temp for temp in θ])

    J_y = J(y, y_underline)

    e0_tilde = 0.

    count = 0
    lhs_error = 1.

    while lhs_error > tol and count < max_iter:
        dvdy = compute_derivatives(v0, 1, dy)
        dvddy = compute_derivatives(v0, 2, dy)

        G = dvdy + (η - 1) / δ * d_Λ

        if σ_y == 0:
            e_tilde = - η / (G * np.sum(πc * θ, axis=0))
        else:
            temp = σ_y**2*(dvddy + (η - 1.) / δ * dd_Λ - G**2 / ξ_w)
            root = np.sum(πc * θ, axis=0)**2 * G**2 - 4 * η * temp
            root[root < 0] = 0.
            e_tilde = (- G * np.sum(πc * θ, axis=0) - np.sqrt(root)) / (2 * temp)

        e_tilde[e_tilde <= 0] = 1e-16
        e_tilde = e_tilde * .5 + e0_tilde * .5

        log_πc_ratio = - G * e_tilde * θ / ξ_a
        πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
        πc = np.exp(πc_ratio) * πc_o
        πc = πc / np.sum(πc, axis=0)
        c_entropy = np.sum(πc * (np.log(πc) - np.log(πc_o)), axis=0)


        g = np.exp(1. / ξ_p * (v0 - ϕ_i))

        A = np.ones_like(y) * (- δ) - J_y * np.sum(g * πd_o, axis=0)
        B = e_tilde * np.sum(πc*θ, axis=0)
        C = .5 * σ_y**2 * e_tilde**2
        D = η * np.log(e_tilde) - C * G**2 / ξ_w\
            + (η - 1.) / δ * d_Λ * e_tilde * np.sum(πc * θ, axis=0)\
            + .5 * (η - 1.) / δ * dd_Λ * σ_y**2 * e_tilde**2 + ξ_a * c_entropy\
            + J_y * ξ_p * np.sum(πd_o * (1 - g + g * np.log(g)), axis=0)\
            + J_y * np.sum( g * πd_o * ϕ_i, axis=0)


        bc = - ξ_p * np.log(np.sum(πd_o[:, -1] * np.exp(- 1. / ξ_p * ϕ_i[:, -1])))

        v = false_transient(A, B, C, D, v0, ε, dy, (0, bc), (False, False))

        rhs_error = A * v + B * dvdy + C * dvddy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0) / ϵ))

        v0 = v
        e0_tilde = e_tilde
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    h = - (dvdy + (η - 1.) / δ * d_Λ) * e_tilde * σ_y / ξ_w
    g = np.exp(1. / ξ_p * (v - ϕ_i))
    ι = np.sum(πd_o * g, axis=0)
    πd  = πd_o * g / ι
    print("Converged. Total iteration: %s;\t LHS Error: %s;\t RHS Error %s;\t" % (count, lhs_error, rhs_error))
    res = {'v': v,
           'dvdy': dvdy,
           'dvddy': dvddy,
           'e_tilde': e_tilde,
           'h': h,
           'πc': πc,
           'g': g,
           'πd': πd,
           'bc': bc,
           'y': y,
           'model_args': model_args}
    return res


def uncertainty_decomposition(y, model_args=(), e_tilde=None, h=None, πc=None, bc=None,
                              v0=None, ϵ=.5, tol=1e-8, max_iter=10_000, print_iteration=True):
    """
    Solve the PDE with a given :math:`\\tilde{e}`. If h is not given, minimize over h; if πc is not given,
    minimize over πc; if ``bc`` is not given, use certainty equivalent as ``bc``.

    Parameters
    ----------
    y : (N, ) array
        Grid of y, :math:`y \\in [0, \\bar{y}]`.
    model_args: tuple of model inputs
        same as ``model_args`` in  :func:`~source.model.solve_hjb_y_jump`.
    e_tilde : (N, ) array, default=None
        :math:`\\tilde{e}` solution from the HJB with full uncertainty configuration.
    h : (N, ) array, default=None::
        Drift distortion. To consider baseline brownian misspecification, assign `np.zeros_like(y)`.
    πc : (L, N) array, default=None::
        Distorted probabilities of :math:`\{\omega_\ell\}_{\ell=1}^L`. To consider baseline smooth ambiguity,
        assign equal weight to coefficients for evey point of y.
    bc : tuple, default=None
        Boundary condition.
    v0 : (N, ) array, default=None
        Initial guess.
    ϵ : float, default=0.5
        Step size.
    tol : float, default=1e-8
        Tolerance level.
    max_iter : int, default=10,000
        Maximum iterations of false transient method.
    print_iteration : bool, default=True
        Print the information of each iteration if `True`.

    Returns
    -------
    res : dict of ndarrays
        ``dict``: {
            v : (N, ) ndarrays
                Solution of value function for uncertainty decomposition.
            dvdy : (N, ) ndarrays
                First order derivatives
            dvddy : (N, ) ndarrays
                Second order derivatives
            y : (N, ) ndarray
                Grid of y
            ME : (N, ) ndarray
                Marginal value of emission.Computed according to () in
        }
    """
    if e_tilde is None:
        print("e_tilde is needed.")
    minimize_h = True if h is None else False
    minimize_πc = True if πc is None else False
    minimize_bc = True if bc is None else False

    η, δ, σ_y, γ_1, γ_2, θ, πc_o, ϕ_i, πd_o, ξ_w, ξ_p, ξ_a = model_args
    dy = y[1] - y[0]

    if v0 is None:
        v0 = -η * ( y + y**2)

    d_Λ = γ_1 + γ_2 * y
    dd_Λ = γ_2

    πc_o = np.array([np.ones_like(y) * temp for temp in πc_o])
    θ = np.array([np.ones_like(y) * temp for temp in θ])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        v_old = v0.copy()
        dvdy = compute_derivatives(v0, 1, dy)
        dvddy = compute_derivatives(v0, 2, dy)

        G = dvdy + (η - 1.) / δ * d_Λ
        F = dvddy + (η - 1.) / δ * dd_Λ

        # Minimize over πc if πc is not specified
        if minimize_πc:
            log_πc_ratio = - G * e_tilde * θ / ξ_a
            πc_ratio = log_πc_ratio - np.max(log_πc_ratio, axis=0)
            πc = np.exp(πc_ratio) * πc_o
            πc = πc / np.sum(πc, axis=0)
            πc[πc <= 0] = 1e-16
        c_entropy = np.sum(πc * (np.log(πc) - np.log(πc_o)), axis=0)

        # Minimize over h if h is not specified
        if minimize_h:
            h = -(dvdy + (η - 1.) / δ * d_Λ) * e_tilde * σ_y / ξ_w

        A = np.ones_like(y) * (-δ)
        B = e_tilde * (np.sum(πc * θ, axis=0) + σ_y * h)
        C = .5 * σ_y**2 * e_tilde**2
        D = η * np.log(e_tilde) + (η - 1.) / δ * d_Λ * e_tilde * (np.sum(πc * θ, axis=0) + σ_y * h)\
            + .5 * (η - 1.) / δ * dd_Λ * σ_y**2 * e_tilde**2 + ξ_w /2. * h**2\
            + ξ_a * c_entropy

        # Use certainty equivalent if bc is not specified
        if minimize_bc:
            bc = - ξ_p * np.log(np.sum(πd_o * np.exp(-1. / ξ_p * ϕ_i[:, -1])))

        v = false_transient(A, B, C, D, v0, ε, dy, (0, bc), (False, True))

        rhs_error = A * v0 + B * dvdy + C * dvddy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v_old)/ϵ))
        error = lhs_error

        v0 = v
        count += 1
        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    ME = - (dvdy + (η - 1.) / δ * d_Λ) * (np.sum(πc * θ, axis=0) + σ_y * h)\
         - (dvddy + (η - 1.) / δ * dd_Λ) * σ_y**2 * e_tilde

    print("Converged. Total iteration: %s;\t LHS Error: %s;\t RHS Error %s" % (count, lhs_error, rhs_error))

    res = {'v': v,
           'dvdy': dvdy,
           'dvddy': dvddy,
           'y': y,
           'ME': ME}
    return res

def minimize_π(y_grid, numy_bar, ems_star,  ϕ_list, args, with_damage=False, ϵ=2, tol=1e-7, max_iter=3_000):
    """
    compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ξp, ξa, ξw, σy, y_lower = args
#     ems_star = solu['ems']
    # solve for HJB with jump function
    y_grid_cap = y_grid[:numy_bar+1]
    dy = y_grid_cap[1] - y_grid_cap[0]
    dΛ = γ1 + γ2*y_grid_cap
    ddΛ = γ2
    r1 = 1.5
    r2 = 2.5
    intensity = r1*(np.exp(r2/2*(y_grid_cap- y_lower)**2)-1) *(y_grid_cap >= y_lower)
    
    loc_2 = np.abs(y_grid_cap - 2).argmin()
    ϕ_ref = np.zeros((len(γ3_list), numy_bar + 1))
    for i in range(len(γ3_list)):
        ϕ_ref[i, :] = ϕ_list[i, loc_2]
    
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)
    if with_damage == False:
        ϕ_bound = np.average(ϕ_list, axis=0, weights=dmg_weight)[:numy_bar+1]
    if with_damage == True:
        ϕ_bound = np.average(np.exp(-1/ξp*ϕ_ref), axis=0, weights=dmg_weight)[:numy_bar+1]
        ϕ_bound = -ξp*np.log(ϕ_bound)
    ϕ = ϕ_bound
    episode = 0
    lhs_error = 1
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = compute_derivatives(ϕ, 1, dy)
        dϕdyy = compute_derivatives(ϕ, 2, dy)
        # solver
        temp = dϕdy + (η-1)/δ*dΛ
        # minimize over π
        weight = np.array([ - 1/ξa*temp*ems_star*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πᶜ = πᶜo*np.exp(weight)
        πᶜ[πᶜ <= 1e-15] = 1e-15
        πᶜ = πᶜ/np.sum(πᶜ, axis=0)
        
        g_list = np.exp(1 / ξp * (ϕ - ϕ_ref))
        A = -δ*np.ones(y_grid_cap.shape) - intensity*(dmg_weight@g_list)
        B = (θ_list@πᶜ)*ems_star
        C = σy**2*ems_star**2/2
        D = η*np.log(ems_star) + (θ_list@πᶜ)*(η-1)/δ*dΛ*ems_star \
        + ξa*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0)\
        + 1/2*(η-1)/δ*ddΛ*ems_star**2*σy**2\
        + ξp * intensity * (dmg_weight@(1 - g_list + g_list * np.log(g_list)))\
        + intensity*(dmg_weight@(g_list*ϕ_ref))
        ϕ_new =  false_transient(A, B, C, D, ϕ, ϵ, dy, (0, ϕ_bound[numy_bar]), (False, True))
        rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))

#     dϕdy = derivative_1d(ϕ, 1, dy, "center")
#     dϕdyy = derivative_1d(ϕ, 2, dy, "center")
    ME = -(dϕdy+(η-1)/δ*dΛ)*(θ_list@πᶜ) - (dϕdyy+(η-1)/δ*ddΛ)*σy**2*ems_star
    ratio = ME/(η/ems_star)
    return ME, ratio


# solve for decompose
def minimize_g(y_grid, numy_bar, ems_star, ϕ_list, args, ϵ=3, tol=1e-6, max_iter=3_000):
    """
    compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ξp, ξa, ξw, σy, y_lower = args
#     ems_star = solu['ems']
    # solve for HJB with jump function
    ϕ_bound = np.average(np.exp(-1/ξp*ϕ_list), axis=0, weights=dmg_weight)
    ϕ_bound = -ξp*np.log(ϕ_bound)
    y_grid_cap = y_grid[:numy_bar+1]
    dy = y_grid_cap[1] - y_grid_cap[0]
    dΛ = γ1 + γ2*y_grid_cap
    ddΛ = γ2
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)
    θ = θ_list@πᶜo 
    
    r1 = 1.5
    r2 = 2.5
    intensity = r1*(np.exp(r2/2*(y_grid_cap- y_lower)**2)-1) *(y_grid_cap >= y_lower)
    

    loc_2 = np.abs(y_grid_cap - 2).argmin()
    ϕ_ref = np.zeros((len(γ3_list), numy_bar + 1))
    for i in range(len(γ3_list)):
        ϕ_ref[i, :] = ϕ_list[i, loc_2]
    
    ϕ = np.average(ϕ_list, axis=0, weights=dmg_weight)[:numy_bar+1]
    episode = 0
    lhs_error = 1
    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = compute_derivatives(ϕ, 1, dy)
        dϕdyy = compute_derivatives(ϕ, 2, dy)
        # solver
        temp = dϕdy + (η-1)/δ*dΛ
        g_list = np.exp(1/ξp*(ϕ - ϕ_ref))
        A = -δ*np.ones(y_grid_cap.shape) - intensity*(dmg_weight@g_list)
        B = θ*ems_star
        C = σy**2*ems_star**2/2
        D = η*np.log(ems_star) + (η-1)/δ*dΛ*ems_star*θ \
        + (η-1)/δ*ddΛ*ems_star**2*σy**2/2\
        + ξp * intensity * (dmg_weight@(1 - g_list + g_list * np.log(g_list)))\
        + intensity*(dmg_weight@(g_list*ϕ_ref))
        ϕ_new =  false_transient(A, B, C, D, ϕ, ϵ, dy, (0, ϕ_bound[numy_bar]), (False, True))
        rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))

#     dϕdy = derivative_1d(ϕ, 1, dy, "up")
#     dϕdyy = derivative_1d(ϕ, 2, dy, "up")
#     temp = dϕdy + (η-1)*dΛ    
    ME = -temp*θ - ( dϕdyy+(η-1)/δ*ddΛ)*σy**2*ems_star
    ratio = ME/(η/ems_star)

    return ME, ratio

def solve_baseline(y_grid, num_stop, ems_star, ϕ_list, args, ϵ=2, tol=1e-8, max_iter=3_000):
    """
    compute jump model with ambiguity over climate models
    """
    δ, η, θ_list, γ1, γ2, γ3_list, ȳ, dmg_weight, ξp, ξa, ξw, σy, y_lower = args
    r1=1.5
    r2=2.5
    y_grid_cap = y_grid[:num_stop+1]
    intensity =  r1*(np.exp(r2/2*(y_grid_cap- y_lower)**2)-1) *(y_grid_cap >= y_lower)
    
    dΛ = γ1 + γ2*y_grid_cap
    ddΛ = γ2

    ϕ = np.average(ϕ_list, axis=0, weights=dmg_weight)[:num_stop+1]

    loc_2 = np.abs(y_grid - 2).argmin()
    ϕ_ref = np.zeros((len(γ3_list), num_stop + 1))
    for i in range(len(γ3_list)):
        ϕ_ref[i, :] = ϕ_list[i, loc_2]
    
    dy = y_grid_cap[1] - y_grid_cap[0]
    episode = 0
    lhs_error = 1
    πᵈo = dmg_weight
    πᶜo = np.ones((len(θ_list), len(y_grid_cap)))/len(θ_list)

    ϕ_average = np.average( np.exp(-1/ξp*ϕ_list), weights=dmg_weight, axis=0)
    ϕ_bound = -ξp*np.log(ϕ_average)

    while lhs_error > tol and episode < max_iter:
        ϕ_old = ϕ.copy()
        dϕdy = compute_derivatives(ϕ, 1, dy)
        dϕdyy = compute_derivatives(ϕ, 2, dy)
        # update control
        temp = dϕdy + (η-1)/δ*dΛ 
        weight = np.array([ - 1/ξa*temp*ems_star*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πᶜ = πᶜo*np.exp(weight)
        πᶜ[πᶜ <= 1e-15] = 1e-15
        πᶜ = πᶜ/np.sum(πᶜ, axis=0)
        # update control

        g_list = np.ones(ϕ_ref.shape)
        # coefficients
        A = -δ*np.ones(y_grid_cap.shape)
        By = (θ_list@πᶜ)*ems_star
        Cyy = ems_star**2*σy**2/2
        D = η*np.log(ems_star) + θ_list@πᶜ*(η-1)/δ*dΛ*ems_star\
        + ξa*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0) \
        + 1/2*(η-1)/δ*ddΛ*ems_star**2*σy**2\
        + ξp * intensity * (dmg_weight@(1 - g_list + g_list * np.log(g_list)))\
        + intensity*(dmg_weight@(g_list*(ϕ_ref - ϕ)))
        # solver
        ϕ_new =  false_transient(A, By, Cyy, D, ϕ, ϵ, dy, (0, ϕ_bound[loc_2]), (False, False))

        rhs = -δ*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new 
        episode += 1
        
    
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    dϕdy = compute_derivatives(ϕ, 1, dy)
    dϕdyy = compute_derivatives(ϕ, 2, dy)
    ent = ξa*np.sum(πᶜ*(np.log(πᶜ) - np.log(πᶜo)), axis=0)
    ME = -(dϕdy+(η-1)/δ*dΛ)*(θ_list@πᶜ) - (dϕdyy+(η-1)/δ*ddΛ)*σy**2*ems_star
    ratio = ME/(η/ems_star)

    return ME, ratio
