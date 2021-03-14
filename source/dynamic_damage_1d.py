# -*- coding: utf-8 -*-
# +
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
from utilities import dLambda
from solver_1d import false_transient_one_iteration_python
import global_parameters as gp
from supportfunctions import finiteDiff
import matplotlib.pyplot as plt
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from derivative import derivatives_2d

import SolveLinSys
import time
from supportfunctions import PDESolver_2d, finiteDiff
import global_parameters as gp
from utilities import dLambda, ddLambda, weightOfPi, relativeEntropy, weightPI, damageDrift, zDrift


# -

@njit
def derivative_1d(data, order, h_data, upwind=False):
    num, = data.shape
    ddata = np.zeros_like(data)
    if order == 1:
        for i in range(num):
            if i == 0:
                ddata[i] = (data[i+1]-data[i])/h_data
            elif i == num-1:
                ddata[i] = (data[i]-data[i-1])/h_data
            else: 
                if upwind == True:
                    ddata[i] = (data[i]-data[i-1])/h_data
                else:
                    ddata[i] = (data[i+1]-data[i-1])/(2*h_data)
    elif order == 2:
        for i in range(num):
            if i == 0:
                ddata[i] = (data[i+2]-2*data[i+1] + data[i])/(h_data**2)
            elif i == num -1:
                ddata[i] = (data[i]-2*data[i-1] + data[i-2])/(h_data**2)
            else:
                ddata[i] = (data[i+1]- 2*data[i] + data[i-1])/(h_data**2)
    
    return ddata


@njit
def get_coeff(A, Bx, Cxx, D, x_grid, ϕ_prev, ϵ, boundspec):
    dx = x_grid[1] - x_grid[0]
    numx = len(x_grid)
    LHS = np.zeros((numx, numx))
    RHS = -1/ϵ*ϕ_prev - D
    for i in range(numx):
        LHS[i,i] += - 1/ϵ + A[i]
        if i == 0:
            LHS[i,i] += - 1/dx*Bx[i]
            LHS[i,i+1] += 1/dx*Bx[i]
        elif i == numx-1:
            if boundspec[0] == True:
                LHS[i,i] = 1
                RHS[i] = boundspec[1]
            else:
                LHS[i,i] += 1/dx*Bx[i]
                LHS[i,i-1] += -1/dx*Bx[i]
        else:
            LHS[i,i+1] += Bx[i]*(1./dx)*(Bx[i]>0) + Cxx[i]/(dx**2)
            LHS[i,i] += Bx[i]*((-1/dx)*(Bx[i]>0) + (1/dx)*(Bx[i]<0)) - 2*Cxx[i]/(dx**2)
            LHS[i,i-1] += Bx[i]*(-1/dx)*(Bx[i]<0) + Cxx[i]/(dx**2)
    return LHS, RHS


def solve_ode( A, By, Cyy, D, y_grid, ϕ_prev, ϵ, boundspec):
    LHS, RHS = get_coeff( A, By, Cyy, D, y_grid, ϕ_prev, ϵ, boundspec)
    phi_grid, exit_code = bicg(csc_matrix(LHS), RHS)
#     phi_grid = np.linalg.solve(LHS, RHS)
    return phi_grid


def false_transient_1d(
    y_grid, z, 
    dmg_params, model_params, bounded=False, bound_var=0, 
    epsilon=0.5, tol = 1e-8, max_iter=10_000,
):
    gamma_1, gamma_2, gamma2pList, gamma_bar, dmg_weights = dmg_params
    delta, eta, mu2, sigma2, rho, v_n = model_params
    numy, = y_grid.shape
    hy = y_grid[1] - y_grid[0]
    dlambda = gamma_1 + gamma_2*y_grid\
    + np.sum(gamma2pList*dmg_weights,axis=0)*(y_grid-gamma_bar)*(y_grid>gamma_bar)
    # initiate v and control
    ems = -delta*eta/((eta-1)*dlambda*z)
    error = 1
    episode = 0
    v0 = - delta*eta*y_grid
    
    while error > tol and episode < max_iter:
        v0_old = v0.copy()
        v0_dy = derivative_1d(v0,1,hy, upwind=True)
        # control
        ems_new = -delta*eta/(v0_dy*z + v_n*dlambda*z)
        ems_new[ems_new<=0] = 1e-15
        ems = ems_new*.5 + ems*.5
        A = -delta*np.ones(y_grid.shape)
        By = z*ems
        Cyy = np.zeros(y_grid.shape)
        D = delta*eta*np.log(ems) + v_n*dlambda*z*ems
        # solve for ODE
        phi_grid = solve_ode(y_grid, A, By, Cyy, D, v0, bound_var, bounded, epsilon)
        phi_grid2 = false_transient_one_iteration_python(
            A, By, Cyy, D, v0, epsilon, hy, (0,bound_var), (False, bounded))
        diff = np.max(abs(phi_grid - phi_grid2))
        rhs = A*phi_grid + By*v0_dy + D
        rhs_error = np.max(abs(rhs))
        error = np.max(abs((phi_grid-v0)/epsilon))
        v0 = phi_grid
        episode += 1
        print('Episode: {:d}\t lhs error: {:.12f}\t rhs error: {:.12f}\t diff: {:.12f}'.format(episode,error,rhs_error,diff))
    return v0, ems


# +
δ = 0.01
η = 0.032
μ = 1.86/1000
ȳ = 2

numy_bar = 100
y_min = 0
y_max = 4
hy = (ȳ - y_min)/numy_bar
y_grid = np.arange(y_min, y_max+hy, hy)

γ1 = 0.00017675
γ2 = 2*0.0022
γ2p = np.array([0, 2*0.0197])
γbar = 2
dmg_weight = np.array([1, 0])
dΛ = γ1 + γ2*y_grid + np.average(γ2p, weights=dmg_weight)*(y_grid - γbar)*(y_grid>γbar)

tol = 1e-8
ϵ = .3
lhs_error = 1
# -

ϕ = - δ*η*y_grid
dy = y_grid[1] - y_grid[0]
ems = -δ*η
episode = 0
while lhs_error > tol:
    ϕ_old = ϕ.copy()
    dϕdy = derivative_1d(ϕ, 1, dy, True)
    dϕdyy = derivative_1d(ϕ, 2, dy, True)
    ems = -δ*η/(dϕdy*μ + (η-1)*dΛ*μ)
    A = -δ*np.ones(y_grid.shape)
    By = μ*ems
    Cyy = np.zeros(y_grid.shape)
    D = δ*η*np.log(ems) + (η-1)*dΛ*μ*ems
    ϕ_new = solve_ode(A, By, Cyy, D, y_grid, ϕ,  ϵ, (False,0))
    rhs = A*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
    rhs_error = np.max(abs(rhs))
    lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
    ϕ = ϕ_new
    episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))

ϕ_low = ϕ
plt.plot(ϕ_low)

plt.plot(ems)

# +
dmg_weight = np.array([0, 1])
dΛ = γ1 + γ2*y_grid + np.average(γ2p, weights=dmg_weight)*(y_grid - ȳ)*(y_grid > ȳ)

tol = 1e-8
ϵ = .3
lhs_error = 1

ϕ = - δ*η*y_grid
dy = y_grid[1] - y_grid[0]
ems = -δ*η
episode = 0
while lhs_error > tol:
    ϕ_old = ϕ.copy()
    dϕdy = derivative_1d(ϕ, 1, dy, True)
    dϕdyy = derivative_1d(ϕ, 2, dy, True)
    ems = -δ*η/(dϕdy*μ + (η-1)*dΛ*μ)
    A = -δ*np.ones(y_grid.shape)
    By = μ*ems
    Cyy = np.zeros(y_grid.shape)
    D = δ*η*np.log(ems) + (η-1)*dΛ*μ*ems
    ϕ_new = solve_ode(A, By, Cyy, D, y_grid, ϕ,  ϵ, (False,0))
    rhs = A*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
    rhs_error = np.max(abs(rhs))
    lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
    ϕ = ϕ_new
    episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
# -

ϕ_high = ϕ

plt.plot(y_grid, ϕ_low)
plt.plot(y_grid, ϕ_high)
# plt.plot(y_grid, v_dict[2])

bd = (ϕ_low[numy_bar]+ϕ_high[numy_bar])/2
bd

# +
dmg_weight = np.array([0.5, 0.5])
y_grid_cap = np.linspace(0,2,100)
dΛ = γ1 + γ2*y_grid_cap + np.average(γ2p, weights=dmg_weight)*(y_grid_cap - γbar)*(y_grid_cap > ȳ)

tol = 1e-8
ϵ = .3
lhs_error = 1

ϕ = - δ*η*y_grid_cap
dy = y_grid_cap[1] - y_grid_cap[0]
ems = δ*η
ems_old = ems
episode = 0
while lhs_error > tol:
    ϕ_old = ϕ.copy()
    dϕdy = derivative_1d(ϕ, 1, dy, True)
    dϕdyy = derivative_1d(ϕ, 2, dy, True)
    ems = -δ*η/(dϕdy*μ + (η-1)*dΛ*μ)
    ems[ems<=0] = 1e-15
    ems = ems*0.5 + ems_old*0.5
    A = -δ*np.ones(y_grid_cap.shape)
    By = μ*ems
    Cyy = np.zeros(y_grid_cap.shape)
    D = δ*η*np.log(ems) + (η-1)*dΛ*μ*ems
    ϕ_new = solve_ode(A, By, Cyy, D, y_grid_cap, ϕ,  ϵ, (True,bd))
    rhs = A*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
    rhs_error = np.max(abs(rhs))
    lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
    ϕ = ϕ_new
    ems_old = ems
    episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
# -

ϕ_pre = ϕ
plt.plot(y_grid_cap, ϕ_pre)

ϕ_pre[-1]

plt.plot(y_grid_cap,ϕ_pre)
plt.plot(y_grid[numy_bar:], (ϕ_low[numy_bar:] + ϕ_high[numy_bar:])/2)
plt.vlines(2, ymin=-.04, ymax=.1, color="black", linestyle="dashed")

# # Jump of damage
# $$
# 0 = -\delta \phi + \frac{\partial \phi }{\partial y} \mu_2 e + \delta\eta\log (e) + (\eta-1)(\tau_1 + \tau_2 y) \mu_2 e +\frac{\exp\{\rho( y - \bar y)\}}{1 - \exp\{\rho( y -\bar y)\}} \cdot \left(\sum_{j=2}^{n}\pi_j\tilde \phi(y) - \phi(y)\right) \quad y \in [0, \bar y)
# $$
# where
# $$
#     e = -\frac{ \delta\eta}{\frac{\partial \phi }{\partial y} \mu_2 +  (\eta-1)(\tau_1 + \tau_2 y) \mu_2 }
# $$
#
# $$
# \phi(\bar y) = \sum_{j=2}^{n} \pi_j \phi_j(\bar y)
# $$
#
# $$
# \text{Jump intensity} = \frac{1}{\sqrt{2\pi}\sigma}\exp\{ - \frac{(y - \bar  y)^2}{2\sigma^2}\}
# $$
#
# Try
# $$
# \sigma = \bar y/10, \quad \bar y/50, \quad \bar y/100
# $$

y_dense = np.arange(0,2,1/20000)
def get_intensity(y_grid, σ, γbar=2):
    temp = 1/(np.sqrt(np.pi*2)*σ)*np.exp(-(y_grid - γbar)**2/(2*σ**2))
#     temp *= v_bar - v_new
    return temp


# +
dmg_weight = np.array([0.5, 0.5])
y_grid_cap = y_grid[:numy_bar+1]
dΛ = γ1 + γ2*y_grid_cap + np.average(γ2p, weights=dmg_weight)*(y_grid_cap - γbar)*(y_grid_cap>γbar)


ϕ_average = np.average([ϕ_low[:numy_bar+1], ϕ_high[:numy_bar+1]], axis=0, weights=dmg_weight)
tol = 1e-8


ϕ = - δ*η*y_grid_cap
dy = y_grid_cap[1] - y_grid_cap[0]
ems = δ*η
episode = 0

ϕ_dict = dict()
for σ in [γbar/10, γbar/50, γbar/100]:
    ϕ = - δ*η*y_grid_cap
    dy = y_grid_cap[1] - y_grid_cap[0]
    ems = δ*η
    ems_old = ems
    episode = 0
    ϵ = .3
    lhs_error = 1
    while lhs_error > tol:
        ϕ_old = ϕ.copy()
        dϕdy = derivative_1d(ϕ, 1, dy, True)
        dϕdyy = derivative_1d(ϕ, 2, dy, True)
        ems = -δ*η/(dϕdy*μ + (η-1)*dΛ*μ)
        ems[ems<=0] = 1e-15
        ems = 0.5*ems + 0.5*ems_old
        A = -δ*np.ones(y_grid_cap.shape) - get_intensity(y_grid_cap, σ)
        By = μ*ems
        Cyy = np.zeros(y_grid_cap.shape)
        D = δ*η*np.log(ems) + (η-1)*dΛ*μ*ems + get_intensity(y_grid_cap, σ)*ϕ_average
        ϕ_new = solve_ode(A, By, Cyy, D, y_grid_cap, ϕ, ϵ, (True, bd))
        rhs = -δ*ϕ_new + By*dϕdy + Cyy*dϕdyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
        ϕ = ϕ_new
        episode += 1
        ems_old = ems
        print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
    ϕ_dict[σ] = ϕ
# -

for σ in [γbar/10, γbar/50, γbar/100]:
    plt.plot(y_grid_cap,ϕ_dict[σ], label=σ)
plt.legend()
plt.plot(y_grid_cap, ϕ_average)
plt.plot(y_grid[numy_bar:], (ϕ_low[numy_bar:] + ϕ_high[numy_bar:])/2)
plt.vlines(2, ymin=-.04, ymax=.1, color="black", linestyle="dashed")

plt.plot(ϕ_average)

# # new setup
# $$
# \begin{aligned}
# 0 = \max_{\tilde e}\min_h & - \delta \phi(y) + \delta\eta \log\tilde e \\
#                            &  + \frac{\xi_m}{2} h'h + \frac{d\phi(y)}{dy} \tilde e (\theta + \sigma_y h) + \frac{1}{2}\frac{d^2\phi(y)}{dy^2}|\sigma_y|^2(\tilde e)^2\\
#                            & + (\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot (\theta + \sigma_y h)
# \end{aligned}
# $$
#
# $$
# \begin{aligned}
# 0 = \min_h & - \delta \psi(z_2) - \delta \eta \log(z_2) + \frac{\xi_m}{2} h'h\\
#  & + \frac{d\psi(z_2)}{dz_2}\left[ - \rho(z_2 - \mu_2) + \sqrt{z_2}\sigma_2 h \right] + \frac{z_2|\sigma_2|^2}{2}\frac{d^2\psi(z_2)}{dz_2^2}
# \end{aligned}
# $$
#
# $$
#     h^* = -\frac{\frac{d\psi(z_2)}{dz_2} \sqrt{z_2}\sigma_2}{\xi_m}
# $$

# z2 grid
ρ = 0.9
ξₘ = 1/10000
σz = 0.42/1000
σ2 = np.sqrt(2*ρ*σz**2/μ)
num_z = 100
z2_min = μ - 4*σz
z2_max = μ + 4*σz
z2_grid = np.linspace(z2_min, z2_max, num_z)
hz = z2_grid[1] - z2_grid[0]
# ODE for z_2
episode = 0
ϵ = .3
tol = 1e-8
lhs_error = 1
ψ = δ*z2_grid
h_star = -δ*np.sqrt(z2_grid)*σ2/ξₘ
while lhs_error > tol:
    ψ_old = ψ.copy()
    dψdz = derivative_1d(ψ, 1, hz)
    dψdzz = derivative_1d(ψ, 2, hz)
    h_star = - dψdz*np.sqrt(z2_grid)*σz/ξₘ*0.5 + h_star*0.5
    A = -δ*np.ones(z2_grid.shape)
    B = - ρ*(z2_grid - μ) + np.sqrt(z2_grid)*σ2*h_star
    C = z2_grid*σ2**2/2
    D = - δ*η*np.log(z2_grid) + 1/2*ξₘ*h_star**2
    ψ_new = solve_ode(A, B, C, D, z2_grid, ψ, ϵ, (False, 0))
    rhs = -δ*ψ_new + B*dψdz + C*dψdzz + D
    rhs_error = np.max(abs(rhs))
    lhs_error = np.max(abs((ψ_new - ψ_old)/ϵ))
    ψ = ψ_new
    episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))

plt.figure(figsize=(20,6))
plt.subplot(121)
plt.plot(z2_grid, ψ)
plt.title("$\\psi(z)$")
plt.subplot(122)
plt.plot(z2_grid, h_star)
plt.title("$h_z$")

# ## ODE for y
# $$
# \begin{aligned}
# 0 = \max_{\tilde e}\min_h & - \delta \phi(y) + \delta\eta \log\tilde e \\
# &  + \frac{\xi_m}{2} h'h + \frac{d\phi(y)}{dy} \tilde e (\theta + \sigma_y h) + \frac{1}{2}\frac{d^2\phi(y)}{dy^2}|\sigma_y|^2(\tilde e)^2\\
#                            & + (\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot (\theta + \sigma_y h)
# \end{aligned}
# $$
#
# $$
# h^* = - \frac{\frac{d\phi(y)}{dy}\tilde e \sigma_y + (\eta - 1)(\gamma_1 + \gamma_2 y)\tilde e \sigma_y}{\xi_m}
# $$
# Plug it  back to the HJB:
# $$
# \begin{aligned}
#  0 = \max_{\tilde e} \quad - & \delta\phi(y) + \delta\eta \log\tilde e \\
#  & + \frac{d\phi(y)}{dy}\tilde e \theta +  (\eta -1 )(\gamma_1 + \gamma_2 y)\tilde e \theta \\ 
#  & + \left[- \frac{1}{2\xi_m}\left( \frac{d\phi(y)}{dy}  + (\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{1}{2}\frac{d^2\phi(y)}{dy^2}\right] \cdot (\tilde e)^2 |\sigma_y|^2
# \end{aligned}
# $$
#
# First order condition for $\tilde e ^*$:
# $$
#   \left[- \frac{1}{\xi_m}\left( \frac{d\phi(y)}{dy}  + (\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2\phi(y)}{dy^2}\right] \cdot|\sigma_y|^2\cdot(\tilde e)^2 + \left[\frac{d\phi(y)}{dy} + (\eta - 1)(\gamma_1 + \gamma_2 y)\right]\theta\tilde e + \delta\eta = 0
# $$
#
# $$
# A = \left[- \frac{1}{\xi_m}\left( \frac{d\phi(y)}{dy}  + (\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2\phi(y)}{dy^2}\right] \cdot|\sigma_y|^2
# $$
#
# $$
# B = \left[\frac{d\phi(y)}{dy} + (\eta - 1)(\gamma_1 + \gamma_2 y)\right]\theta
# $$
#
# $$
# C = \delta\eta
# $$
#
# $$
# \tilde e^* = \cfrac{-B - \sqrt{B^2  - 4AC}}{2A}
# $$

# +
# y grid
ξₘ = 1/400
μ = 1.86/1000
θ = μ
σy = 1.2*θ
num_y = 50
y_min = 1e-10
y_max = 10
y_grid = np.linspace(y_min, y_max, num_y)
hy = y_grid[1] - y_grid[0]
ϵ = .3
tol = 1e-8
# ODE for z_2
ϕ_dict = dict()
ems_dict = dict()

episode = 0
lhs_error = 1
dΛ = γ1 + γ2*y_grid
ϕ =  - δ*η*y_grid**2
ems = - δ*η/((η-1)*dΛ*θ)
ems_old = ems
h_star = 0
while lhs_error > tol:
    ϕ_old = ϕ.copy()
    dϕdy = derivative_1d(ϕ, 1, hy)
    dϕdyy = derivative_1d(ϕ, 2, hy)

    a = dϕdyy*σy**2 - (dϕdy + (η-1)*dΛ)**2*σy**2/ξₘ 
    b = (dϕdy + (η-1)*dΛ)*θ
    c = δ*η
    Δ = b**2 - 4*a*c
    ems_new = (-b - np.sqrt(Δ))/(2*a)
    ems_new[ems_new <= 1e-15] = 1e-15    
    ems = ems_new*1 + ems_old*0
    # coefficients
    temp = dϕdy + (η-1)*dΛ
    A = -δ*np.ones(y_grid.shape)
    B = ems*θ
    C = σy**2*ems**2/2
    D = δ*η*np.log(ems) + (η-1)*dΛ*ems*θ - 1/(2*ξₘ)*temp**2*ems**2*σy**2

    ϕ_new = solve_ode(A, B, C, D, y_grid, ϕ, ϵ, (False, 0))
    rhs = -δ*ϕ_new + B*dϕdy + C*dϕdyy + D
    rhs_error = np.max(abs(rhs))
    lhs_error = np.max(abs((ϕ_new - ϕ_old)/ϵ))
    ϕ = ϕ_new
    ems_old = ems
    episode += 1
    print("episode: {},\t ode error: {},\t ft error: {}".format(episode, rhs_error, lhs_error))
# -

ems_1d = ems
plt.plot(y_grid, ems_1d)
plt.xlabel('y')
plt.title("$\\tilde e(y)$")

ϕ_1d = ϕ
plt.plot(y_grid, ϕ_1d)
plt.xlabel('y')
plt.title('ϕ(y)')


# compute hˢ
def compute_h_star(ϕ, ems, y_grid, args=(η, σy, γ1, γ2, ξₘ)):
    η, σy, γ1, γ2, ξₘ = args
    dy = y_grid[1] - y_grid[0]
    dϕdy = derivative_1d(ϕ, 1, dy)
    dΛ = γ1 + γ2*y_grid
    h_star = dϕdy + (η-1)*dΛ
    h_star *= ems*σy
    h_star *= - 1/ξₘ
    return h_star


args = (η, σy, γ1, γ2, ξₘ)
h_1d = compute_h_star(ϕ_1d, ems_1d, y_grid, args)

plt.plot(y_grid, h_1d)
plt.xlabel('y')
plt.title("h")

plt.plot( (dϕdy + (η-1)*(γ1 + γ2*y_grid))*ems_1d)

ems_1d

plt.plot(y_grid, dϕdyy)
plt.hlines(y=0, xmin=0, xmax=10)

# # With reserve constraint
# $$
# \begin{aligned}
# 0 = \max_{\tilde e}\min_h & \quad b \delta\eta \log\tilde e \\
# &  + b \frac{\xi_m}{2} h'h + \frac{dV}{dy} \tilde e (\theta + \sigma_y h) - \delta b \frac{dV}{db} + \frac{1}{2}\frac{d^2V}{dy^2}|\sigma_y|^2(\tilde e)^2\\
#                            & + b(\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot (\theta + \sigma_y h) - \ell \tilde e
# \end{aligned}
# $$
#
# $$
# h^* = - \frac{\frac{dV}{dy} + b(\eta-1)(\gamma_1 + \gamma_2 y)}{b \xi_m} \cdot \tilde e \sigma_y
# $$
# Plug back into the HJB:
# $$
# \begin{aligned}
# 0 = \max_{\tilde e } \quad & b \delta\eta \log \tilde e + \frac{dV}{dy} \tilde e \theta - \delta b \frac{dV}{db} + \frac{1}{2}\frac{d^2V}{dy^2}|\sigma_y|^2(\tilde e)^2 \\ 
# & +  b(\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot \theta - \ell \tilde e \\
# & -\frac{1}{2b\xi_m}\left(\frac{dV}{dy} + b(\eta-1)(\gamma_1 + \gamma_2 y)\right)^2 (\tilde e)^2 |\sigma_y|^2
# \end{aligned}
# $$
# First order condition for $\tilde e ^*$:
# $$
#   \left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b(\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2V}{dy^2}\right] \cdot|\sigma_y|^2\cdot(\tilde e)^2 + \left[\frac{dV}{dy}\theta + b(\eta - 1)(\gamma_1 + \gamma_2 y)\theta - \ell \right]\tilde e + b \delta\eta = 0
# $$
#
# $$
# A =  \left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b(\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2V}{dy^2}\right] \cdot|\sigma_y|^2
# $$
#
# $$
# B = \left[\frac{dV}{dy} + b(\eta - 1)(\gamma_1 + \gamma_2 y)\right]\theta - \ell
# $$ 
#
# $$
# C = b \delta\eta
# $$
# And
# $$
# \tilde e^* = \frac{-B -  \sqrt{B^2 - 4AC}}{2A}
# $$

b_grid = np.linspace(1e-10, 1, 100)
y_grid = np.linspace(1e-10, 10, 100)
# mesh grid and construct state space
(y_mat, b_mat) = np.meshgrid(y_grid, b_grid, indexing = 'ij')
stateSpace = np.hstack([y_mat.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])
hb = b_grid[1] - b_grid[0]
hy = y_grid[1] - y_grid[0]

# +
# 2 state HJB with constraints
θ = μ
δ = 0.01
η = 0.032

ϵ = 0.3
# ℓ = 1e-12
ξₘ = 1/400
σy = 1.2*θ

tol = 1e-8
dΛ = γ1 + γ2*y_mat


v_dict = dict()
ems_dict = dict()
ℓ_step = 1e-16

for ℓ in [1e-12, 1e-5]:
    episode = 0
    lhs_error = 1
    ems = - δ*η/(b_mat*(η-1)*dΛ*θ)
    ems_old = ems
    while lhs_error > tol:
        if episode ==0:
            v0 =  - δ*η*y_mat**2
        else:
            vold = v0.copy()
        v0_dy = derivatives_2d(v0,0,1,hy)
        v0_dyy = derivatives_2d(v0,0,2,hy)
        v0_db = derivatives_2d(v0,1,1,hb)
        # updating controls

        print(np.min(ems))
        temp = v0_dy + b_mat*(η-1)*dΛ
        a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2
        b = temp*θ  - ℓ
        c = δ*η*b_mat
        Δ = b**2 - 4*c*a
        Δ[Δ<0] = 0
        ems_new =  -b/(2*a) - np.sqrt(Δ)/(2*a)
        ems_new[ems_new <= 0] = 1e-15
        ems = ems_new
        # HJB coefficient
        A =  np.zeros(y_mat.shape)
        B_y =  ems*θ
        B_b = - δ*b_mat
        C_yy = ems**2*σy**2/2
        C_bb = np.zeros(y_mat.shape)
        D = b_mat*δ*η*np.log(ems) + b_mat*(η-1)*dΛ*ems*θ - ℓ*ems - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
        # PDE solver
        solve_start = time.time()
        out = PDESolver_2d(stateSpace, A, B_y, B_b, C_yy, C_bb, D, v0, ϵ, solverType = 'False Transient')
        out_comp = out[2].reshape(v0.shape,order = "F")
        rhs = A*v0 + B_y*v0_dy + B_b*v0_db + C_yy*v0_dyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((out_comp - v0)))
        #     if episode % 1 == 0:
        print("Episode {:d}: PDE Error: {:.12f}; False Transient Error: {:.12f}; Iterations: {:d}; CG Error: {:.12f}".format(episode,
              rhs_error, lhs_error, out[0], out[1]))
        episode += 1
        v0 = out_comp
        ems_old = ems
        print("End of PDE solver, takes time: {}".format(time.time() - solve_start))
    
    v_dict[ℓ] = v0
    ems_dict[ℓ] = ems
# -

# plt.plot(y_grid, ϕ_1d, label="no constraint")
plt.plot(y_mat[:,0], v_dict[1e-12][:,-1], label="$ℓ = 10^{-12}$")
# plt.plot(y_mat[:,0], v_dict[1e-5][:,-1], label="$ℓ = 10^{-5}$")
plt.legend()
plt.xlabel('y')
plt.title('ϕ(y)')
# plt.savefig("phi.png", dpi=148, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches="tight", pad_inches=0.1,)

# plt.plot(y_grid, ems_1d, label="no contrainst")
plt.plot(y_mat[:,0], ems_dict[1e-12][:,-1], label="$ℓ = 10^{-12}$")
# plt.plot(y_mat[:,0], ems_dict[1e-5][:,-1], label="$ℓ = 10^{-5}$")
plt.legend()
plt.title('$\\tilde e(y)$')
# plt.savefig("e_tilde.png", dpi=148, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches="tight", pad_inches=0.1,)

def compute_h_2d(ϕ, y_mat, b_mat, ems, args=(η, σy, γ1, γ2, ξₘ)):
    η, σy, γ1, γ2, ξₘ = args
    dΛ = γ1 + γ2*y_mat
    dy = y_mat[1,0] - y_mat[0,0]
    dϕdy = derivatives_2d(ϕ, 0, 1, dy)
    h_star = dϕdy + b_mat*(η-1)*dΛ
    h_star *= ems*σy
    h_star *= - 1/(b_mat*ξₘ)
    return h_star


args=(η, σy, γ1, γ2, ξₘ)
h_2d_dict = dict()
for ℓ in [1e-12, 1e-5]:
    h_2d_dict[ℓ] = compute_h_2d(v_dict[ℓ], y_mat, b_mat, ems_dict[ℓ], args)

# plt.plot(y_grid[:], h_1d[:], label="no constraint")
plt.plot(y_mat[:,0], h_2d_dict[1e-12][:,-1], label="$ℓ = 10^{-12}$")
# plt.plot(y_mat[:,0], h_2d_dict[1e-5][:,-1], label="$ℓ = 10^{-5}$")
plt.legend()
plt.xlabel('y')
plt.title('h')
# plt.savefig("h_star.png", dpi=148, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches="tight", pad_inches=0.1,)

# 2 state HJB with constraints
ℓ_step = 1e-20
dΛ = γ1 + γ2*y_mat
V_dict = dict()
E_dict = dict()
for ℓ in [1e-12, ℓ_step + 1e-12]:
    episode = 0
    lhs_error = 1
    ems = - δ*η/(b_mat*(η-1)*dΛ*θ)
    ems_old = ems
    while lhs_error > tol:
        if episode ==0:
            v0 =  - δ*η*y_mat**2
        else:
            vold = v0.copy()
        v0_dy = derivatives_2d(v0,0,1,hy)
        v0_dyy = derivatives_2d(v0,0,2,hy)
        v0_db = derivatives_2d(v0,1,1,hb)
        # updating controls

        print(np.min(ems))
        temp = v0_dy + b_mat*(η-1)*dΛ
        a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2
        b = temp*θ  - ℓ
        c = δ*η*b_mat
        Δ = b**2 - 4*c*a
        Δ[Δ<0] = 0
        ems_new =  -b/(2*a) - np.sqrt(Δ)/(2*a)
        ems_new[ems_new <= 0] = 1e-15
        ems = ems_new
        # HJB coefficient
        A =  np.zeros(y_mat.shape)
        B_y =  ems*θ
        B_b = - δ*b_mat
        C_yy = ems**2*σy**2/2
        C_bb = np.zeros(y_mat.shape)
        D = b_mat*δ*η*np.log(ems) + b_mat*(η-1)*dΛ*ems*θ - ℓ*ems - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
        # PDE solver
        solve_start = time.time()
        out = PDESolver_2d(stateSpace, A, B_y, B_b, C_yy, C_bb, D, v0, ϵ, solverType = 'False Transient')
        out_comp = out[2].reshape(v0.shape,order = "F")
        rhs = A*v0 + B_y*v0_dy + B_b*v0_db + C_yy*v0_dyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((out_comp - v0)))
        #     if episode % 1 == 0:
        print("Episode {:d}: PDE Error: {:.12f}; False Transient Error: {:.12f}; Iterations: {:d}; CG Error: {:.12f}".format(episode,
              rhs_error, lhs_error, out[0], out[1]))
        episode += 1
        v0 = out_comp
        ems_old = ems
        print("End of PDE solver, takes time: {}".format(time.time() - solve_start))
    
    V_dict[ℓ] = v0
    E_dict[ℓ] = ems

# $$
#  - \frac{\partial V}{\partial \ell} (y,1;\ell) = r
# $$

plt.plot(E_dict[1e-12][:,-1])

y_mat.shape

(V_dict[1e-12][:,-1] -  V_dict[1e-12+ℓ_step][:,-1])/ℓ_step

plt.plot(y_mat[:,0], - (V_dict[1e-12+ℓ_step][:,-1] - V_dict[1e-12][:,-1])/ℓ_step)
plt.title('r')
plt.xlabel('y')
plt.savefig("r.png", dpi=148, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches="tight", pad_inches=0.1,)
