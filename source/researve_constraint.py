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
def derivative_1d(data, order, h_data, upwind=True):
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

# # With reserve constraint
# $$
# \begin{aligned}
# 0 = \max_{\tilde e}\min_h & \quad b \delta\eta \log\tilde e \\
# &  + b \frac{\xi_m}{2} h'h + \frac{dV}{dy} \tilde e (\theta + \sigma_y h) - \delta b \frac{dV}{db} + \frac{1}{2}\frac{d^2V}{dy^2}|\sigma_y|^2(\tilde e)^2\\
# & + b(\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot (\theta + \sigma_y h) + b\frac{1}{2}(\eta -1)(\gamma_2)(\tilde e)^2|\sigma_y|^2 - \ell \tilde e
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
# & +  b(\eta -1 )\cdot(\gamma_1 + \gamma_2 y)\cdot\tilde e\cdot \theta  + b\frac{1}{2}(\eta -1)(\gamma_2)(\tilde e)^2|\sigma_y|^2 \color{red}{ - \ell \tilde e}\\
# & -\frac{1}{2b\xi_m}\left(\frac{dV}{dy} + b(\eta-1)(\gamma_1 + \gamma_2 y)\right)^2 (\tilde e)^2 |\sigma_y|^2
# \end{aligned}
# $$
#
# $$
# b \in (0,1]
# $$
#
# $$
# y: \text{celsius}\quad^oC
# $$
#
# $$
#     \theta: \quad \text{celsius per gigatonne of carbon}, \quad ^oC/GtC
# $$
#
# $$
#     \sigma_y = 1.2 \theta:  \quad \text{celsius per gigatonne of carbon}, \quad ^oC/GtC
# $$
#
# $$
# \tilde e: \quad \text{gigatonne of carbon}, \quad GtC
# $$
#
# $$
# \gamma_1 = 1.7675\times 10^{-4}
# $$
#
# $$
# \gamma_2 = 0.0044
# $$
#
# First order condition for $\tilde e ^*$:
# $$
#   \left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b(\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2V}{dy^2} + b(\eta -1) \gamma_2 \right] \cdot|\sigma_y|^2\cdot(\tilde e)^2 + \left[\frac{dV}{dy}\theta + b(\eta - 1)(\gamma_1 + \gamma_2 y)\theta - \ell \right]\tilde e + b \delta\eta = 0
# $$
#
# $$
# A =  \left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b(\eta -1)(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2V}{dy^2}+ b(\eta -1) \gamma_2 \right] \cdot|\sigma_y|^2
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
y_grid = np.linspace(1e-10, 4, 100)
# mesh grid and construct state space
(y_mat, b_mat) = np.meshgrid(y_grid, b_grid, indexing = 'ij')
stateSpace = np.hstack([y_mat.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])
hb = b_grid[1] - b_grid[0]
hy = y_grid[1] - y_grid[0]

# +
# 2 state HJB with constraints
θ = 1.86/1000
δ = 0.01
η = 0.032

ϵ = 1
# ℓ = 1e-12
ξₘ = 1000
σy = 1.2*θ

γ3_list = np.array([0, 2*0.0197, 2*0.3853])
π_p = np.array([1/3, 1/3, 1/3])
tol = 1e-8
dΛ = γ1 + γ2*y_mat + π_p@γ3_list*(y_mat - ȳ)*(y_mat >ȳ)
ddΛ = γ2 + π_p@γ3_list*(y_mat >ȳ)

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
        a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2 + b_mat*(η - 1)*ddΛ*σy**2
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
        D = b_mat*δ*η*np.log(ems) + b_mat*(η-1)*(dΛ*ems*θ + 1/2*ddΛ*ems**2*σy**2)\
        - ℓ*ems - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
        # PDE solver
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

plt.plot(ems_dict[1e-5][:,-1])
plt.plot(ems_dict[1e-12][:,-1])


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
plt.ylim(0,0.3)
# plt.savefig("h_star.png", dpi=148, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches="tight", pad_inches=0.1,)

# +
# 2 state HJB with constraints
θ = 1.86/1000
δ = 0.01
η = 0.032

ϵ = 1
# ℓ = 1e-12
ξₘ = 1000
σy = 1.2*θ

γ3_list = np.array([0, 2*0.0197, 2*0.3853])
π_p = np.array([1, 0, 0])
dΛ = γ1 + γ2*y_mat + π_p@γ3_list*(y_mat - ȳ)*(y_mat >ȳ)
ddΛ = γ2 + π_p@γ3_list*(y_mat >ȳ)
R_max = 9000

tol = 1e-7
ℓ = 1e-15
ℓ_step = 1e-15

ℓ_list = [ℓ, ℓ+ℓ_step]
# ℓ_list = np.linspace(1e-20,1e-10 , 10)
V_dict = dict()
E_dict = dict()

for ℓ_i in ℓ_list:
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
        a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2+ b_mat*(η - 1)*ddΛ*σy**2
        b = temp*θ  - ℓ_i*θ 
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
        D = b_mat*δ*η*np.log(ems) +  b_mat*(η-1)*(dΛ*ems*θ + 1/2*ddΛ*ems**2*σy**2)\
        - ℓ_i*ems*θ - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
        # PDE solver
        solve_start = time.time()
        out = PDESolver_2d(stateSpace, A, B_y, B_b, C_yy, C_bb, D, v0, ϵ, solverType = 'False Transient')
        out_comp = out[2].reshape(v0.shape,order = "F")
        rhs = A*v0 + B_y*v0_dy + B_b*v0_db + C_yy*v0_dyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((out_comp - v0)))
        #     if episode % 1 == 0:
        print("Episode {:d}: PDE Error: {:.12f}; False Transient Error: {:.12f}; Iterations: {:d}; CG Error: {:.12f}".format(episode, rhs_error, lhs_error, out[0], out[1]))
        episode += 1
        v0 = out_comp
        ems_old = ems
        print("End of PDE solver, takes time: {}".format(time.time() - solve_start))
    
    V_dict[ℓ_i] = v0
# -

V_list = list()
for ℓ_i in ℓ_list:
    V_list.append(V_dict[ℓ_i][12, -1]/9000 )
plt.plot(ℓ_list,  np.array(V_list)+ℓ_list )
# plt.plot(ℓ_list, -ℓ_list)
plt.xlabel('ℓ')
plt.title('ϕ')

V_dict[ℓ_list[0]][25, -1]/9000  , V_dict[ℓ_list[1]][25, -1]/9000, 

V_list

ℓ_list

# $$
#  V( \ell)  = \min_{\ell \geqslant 0} \tilde V(\ell) + \ell r
# $$
#
# $$
#  - \frac{\partial V}{\partial \ell} (\ell) = r
# $$

ℓ_list[1] - ℓ_list[0]

plt.plot(y_mat[:,0], - (V_dict[ℓ_list[1]][:,-1] - V_dict[ℓ_list[0]][:,-1])/ℓ_step/θ)
plt.title('r')
plt.xlabel('y')
# plt.savefig("r.png", dpi=148, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches="tight", pad_inches=0.1,)

- (V_dict[ℓ_list[1]][12,-1] - V_dict[ℓ_list[0]][12,-1])/ℓ_step

# plt.plot(y_mat[:,0], V_dict[0][:,-1])
plt.plot(y_mat[:,0], V_dict[ℓ][:,-1])
plt.plot(y_mat[:,0], V_dict[ℓ+ℓ_step][:,-1])
plt.title('ϕ')
plt.xlabel('y')

5/9000, 20/9000
