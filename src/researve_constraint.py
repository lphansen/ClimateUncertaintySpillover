# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
from utilities import dLambda
# from solver_1d import false_transient_one_iteration_python
from solver_2d import false_transient_one_iteration_python
import global_parameters as gp
from supportfunctions import finiteDiff
import matplotlib.pyplot as plt
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg
from derivative import derivatives_2d
import time
import SolveLinSys
import time
from supportfunctions import PDESolver_2d, finiteDiff
import global_parameters as gp
from utilities import dLambda, ddLambda, weightOfPi, relativeEntropy, weightPI, damageDrift, zDrift


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
ȳ = 2

θ_list = pd.read_csv("../data/model144.csv", header=None)[0].to_numpy()
θ_list = θ_list/1000
σy = 1.2*np.mean(θ_list)

γ1 = 0.00017675
γ2 = 2*0.0022
γ2p = np.array([0, 2*0.0197, 2*0.3853])
dmg_weight = np.array([1, 0, 0])
dΛ = γ1 + γ2*y_grid + np.average(γ2p, weights=dmg_weight)*(y_grid - γbar)*(y_grid>γbar)
# -

# # With reserve constraint
# $$
# \begin{aligned}
# 0 = \max_{\tilde e}\min_h & \quad b \delta\eta \log\tilde e \\
# &  + b \frac{\xi_m}{2} h'h + \frac{dV}{dy} \tilde e (\sum_i \hat\pi^a_i\theta_i + \sigma_y h) - \delta b \frac{dV}{db} + \frac{1}{2}\frac{d^2V}{dy^2}|\sigma_y|^2(\tilde e)^2\\
#     & + b(\eta -1 )\cdot(\gamma_1 + \gamma_2 y + \gamma_3^m (y - \bar y)\mathbb{I}\{y >\bar y\})\cdot\tilde e\cdot (\sum_i \hat \pi_i^a\theta_i + \sigma_y h) + b\frac{1}{2}(\eta -1)(\gamma_2 + \gamma_3 \mathbb{I}\{y>\bar y\})(\tilde e)^2|\sigma_y|^2 - \ell \tilde e \\
#         & + \xi_a \sum_i \hat \pi_i^a (\log \hat \pi_i^a - \log \pi_i^a)
# \end{aligned}
# $$
#
# $$
# h^* = - \frac{\frac{dV}{dy} + b\frac{(\eta-1)}{\delta}(\gamma_1 + \gamma_2 y + \gamma_3 (y - \bar y)\mathbb{I}\{y>\bar y \})}{b \xi_m} \cdot \tilde e \sigma_y
# $$
# Plug back into the HJB:
# $$
# \begin{aligned}
# 0 = \max_{\tilde e } \quad & b \delta\eta \log \tilde e + \frac{dV}{dy} \tilde e \sum_i \hat \pi^a_i\theta_i - \delta b \frac{dV}{db} + \frac{1}{2}\frac{d^2V}{dy^2}|\sigma_y|^2(\tilde e)^2 \\ 
# & +  b\frac{(\eta -1 )}{\delta}\cdot(\gamma_1 + \gamma_2 y + \gamma_3^m (y - \bar y)\mathbb{I}\{y > \bar y\})\cdot\tilde e\cdot \sum_i \hat\pi^a_i\theta_i  + b\frac{1}{2}\frac{(\eta -1)}{\delta}(\gamma_2 + \gamma_3^m \mathbb{I}\{y>\bar y\})(\tilde e)^2|\sigma_y|^2 \color{red}{ - \ell \tilde e}\\
# & -\frac{1}{2b\xi_m}\left(\frac{dV}{dy} + b\frac{(\eta-1)}{\delta}(\gamma_1 + \gamma_2 y + \gamma_3(y-\bar y)\mathbb{I}\{\{y>\bar y\})\right)^2 (\tilde e)^2 |\sigma_y|^2\\
# \\
#  & + \xi_a \sum_i \hat \pi_i^a \left(\log \hat \pi_i^a - \log \pi_i^a\right)
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
# \begin{aligned}
#   &\left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b\frac{(\eta -1)}{\delta}(\gamma_1 + \gamma_2 y + \gamma_3^m (y - \bar y)\mathbb{I}\{y > \bar y\}) \right)^2 + \frac{d^2V}{dy^2} + b\frac{(\eta -1)}{\delta} (\gamma_2 + \gamma_3^m \mathbb{I}\{y > \bar y\}) \right] \cdot|\sigma_y|^2\cdot(\tilde e)^2 \\
#   \\
#   + &\left[\sum_i \hat \pi_i^a \theta_i\left(\frac{dV}{dy}  + b\frac{(\eta - 1)}{\delta}(\gamma_1 + \gamma_2 y + \gamma_3^m (y - \bar y) \mathbb{I}\{y > \bar y\}) \right) - \ell \right]\tilde e + b \eta = 0
#  \end{aligned}
# $$
#
# $$
# A =  \left[- \frac{1}{b\xi_m}\left( \frac{dV}{dy}  + b\frac{(\eta -1)}{\delta}(\gamma_1 + \gamma_2 y) \right)^2 + \frac{d^2V}{dy^2}+ b\frac{(\eta -1)}{\delta} \gamma_2 \right] \cdot|\sigma_y|^2
# $$
#
# $$
# B = \left[\frac{dV}{dy} + b\frac{(\eta - 1)}{\delta}(\gamma_1 + \gamma_2 y)\right]\theta - \ell
# $$ 
#
# $$
# C = b \delta\eta
# $$
# And
# $$
# \tilde e^* = \frac{-B -  \sqrt{B^2 - 4AC}}{2A}
# $$

b_grid = np.linspace(1e-10, 1, 50)
numy_bar = 50
y_min = 0
y_max = 4
hy = (ȳ - y_min)/numy_bar
y_grid = np.arange(y_min, y_max+hy, hy)
# y_grid = np.linspace(1e-10, 4, 100)
# mesh grid and construct state space
(y_mat, b_mat) = np.meshgrid(y_grid, b_grid, indexing = 'ij')
stateSpace = np.hstack([y_mat.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])
hb = b_grid[1] - b_grid[0]
hy = y_grid[1] - y_grid[0]

# +
# 2 state HJB with constraints
(y_mat, b_mat) = np.meshgrid(y_grid, b_grid, indexing = 'ij')
stateSpace = np.hstack([y_mat.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])
hb = b_grid[1] - b_grid[0]
hy = y_grid[1] - y_grid[0]

# ℓ = 1e-12
ξₘ = 1000
ξa = 1/100
γ3_list = np.array([0, 2*0.0197, 2*0.3853])
π_p = np.array([1/3, 1/3, 1/3])
πa_o = np.ones((len(θ_list), len(y_grid), len(b_grid)))/len(θ_list)

ℓ1 = 1e-4
ℓ2 = 1e-3
ℓ_list = [0 , ℓ1, ℓ1*(1+0.01), ℓ2, ℓ2*(1 + 0.01)]

tol = 1e-8
ϵ = 3
v_dict = dict()
ems_dict = dict()
for i in range(len(γ3_list)):
    π_p = np.zeros(len(γ3_list))
    π_p[i] = 1
    dΛ = γ1 + γ2*y_mat + π_p@γ3_list*(y_mat - ȳ)*(y_mat >ȳ)
    ddΛ = γ2 + π_p@γ3_list*(y_mat >ȳ)
    
    v_list = list()
    ems_list = list()
    for ℓ_i in ℓ_list:
        episode = 0
        lhs_error = 1
        πa = πa_o
        ems = - η/(b_mat*(η-1)/δ*dΛ*(πa.T@θ_list).T)
        ems_old = ems
        while lhs_error > tol:
            if episode ==0:
                v0 =  - η*(y_mat + y_mat**2)
            else:
                vold = v0.copy()
            v0_dy = derivatives_2d(v0,0,1,hy)
            v0_dyy = derivatives_2d(v0,0,2,hy)
            v0_db = derivatives_2d(v0,1,1,hb)
            # updating controls
            temp = v0_dy + b_mat*(η-1)/δ*dΛ
            weight = np.array([-1/ξa*temp*ems*θ for θ in θ_list])
            weight = weight - np.max(weight, axis=0)
            πa = πa_o*np.exp(weight)
#         πa[πa<1e-15] = 1e-15
            πa = πa/np.sum(πa, axis=0)
            print(np.min(ems))
            
            a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2 + b_mat*(η - 1)/δ*ddΛ*σy**2
            b = (πa.T@θ_list).T*temp  - ℓ_i
            c = η*b_mat
            Δ = b**2 - 4*c*a
            Δ[Δ<0] = 0
            ems_new =  -b/(2*a) - np.sqrt(Δ)/(2*a)
            ems_new[ems_new <= 0] = 1e-15
            ems = ems_new
            # HJB coefficient
            A =  np.zeros(y_mat.shape)
            B_y =  ems*(πa.T@θ_list).T
            B_b = - δ*b_mat
            C_yy = ems**2*σy**2/2
            C_bb = np.zeros(y_mat.shape)
            D = b_mat*η*np.log(ems) + b_mat*(η-1)/δ*(dΛ*ems*(πa.T@θ_list).T + 1/2*ddΛ*ems**2*σy**2)\
            - ℓ_i*ems - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
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
    #         print("End of PDE solver, takes time: {}".format(time.time() - solve_start))

        v_list.append(v0)
        ems_list.append(ems)
    
    v_list = np.array(v_list)
    ems_list = np.array(ems_list)
    v_dict[γ3_list[i]] = v_list
    ems_dict[γ3_list[i]] = ems_list
# -

plt.plot(ems_dict[0][0][:,-1])
plt.plot((ems_dict[0][1][:,-1] + ems_dict[0][2][:,-1] )/2)

ξp = 5
dmg_weight = np.array([1/3,1/3,1/3])
ϕ_bound = list()
for i in range(len(ℓ_list)):
    ϕ_weight_i =  np.array([ np.exp(- 1/ξp*v_dict[γ3][i])for γ3 in γ3_list] ) 
    ϕ_weight = np.average(ϕ_weight_i, axis=0, weights=dmg_weight)
    ϕ_bound_i = -ξp*np.log(ϕ_weight)
    ϕ_bound.append(ϕ_bound_i)
ϕ_bound = np.array(ϕ_bound)

v_dict[0].shape, ϕ_bound.shape

# +
(y_mat_cap, b_mat) = np.meshgrid(y_grid[:numy_bar+1], b_grid, indexing = 'ij')
# stateSpace = np.hstack([y_mat_cap.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])
num_y = len(y_grid[:numy_bar+1])
num_b = len(b_grid)
# ℓ_list = np.array([ℓ, ℓ+ℓ_step])
πa_o = np.ones((len(θ_list), len(y_grid[:numy_bar+1]), len(b_grid)))/len(θ_list)
dΛ = γ1 + γ2*y_mat_cap
ddΛ = γ2
vjp_list = list()
emsjp_list = list()
ϵ = 5
tol = 1e-6
max_iter = 5_000

for i in range(len(ℓ_list)):
    episode = 0
    lhs_error = 1
    πa = πa_o
    ems = - η/(b_mat*(η-1)/δ*dΛ*(πa.T@θ_list).T)
    ems_old = ems
    v0 =  - η*(y_mat_cap + y_mat_cap**2)
    while lhs_error > tol and episode < max_iter:
        v0_old = v0.copy()
        v0_dy = derivatives_2d(v0,0,1,hy)
        v0_dyy = derivatives_2d(v0,0,2,hy)
        v0_db = derivatives_2d(v0,1,1,hb)
        # updating controls
        temp = v0_dy + b_mat*(η-1)/δ*dΛ
        weight = np.array([-1/ξa*temp*ems*θ for θ in θ_list])
        weight = weight - np.max(weight, axis=0)
        πa = πa_o*np.exp(weight)
#         πa[πa<1e-15] = 1e-15
        πa = πa/np.sum(πa, axis=0)
        print(np.min(ems))

        a = v0_dyy*σy**2 - temp**2/(b_mat*ξₘ)*σy**2 + b_mat*(η - 1)/δ*ddΛ*σy**2
        b = (πa.T@θ_list).T*temp  - ℓ_list[i]
        c = η*b_mat
        Δ = b**2 - 4*c*a
        Δ[Δ<0] = 0
        ems_new =  -b/(2*a) - np.sqrt(Δ)/(2*a)
        ems_new[ems_new <= 0] = 1e-15
        ems = ems_new*1. + ems_old*0.
        # HJB coefficient
        A =  np.zeros(y_mat_cap.shape)
        B_y =  ems*(πa.T@θ_list).T
        B_b = - δ*b_mat
        C_yy = ems**2*σy**2/2
        C_bb = np.zeros(y_mat_cap.shape)
        D = b_mat*η*np.log(ems) + b_mat*(η-1)/δ*(dΛ*ems*(πa.T@θ_list).T + 1/2*ddΛ*ems**2*σy**2)\
        - ℓ_list[i]*ems - temp**2*ems**2*σy**2/(2*b_mat*ξₘ)
        # PDE solver
        phi_mat = false_transient_one_iteration_python(A, B_y, B_b, C_yy, C_bb, D, v0, ϵ, 
                                                   hy, hb, 
                                                   bc=(np.zeros(num_b), ϕ_bound[i][numy_bar], np.zeros(num_y), np.zeros(num_y)), 
                                                   impose_bc=(False, True, False, False))
        rhs = A*phi_mat + B_y*v0_dy + B_b*v0_db + C_yy*v0_dyy + D
        rhs_error = np.max(abs(rhs))
        lhs_error = np.max(abs((phi_mat-v0_old)/ϵ))
        v0 = phi_mat
        ems_old = ems
        episode += 1
        print('Episode: {:d}\t lhs error: {:.12f}\t rhs error: {:.12f}'.format(episode, lhs_error, rhs_error))
#         print("End of PDE solver, takes time: {}".format(time.time() - solve_start))

    vjp_list.append(v0)
    emsjp_list.append(ems)

vjp_list = np.array(vjp_list)
emsjp_list = np.array(emsjp_list)
# -

# $$
#  V( \ell)  = \min_{\ell \geqslant 0} \tilde V(\ell) + \ell r
# $$
#
# $$
#  - \frac{\partial V}{\partial \ell} (\ell) = r
# $$

plt.plot(- (vjp_list[2][:,-1] - vjp_list[1][:,-1])/(ℓ_list[1]*0.01) )
plt.plot(- (vjp_list[4][:,-1] - vjp_list[3][:,-1])/(ℓ_list[3]*0.01) )


def compute_res_ems(vjp_list, emsjp_list, ℓ_list):
    num_ell_total, numy_cap, num_b = vjp_list.shape
    num_ell = int(num_ell_total/2)
    res_grid = np.zeros((num_ell, numy_cap))
    ℓ_list_new = np.zeros(num_ell)
    ems_list_new = np.zeros((num_ell, numy_cap))
    for i in range(num_ell):
        res_grid[i] =  -(vjp_list[2*i+1][:,-1] - vjp_list[2*i][:,-1])/(ℓ_list[i*2]*0.01) 
        ℓ_list_new[i] = (ℓ_list[2*i +1] + ℓ_list[2*i])/2
        ems_list_new[i] = (emsjp_list[2*i +1][:,-1] + emsjp_list[2*i][:,-1])/2
    return res_grid, ℓ_list_new, ems_list_new


res_grid, ℓ_list_new, ems_list_new = compute_res_ems(vjp_list[1:], emsjp_list[1:], ℓ_list[1:])

# res_grid.shape
plt.plot(res_grid.T)
plt.show()

plt.plot(ems_list_new.T)

emsjp_list[0].shape

# +
from scipy import interpolate
def simulate_res_ems(y_grid_cap, res_grid, ems_list_new, T=100, dt=1):
    periods = int(T/dt)
    num_ell = len(ems_list_new)
    yt = np.zeros((num_ell,periods))
    rt = np.zeros((num_ell,periods))
    ems_t = np.zeros((num_ell,periods))
    ems_func = interpolate.interp1d(y_grid_cap, ems_list_new, )
    res_func = interpolate.interp1d(y_grid_cap, res_grid, )
#     f_π = interpolate.interp2d(y2_grid, y1_grid, πa, )
    y= np.ones(num_ell)
    for t in range(periods):
        if y.any() > np.max(y_grid_cap):
            break
        ems_point = ems_func(y).diagonal()
        res_point = res_func(y).diagonal()
#         π_list = f_π(y2, y1)
        ems_t[:,t] = ems_point
        rt[:, t] = res_point
        yt[:, t] = y
        y += ems_point*np.mean(θ_list)*dt
    return yt, ems_t, rt

def simulate_ems(y_grid_cap, ems, T=100, dt=1):
    periods = int(T/dt)
    yt = np.zeros(periods)
    ems_t = np.zeros(periods)
    ems_func = interpolate.interp1d(y_grid_cap, ems )
#     f_π = interpolate.interp2d(y2_grid, y1_grid, πa, )
    y= 1
    for t in range(periods):
        if y > np.max(y_grid_cap):
            break
        ems_point = ems_func(y)
        ems_t[t] = ems_point
        yt[t] = y
        y += ems_point*np.mean(θ_list)*dt
    return yt, ems_t


# -

yt_null, ems_t_null = simulate_ems(y_grid[:numy_bar+1], emsjp_list[0][:,-1])
yt, ems_t, rt = simulate_res_ems(y_grid[:numy_bar+1], res_grid, ems_list_new, T=100, dt=1)

plt.plot(ems_t.T)
plt.plot(ems_t_null)
plt.legend(["ℓ=1e-4", "ℓ=1e-3","no constraint"])

plt.plot(rt.T)
plt.legend([""])

σy, 1.2*1.86/1000, γ1

σy/np.sqrt(100)

1.2*np.mean(θ_list)

σy/5
