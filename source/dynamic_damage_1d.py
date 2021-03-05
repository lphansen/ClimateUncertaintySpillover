# #!/bin/env python
"""
module for dynamic damages
1d version
"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
from utilities import dLambda
import global_parameters as gp
from supportfunctions import finiteDiff
import matplotlib.pyplot as plt
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg

# global variables
delta = gp.DELTA
eta = gp.ETA
mu2 = gp.MU_2
sigma_z = gp.SIGMA_Z
rho =gp.RHO
sigma2 = np.sqrt(2*sigma_z**2*rho/mu2)
gamma_1 = 0.00017675
gamma_2 = 2*.0022
gamma_bar = 2
gamma2pList = np.array([0, 2*0.0197])
v_n = eta - 1

# grid setting
z = mu2
numy_bar = 20
y_min = 0
y_bar = gamma_bar
y_max = 10
hy = (y_bar - y_min)/numy_bar
y_grid = np.arange(y_min, y_max+hy, hy)


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
def get_coeff(y_grid, A, By, Cyy, D, v0, bound_var, bounded=False, epsilon=.3):
    numy, = y_grid.shape
    hy = y_grid[1] - y_grid[0]
    LHS = np.zeros((numy, numy))
    RHS = np.zeros(numy)
    RHS += - D - 1/epsilon*v0
    for i in range(numy):
        LHS[i,i] += A[i] - 1/epsilon
        if i == 0:
            LHS[i,i+1] += By[i]*(1/hy)
            LHS[i,i] += - By[i]*(1/hy)
        if i == numy -1:     
            if bounded == True:
                LHS[i] = 0
                LHS[i,i] = 1
                RHS[i] = bound_var
            else:
                LHS[i,i] += By[i]*(1/hy)
                LHS[i,i-1] += - By[i]*(1/hy)
        else:
            LHS[i,i+1] += By[i]*(1/hy)*(By[i]>0) + Cyy[i]/(hy**2)
            LHS[i,i] += By[i]*((-1/hy)*(By[i]>0) + (1/hy)*(By[i]<=0)) - 2*Cyy[i]/(hy**2)
            LHS[i,i-1] += By[i]*(-1/hy)*(By[i]<=0) + Cyy[i]/(hy**2)
    return LHS, RHS


def solve_ode(y_grid, A, By, Cyy, D, v0, bound_var=0, bounded=False, epsilon=.3):
    LHS, RHS = get_coeff(y_grid, A, By, Cyy, D, v0, bound_var, bounded, epsilon)
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
        rhs = A*phi_grid + By*v0_dy + D
        rhs_error = np.max(abs(rhs))
        error = np.max(abs((phi_grid-v0)/epsilon))
        v0 = phi_grid
        episode += 1
        print('Episode: {:d}\t lhs error: {:.12f}\t rhs error: {:.12f}'.format(episode,error,rhs_error))
    return v0, ems


gamma2pList, y_grid

# +
dmg_weights_list = np.array([[1,0],[0,1]])
modelParams = (delta, eta, mu2, sigma2, rho, v_n)
dmgParamsPart = (gamma_1, gamma_2, gamma2pList, gamma_bar)

v_dict = dict()
ems_dict = dict()
for i in range(len(dmg_weights_list)):
    dmgParams = (gamma_1, gamma_2, gamma2pList, gamma_bar, dmg_weights_list[i])
    v_dict[i], ems_dict[i] = false_transient_1d(y_grid, z, dmgParams, modelParams)
# -

plt.plot(y_grid, v_dict[0])
plt.plot(y_grid, v_dict[1])
# plt.plot(y_grid, v_dict[2])

v_dict[0][-1]

bd = (v_dict[0][numy_bar]+v_dict[1][numy_bar])/2
bd

# +
dmg_weights = np.array([0.5, 0.5])
dmgParams = (gamma_1, gamma_2, gamma2pList, gamma_bar, dmg_weights)
modelParams = (delta, eta, mu2, sigma2, rho, v_n)

y_grid_cap = np.linspace(0,10,100)

dmgParams, modelParams, bd
# -

v, ems = false_transient_1d(
    y_grid=y_grid_cap, z=z, dmg_params=dmgParams, model_params=modelParams, 
    bound_var=bd, bounded=False, max_iter=1_000)

plt.plot(y_grid[:numy_bar+1],v)
plt.plot(y_grid[numy_bar:], (v_dict[0][numy_bar:] + v_dict[1][numy_bar:])/2)
plt.vlines(2, ymin=-.04, ymax=.1, color="black", linestyle="dashed")

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

np.pi


def false_transient_jump(
    y_grid, z, v_bar,
    dmg_params, model_params, v0=None, bounded=False, bound_var=0, 
    epsilon=.5, tol = 1e-8, max_iter=10_000,
):
    gamma_1, gamma_2, gamma2pList, gamma_bar, dmg_weights = dmg_params
    delta, eta, mu2, sigma2, v_n , sigma = model_params
    numy, = y_grid.shape
    hz = y_grid[1] - y_grid[0]
    dlambda = gamma_1 + gamma_2*y_grid\
    + np.sum(gamma2pList*dmg_weights,axis=0)*(y_grid-gamma_bar)*(y_grid>gamma_bar)
    # initiate v and control
    ems = delta*eta
    error = 1
    episode = 0
    if v0 == None:
        v0 = - delta*eta*y_grid
    while error > tol:
        v0_old = v0.copy()
        v0_dy = derivative_1d(v0,1,hy, upwind=True)
        # control
        ems = (-delta*eta/(v0_dy*z + v_n*dlambda*z))*.5 + ems*0.5
        ems[ems<=0] = 1e-16
        A = -delta*np.ones(y_grid.shape) \
        - 1/(np.sqrt(np.pi*2)*sigma)*np.exp(-(y_grid - gamma_bar)**2/(2*sigma**2))
        By = z*ems
        Cyy = np.zeros(y_grid.shape)
        D = delta*eta*np.log(ems) + v_n*dlambda*z*ems\
        + 1/(np.sqrt(np.pi*2)*sigma)*np.exp(-(y_grid - gamma_bar)**2/(2*sigma**2))*v_bar
        # solve for ODE
        phi_grid = solve_ode(y_grid, A, By, Cyy, D, v0, bound_var, bounded, epsilon)
        rhs = A*phi_grid + By*v0_dy  + D
        rhs_error = np.max(abs(rhs))
        error = np.max(abs((phi_grid-v0)/epsilon))
        v0 = phi_grid
        episode += 1
        print('Episode: {:d}\t lhs error: {:.12f}\t rhs error: {:.12f}'.format(episode,error,rhs_error))
    return v0, ems


v_bar = np.average([v_dict[0], v_dict[1]], axis=0, weights=[.5,.5])

v_bar_short = v_bar[:numy_bar]
y_grid_short = y_grid[:numy_bar]

modelParams, z, dmgParams, y_grid_short.shape,z, v_bar_short[-1]

# +
dmg_weights = np.array([0.5, 0.5])
dmgParams = (gamma_1, gamma_2, gamma2pList, gamma_bar, dmg_weights)

v_try = dict()
for sigma in [gamma_bar/10, gamma_bar/50, gamma_bar/100, gamma_bar/1000, 100*gamma_bar]:
    modelParams = (delta, eta, mu2, sigma2, v_n, sigma)
    v_try[sigma], ems = false_transient_jump(y_grid=y_grid_short, z=z, v_bar=v_bar_short, dmg_params=dmgParams, model_params=modelParams, bound_var=bd, bounded=False)

# +
plt.figure(figsize=(10,8))
for sigma in [ gamma_bar/10, gamma_bar/50, gamma_bar/100, gamma_bar/1000, 100*gamma_bar]:
    plt.plot(y_grid_short, v_try[sigma], label="$\\sigma = {}$".format(sigma))
    

# plt.plot(y_grid[numy_bar:], v_dict[0][numy_bar:], linestyle="dotted", color="black", label="low damage")
plt.plot(y_grid[numy_bar:], (v_dict[0][numy_bar:] + v_dict[1][numy_bar:])/2, color ="black", label="equally weighted")
# plt.plot(y_grid[numy_bar:], v_dict[1][numy_bar:], linestyle="--", color="black", label="high damage")
plt.vlines(2, ymin=-0.02, ymax=.06, color="black", linewidth=1)
plt.legend(loc=3, fontsize=18)
plt.xlim(0,4)
plt.ylim(-.02,.06)
plt.xlabel("y", fontsize=18)
plt.ylabel("$\\phi$", rotation=0, fontsize=18)
plt.title('$\phi(y)$', fontsize=18)
# plt.savefig("phi_rho.png", dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches='tight', pad_inches=0.1,
#         metadata=None )
# -

y_dense = np.arange(0,2,1/20000)
def get_intensity(y_grid, sigma, gamma_bar=2):
    temp = 1/(np.sqrt(np.pi*2)*sigma)*np.exp(-(y_grid - gamma_bar)**2/(2*sigma**2))
#     temp *= v_bar - v_new
    return temp


y_dense.shape

inten = dict()
for sigma in [gamma_bar/10, gamma_bar/50, gamma_bar/100]:
    inten[sigma] = get_intensity(y_dense,  sigma)

plt.plot(v_bar_short)

inten

for sigma in [gamma_bar/10, gamma_bar/50, gamma_bar/100]:
    plt.plot(y_dense,inten[sigma], label="$\\sigma = {}$".format(sigma))
# plt.xlim(1.988 ,2)
plt.legend()
plt.title("intensity function")
plt.xlabel("y")
# plt.savefig("inten.png", dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', format=None,
#         transparent=False, bbox_inches='tight', pad_inches=0.1,
#         metadata=None)
