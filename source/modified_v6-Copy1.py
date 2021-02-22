# -*- coding: utf-8 -*-
"""
module for modified model
"""
import time
import numpy as np
import SolveLinSys
import pickle
from supportfunctions import *
from global_parameters import *
######################global variable
delta = .01
eta = .032
mu2 = 1.86/1000
sigma_z_100 = .21/1000
gamma_low = .012
gamma_high = .024
rho = .9

def compute_sigma2(rho, sigma_z, mu_2):
    """
    compute_sigma2
    """
    return np.sqrt(2*sigma_z**2*rho/mu_2)


sigma2_100 = compute_sigma2(rho, sigma_z_100, mu2)
xi_m = 1000
xi_a = 1000
# state variable
numz = 50
z2_min = mu2 - 3*sigma_z_100
z2_max = mu2 + 3*sigma_z_100
z = np.linspace(z2_min, z2_max, numz)
hz = z[1]-z[0]

gamma_1 = 0.00017675
gamma_2 = 2*.0022
gamma_bar = 2
gamma2pList = np.array([0, 2*0.0197])

numy = 50
y_min = 0
y_max = 3000
y = np.linspace(y_min, y_max, numy)
hy = y[1] - y[0]

# specify number of damage function
numDmg = 2
PI0 = np.ones(numDmg)/numDmg

gamma2pMat = np.zeros((numDmg, numz, numy))
gamma2pMat[0] = gamma2pList[0]
gamma2pMat[1] = gamma2pList[1]

(z_mat, y_mat) = np.meshgrid(z, y, indexing = 'ij')
stateSpace = np.hstack([z_mat.reshape(-1,1, order='F'), y_mat.reshape(-1,1,order='F')])

def PDESolver_2d(stateSpace, A, B_r, B_f, C_rr, C_ff, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':

        if smartguess:
            iters = 1
        else:
            iters = 400000

        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)
        return out

# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-8
epsilon = .3

PI = PI0
PIThis = np.zeros((numDmg, numz, numz))
PILast = np.zeros((numDmg, numz, numz))
while episode == 0 or FC_Err > tol:
    print('Episode:{:d}'.format(episode))
    if episode ==0:
        v0 = - delta*eta*y_mat*z_mat
    else:
        vold = v0.copy()


    # calculating partial derivatives
    v0_dz = finiteDiff(v0,0,1,hz)
    v0_dzz = finiteDiff(v0,0,2,hz)
    #v0_dF[v0_dF < 1e-16] = 1e-16
    # With only v0_dFF[v0_dFF < 1e-16] = 0
    v0_dy = finiteDiff(v0,1,1,hy)
    v0_dyy = finiteDiff(v0,1,2,hy)
    print(v0_dy)
    # updating controls
    if episode == 0:
        PIThis = np.ones((numDmg, numz, numy))/numDmg
        PILast = PIThis
    else:
        PISum = PILast[0]*np.exp(-1/xi_a*(eta-1)*gamma2pMat[0]*(y_mat*z_mat>=gamma_bar)*((y_mat*z_mat - gamma_bar )*(z_mat*e + y_mat*(-rho*(z_mat - mu2) ) + y_mat*np.sqrt(z_mat)*sigma2_100*h2 ) + 0.5*z_mat*y_mat**2*sigma2_100**2 )  ) + PILast[1]*np.exp(-1/xi_a*(eta-1)*gamma2pMat[1]*(y_mat*z_mat>=gamma_bar)*((y_mat*z_mat - gamma_bar )*(z_mat*e + y_mat*(-rho*(z_mat - mu2) ) + y_mat*np.sqrt(z_mat)*sigma2_100*h2 ) + .5*z_mat*y_mat**2*sigma2_100**2 )  )
        print(PISum.shape)
        PIThis[0] = PILast[0]*np.exp(-1/xi_a*(eta-1)*gamma2pList[0]*(y_mat*z_mat>=gamma_bar)*((y_mat*z_mat - gamma_bar )*(z_mat*e + y_mat*(-rho*(z_mat - mu2) ) + y_mat*np.sqrt(z_mat)*sigma2_100*h2 ) + .5*z_mat*y_mat**2*sigma2_100**2 )  )/PISum
        PIThis[1] = PILast[1]*np.exp(-1/xi_a*(eta-1)*gamma2pList[1]*(y_mat*z_mat>=gamma_bar)*((y_mat*z_mat - gamma_bar )*(z_mat*e + y_mat*(-rho*(z_mat - mu2) ) + y_mat*np.sqrt(z_mat)*sigma2_100*h2 ) + .5*z_mat*y_mat**2*sigma2_100**2 )  )/PISum
    print("entering control update")
    compute_start = time.time()
    e =  - delta*eta/(v0_dy+(eta-1)*(gamma_1 + gamma_2*y_mat*z_mat + np.sum(gamma2pMat*PIThis, axis=0)*(y_mat*z_mat - gamma_bar)*(y_mat*z_mat>=gamma_bar) )*z_mat)
    e[e<0] = 1e-300
    h2 = - (v0_dz*np.sqrt(z_mat)*sigma2_100+ (eta -1 )*(gamma_1 + gamma_2*y_mat*z_mat + np.sum(PIThis*gamma2pMat, axis=0)*(y_mat*z_mat - gamma_bar)*(y_mat*z_mat>= gamma_bar))*y_mat*np.sqrt(z_mat)*sigma2_100)/xi_m
    print(np.min(e))
    # HJB coefficient
    A =  - delta*np.ones(y_mat.shape)
    B_z = - rho*(z_mat - mu2) + np.sqrt(z_mat)*sigma2_100*h2
    B_y = e
    C_zz = z_mat*sigma2_100**2/2
    C_yy = np.zeros(z_mat.shape)
    D =  delta*eta*np.log(e) - (1 - eta)*((gamma_1 + gamma_2*y_mat*z_mat +np.sum(PIThis*gamma2pMat, axis=0)*(y_mat*z_mat - gamma_bar )*(y_mat*z_mat>=gamma_bar))*(z_mat*e + y_mat*(-rho*(z_mat - mu2)) + y_mat*np.sqrt(z_mat)*sigma2_100*h2) + 0.5*(gamma_2 + np.sum(PIThis*gamma2pMat, axis=0)*(y_mat*z_mat>=gamma_2) ) *z_mat*y_mat**2*sigma2_100**2) + xi_m*h2**2/2 + xi_a*np.sum(PIThis*(np.log(PIThis) -np.log(PILast)), axis=0)
    print("End of update, takes: {}".format(time.time() - compute_start))
    print('Entering PDE solver...')
    solve_start = time.time()
    out = PDESolver_2d(stateSpace, A, B_z, B_y, C_zz, C_yy, D, v0,
                       epsilon, solverType = 'False Transient')

    out_comp = out[2].reshape(v0.shape,order = "F")

    PDE_rhs = A*v0 + B_z*v0_dz + B_y*v0_dy + C_zz*v0_dzz + C_yy*v0_dyy + D

    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - v0)))
    #     if episode % 1 == 0:
    print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode,
          PDE_Err, FC_Err, out[0], out[1]))
    episode += 1
    v0 = out_comp
    PILast = PIThis
    print("End of PDE solver, takes time: {}".format(time.time() - solve_start))


print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]))
print("--- %s seconds ---" % (time.time() - start_time))

solution_20_100_v2 = dict(e = e, phi = v0, dphidz = v0_dz, dphidy = v0_dy)

# restore results
from datetime import datetime
nowtime = datetime.now()
time_store = nowtime.strftime("%m%d_%H:%M")
dataFile = '../data/solution/solu_modified_v6_{}*{}_{}'.format(numz, numy, time_store)
with open(dataFile, 'wb') as handle:
    pickle.dump(solution_20_100_v2, handle, protocol=pickle.HIGHEST_PROTOCOL)
