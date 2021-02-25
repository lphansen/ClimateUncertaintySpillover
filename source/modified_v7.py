# -*- coding: utf-8 -*-
"""
module for modified model
consider ambiguity only
"""
import time
import numpy as np
import pandas as pd
import SolveLinSys
import pickle
from supportfunctions import PDESolver_2d, finiteDiff
import global_parameters as gp
from utilities import dLambda, ddLambda, weightOfPi, relativeEntropy, weightPI, damageDrift, zDrift
######################global variable
delta = gp.DELTA
eta = gp.ETA
mu2 = gp.MU_2
sigma_z = gp.SIGMA_Z
gamma_low = .012
gamma_high = .024
rho =gp.RHO

def compute_sigma2(rho, sigma_z, mu_2):
    """
    compute_sigma2
    """
    return np.sqrt(2*sigma_z**2*rho/mu_2)


sigma2 = compute_sigma2(rho, sigma_z, mu2)
sigma2 = sigma2/3
xi_m = 1000
xi_a = 1000
# state variable

gamma_1 = 0.00017675
gamma_2 = 2*.0022
gamma_bar = 2
gamma2pList = np.array([0, 2*0.0197])
v_n = eta - 1

numz = 101
z2_min = gp.Z_MIN
z2_max = gp.Z_MAX
z = np.linspace(z2_min, z2_max, numz)
hz = z[1] - z[0]

numy = 50
y_min = 1e-2
y_max = 1500
y = np.linspace(y_min, y_max, numy)
hy = y[1]-y[0]

# specify number of damage function
numDmg = 2
gamma2pMat = np.zeros((numDmg, numz, numy))
gamma2pMat[0] = gamma2pList[0]
gamma2pMat[1] = gamma2pList[1]

# climate models
mu2List = pd.read_excel("../data/model153.xlsx")[0]/1000
# construct model parameters for damage and climate
modelParam = list()
for mu2Var in mu2List:
    for gamma2p in gamma2pList:
        modelParam.append([mu2Var, gamma2p])
modelParam = np.array(modelParam)

# mesh grid and construct state space
(z_mat, y_mat) = np.meshgrid(z, y, indexing = 'ij')
stateSpace = np.hstack([z_mat.reshape(-1,1, order='F'), y_mat.reshape(-1,1,order='F')])

solution_v6 = dict()
# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-8
epsilon = .5
FC_Err = 1

PIThis = np.ones((len(modelParam), numz, numy))/len(modelParam)
prior = PIThis
gamma2pMat = np.zeros_like(PIThis)
for i in range(len(modelParam)):
    gamma2pMat[i] = modelParam[i,1]
dlambda = dLambda(y_mat, z_mat, gamma_1, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
# ddlambda = ddLambda(y_mat, z_mat, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
v0 =  -delta*eta*y_mat*z_mat

while FC_Err > tol:
    print('Episode:{:d}'.format(episode))
    # if episode > 2000 & episode <= 4000:
        # epsilon = .2
    # elif episode > 4000:
        # epsilon = .1
    # else:
        # pass
    if episode ==0:
        v0 =  - delta*eta*y_mat
    else:
        vold = v0.copy()


    # calculating partial derivatives
    v0_dz = finiteDiff(v0,0,1,hz)
    v0_dzz = finiteDiff(v0,0,2,hz)
    #v0_dF[v0_dF < 1e-16] = 1e-16
    # With only v0_dFF[v0_dFF < 1e-16] = 0
    v0_dy = finiteDiff(v0,1,1,hy)
#     v0_dy[v0_dy >0] =0
    v0_dyy = finiteDiff(v0,1,2,hy)
    print(v0_dy)
    # updating controls
    print("entering control update")
    compute_start = time.time()
    e =  - delta*eta/(v0_dy+v_n*dlambda*z_mat)
    e[e<0] = 1e-16
    print(np.min(e))
    PIThis = weightPI(y_mat, z_mat, e, prior , modelParam, v0_dz, rho, gamma_bar, v_n, xi_a, sigma2)
    # print(PIThis[:,0,0],PIThis[:, -1,int(numy/2)], PIThis[:, -1,-1])
    print(np.sum(PIThis[:,-1,-1]))
    # update intermediate terms
    dlambda =  dLambda(y_mat, z_mat, gamma_1, gamma_2, np.sum(PIThis*gamma2pMat, axis=0), gamma_bar)
    # ddlambda = ddLambda(y_mat, z_mat, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
    zDriftMat = zDrift(z_mat, modelParam, rho)
    dmgDriftMat = damageDrift(y_mat, z_mat, e, modelParam, gamma_1, gamma_2, gamma_bar, rho, sigma2)
    # HJB coefficient
    A =  - delta*np.ones(y_mat.shape)
    B_z =  np.sum(PIThis*zDriftMat, axis=0)
    B_y = e
    C_zz = z_mat*sigma2**2/2
    C_yy = np.zeros(z_mat.shape)
    D =  delta*eta*np.log(e) + v_n*np.sum(PIThis*dmgDriftMat, axis=0) + xi_a*relativeEntropy(PIThis, prior)
    print("End of update, takes: {}".format(time.time() - compute_start))
    # PDE solver
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
    # PILast = PIThis
    print("End of PDE solver, takes time: {}".format(time.time() - solve_start))


print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]))
print("--- %s seconds ---" % (time.time() - start_time))

solution_v6 = dict(e = e, phi = v0, dphidz = v0_dz, dphidy = v0_dy,  yGrid=y, zGrid=z, pi =PIThis)

# restore results
from datetime import datetime
nowtime = datetime.now()
time_store = nowtime.strftime("%m%d_%H:%M")
dataFile = '../data/solution/solu_modified_v6_{}*{}_{}_{}'.format(numz, numy, xi_a, time_store)
with open(dataFile, 'wb') as handle:
    pickle.dump(solution_v6, handle, protocol=pickle.HIGHEST_PROTOCOL)
