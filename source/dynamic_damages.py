# -*- coding: utf-8 -*-
"""
module for dynamic damages experiment
currently with three z's
"""
import time
import numpy as np
import pandas as pd
import SolveLinSys
import pickle
from supportfunctions import PDESolver_2d
import global_parameters as gp
from utilities import dLambda, ddLambda, weightPITemp, relativeEntropy, zDrift, damageDrift
from joblib import Parallel, delayed
######################global variable
delta = gp.DELTA
eta = gp.ETA
mu2 = gp.MU_2
sigma_z = gp.SIGMA_Z
rho =gp.RHO

sigma2 = np.sqrt(2*sigma_z**2*rho/mu2)
# sigma2 = sigma2
xi_m = 1000
xi_a = 1/4000
# state variable

gamma_1 = 0.00017675
gamma_2 = 2*.0022
gamma_bar = 2
gamma2pList = np.array([0, 2*0.0197, 2*0.3853])
v_n = eta - 1


sigma_z = np.sqrt(sigma2**2*mu2/2/rho)
numz = 51
z2_min = mu2 - 4*sigma_z
z2_max = mu2 + 4*sigma_z
z_grid = np.linspace(z2_min, z2_max, numz)
hz = z_grid[1] - z_grid[0]

numy_bar = 21
y_min = 0
y_bar = 2
y_max = 10
hy = (y_bar - y_min)/(numy_bar-1)
y_grid = np.arange(y_min, y_max+hy, hy)
numy = len(y_grid)
# specify number of damage function
numDmg = 3

gamma2pMat = np.zeros((numDmg, numz, numy))
gamma2pMat[0] = gamma2pList[0]
gamma2pMat[1] = gamma2pList[1]

dmgDist = np.array([.4875, .4875, .025])
dmgParDist = np.hstack([gamma2pList.reshape((numDmg,1)), dmgDist.reshape((numDmg,1))])
# climate models
mu2List = pd.read_csv("../data/model144.csv", header=None).to_numpy()[:,0]/1000
mu2Dist = np.ones_like(mu2List)/len(mu2List)
mu2ParDist = np.hstack([mu2List.reshape((len(mu2List),1)), mu2Dist.reshape((len(mu2List),1))])
# construct model parameters for damage and climate
modelParam = list()
prior = list()
for mu2Var, mu2Prob in mu2ParDist:
    for gamma2p, gamma2pProb in dmgParDist:
        modelParam.append([mu2Var, gamma2p])
        prior.append(mu2Prob*gamma2pProb)
modelParam = np.array(modelParam)
prior = np.array(prior)

(z_mat, y_mat) = np.meshgrid(z_grid, y_grid, indexing = 'ij')
stateSpace = np.hstack([z_mat.reshape(-1,1, order='F'), y_mat.reshape(-1,1,order='F')])


# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-8
epsilon = .5
FC_Err = 1


PIThis = np.zeros((len(modelParam), numz, numy))
for i in range(len(modelParam)):
    PIThis[i] = prior[i]
prior = PIThis

gamma2pMat = np.zeros_like(PIThis)
for i in range(len(modelParam)):
    gamma2pMat[i] = modelParam[i,1]

# dlambda = dLambda(y_mat, 1, gamma_1, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
# ddlambda = ddLambda(y_mat, z_mat, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
def false_transient(
    stateSpace, state_mat, state_step,
    gamma2p, gamma_1=0.00017675, gamma_2=2*0.0022, gamma_bar=2,
    global_params=(.01, .032, .032-1, .9, 1.86/1000, np.sqrt(2*(.42/1000)**2*.9/(1.86/1000))),
    tol = 1e-8, epsilon=.3):

    z_mat, y_mat = state_mat
    hz, hy = state_step
    delta, eta, v_n, rho, mu2, sigma2 = global_params

    mean = -rho*(z_mat - mu2)
    v0 =  -delta*eta*y_mat*z_mat
    e = delta*eta
    episode = 0
    FC_Err = 1

    while FC_Err > tol:
        print('Episode:{:d}'.format(episode))
        vold = v0.copy()
        # calculating partial derivatives
        v0_dz = finiteDiff(v0,0,1,hz)
        v0_dzz = finiteDiff(v0,0,2,hz)
        v0_dy = finiteDiff(v0,1,1,hy)
        v0_dyy = finiteDiff(v0,1,2,hy)
        print(v0_dy)
        # updating controls
        print("entering control update")
        compute_start = time.time()
        # update intermediate terms
        dlambda =  dLambda(y_mat, 1, gamma_1, gamma_2, gamma2p, gamma_bar)
        # dlambda = dLambda(y_mat, z_mat, gamma_1, gamma_2, np.sum(PIThis*gamma2pMat, axis=0), gamma_bar)
        # ddlambda = ddLambda(y_mat, z_mat, gamma_2, np.sum(gamma2pMat*PIThis, axis=0), gamma_bar)
        e =  (- delta*eta/(v0_dy*z_mat + v_n*dlambda*z_mat))*0.5 + e*0.5
        e[e<0] = 1e-16
        print(np.min(e))
        # HJB coefficient
        A =  - delta*np.ones(y_mat.shape)
        B_z = -rho*(z_mat - mu2)
        B_y = e*z_mat
        C_zz = z_mat*sigma2**2/2
        C_yy = np.zeros(z_mat.shape)
        D =  delta*eta*np.log(e) + v_n*dlambda*z_mat*e

        # print(h2)
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
        # PILast = PIThis
        print("End of PDE solver, takes time: {}".format(time.time() - solve_start))


    print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]))
    print("--- %s seconds ---" % (time.time() - start_time))

    solution = dict(e = e, phi = v0, dphidz = v0_dz, dphidy = v0_dy )
    return solution

solve = false_transient(stateSpace, [z_mat, y_mat], [hz,hy], gamma2p=0)

solve

solu_dynamicdmg = dict()

start = time.time()
if __name__ == "__main__":
    solution = Parallel(n_jobs=16)(delayed(false_transient)(stateSpace, (z_mat, y_mat), (hz, hy), gamma2p)
                                   for gamma2p in gamma2pList)
end = time.time()
print("time:{:.5f}s".format(end - start))

for gamma2p, solu in zip(gamma2pList, solution):
    solu_dynamicdmg[gamma2p] = solu
# restore results
from datetime import datetime
nowtime = datetime.now()
time_store = nowtime.strftime("%m%d_%H:%M")
dataFile = '../data/solution/solu_dynamicdmg_{}*{}_{}'.format(numz, numy,  time_store)
my_shelf = {}
for key in dir():
    if isinstance(globals()[key], (int,float, np.float, str, bool, np.ndarray,list, dict)):
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    else:
        pass

file = open(dataFile, 'wb')
pickle.dump(my_shelf, file)
file.close()


# with open(dataFile, 'wb') as handle:
    # pickle.dump(solution_v6, handle, protocol=pickle.HIGHEST_PROTOCOL)
