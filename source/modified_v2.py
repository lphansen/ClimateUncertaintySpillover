"""
module for modified model
"""
import sys, os
sys.path.append(os.getcwd() + '/source')
import time
import numpy as np
import SolveLinSys
from supportfunctions import *
import pickle
from globalParameters import *
from joblib import Parallel, delayed
######################global variable
delta = DELTA
eta = ETA
xi_m = XI_m
mu2 = 1.
sigma_z_50 = .30
sigma_z_100 = .21
sigma_z_200 = .16
gamma_base = .018
gamma_low = .012
gamma_high = .024
rho = .5

def compute_sigma2(rho, sigma_z, mu_2):
    """
    compute_sigma2
    """
    return np.sqrt(2*sigma_z**2*rho/mu_2)


sigma2_100 = compute_sigma2(rho, sigma_z_100, mu2)
tau = .00175*gamma_base

# state variable
numz = 50
z2_min = 1e-5
z2_max = 2
hz = (z2_max - z2_min)/numz
z = np.linspace(z2_min, z2_max, num=numz)

numb = 200
b_min = 1e-6
b_max = 1
hb = (b_max - b_min)/numb
b = np.linspace(b_min, b_max, num=numb)

(z_mat, b_mat) = np.meshgrid(z, b, indexing = 'ij')
stateSpace = np.hstack([z_mat.reshape(-1,1, order='F'), b_mat.reshape(-1,1,order='F')])

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





def falseTransient( ell, epsilon,tolerance, stateVariable, stateSteps, stateSpace, params):
    """False Transient solver with step epsilon and tolerance

    :stateVariablen: TODO
    :ell: TODO
    :v0: TODO
    :epsilon: TODO
    :tolerance: TODO
    :logFile: TODO
    :params: TODO
    :returns: TODO

    """
    z_mat, b_mat = stateVariable
    hz, hb = stateSteps
    tol = tolerance
    delta, eta, mu2, tau, rho, sigma2_100 = params
    episode = 0
    while episode == 0 or FC_Err > tol:
        print('Episode:{:d}'.format(episode))
        if episode ==0:
            v0 = delta*eta*b_mat
        else:
            vold = v0.copy()

        # time-varying dt
        # if episode > 2000:
            # epsilon = 0.1
        # elif episode > 1000:
            # epsilon = 0.2
        # else:
            # pass

        # calculating partial derivatives
        v0_dz = finiteDiff(v0,0,1,hz)
        v0_dzz = finiteDiff(v0,0,2,hz)
        #v0_dF[v0_dF < 1e-16] = 1e-16
        #v0_dFF[v0_dFF < 1e-16] = 0
        v0_db = finiteDiff(v0,1,1,hb)
        v0_dbb = finiteDiff(v0,1,2,hb)
        print(v0_db)
        # updating controls
        Converged = 0
        nums = 0
        e =  b_mat*delta * eta / ( tau*z_mat*b_mat + ell)
        e[e<0] = 1e-300
        h2 = - v0_dz*np.sqrt(z_mat)*sigma2_100/(xi_m*b_mat)
        print(np.min(e))
        # HJB coefficient
        A = np.zeros(z_mat.shape)
        B_z = - rho*(z_mat - mu2) + np.sqrt(z_mat)*sigma2_100*h2
        B_b = - delta*b_mat
        C_zz = z_mat*sigma2_100**2/2
        C_bb = np.zeros(z_mat.shape)
        D = b_mat*( delta * eta * np.log(e) - tau*z_mat*e + xi_m*h2**2/2) - e*ell

        print('here')
        out = PDESolver_2d(stateSpace, A, B_z, B_b, C_zz, C_bb, D, v0,
                           epsilon, solverType = 'False Transient')

        out_comp = out[2].reshape(v0.shape,order = "F")

        PDE_rhs = A*v0 + B_z*v0_dz + B_b*v0_db + C_zz*v0_dzz + C_bb*v0_dbb + D

        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)))
        #     if episode % 1 == 0:
        print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]))
        episode += 1

        v0 = out_comp

    print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]) )
    print("--- %s seconds ---" % (time.time() - start_time))

    solution = dict(e = e, psi = v0)
    return solution


def one_ell(ell, parameters):
    [z_mat, b_mat], [hz, hb], stateSpace, params = parameters
    solution = falseTransient(ell, .5, 1e-8, stateVariable = [z_mat, b_mat], stateSteps = [hz, hb], stateSpace = stateSpace,  params = params)
    return solution


solution_20_200_200_v2 = dict()
# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-8
epsilon = .5
params = (delta, eta, mu2, tau, rho, sigma2_100)
numell = 400
log_ell = np.linspace(-13, -5 , num=numell)
h_ell = 1e-7

ells = np.zeros(numell*2)
for i, ell_i in enumerate(np.exp(log_ell)):
    ells[2*i] = ell_i
    ells[2*i + 1] = ell_i + h_ell
# for ell in np.exp(log_ell):
    # solution = falseTransient(ell, .5, 1e-8, stateVariable = [z_mat, b_mat], stateSteps = [hz, hb], stateSpace = stateSpace,  params = params, logFileName = logFileName)
    # solution_next = falseTransient( ell + h_ell,  .5,  1e-8, stateVariable = [z_mat, b_mat], stateSteps =  [hz, hb], stateSpace = stateSpace, params=params, logFileName = logFileName)
    # solution_20_200_200_v2[ell] = solution
    # solution_20_200_200_v2[ell+h_ell] = solution_next

start = time.time()
parameters = ([z_mat, b_mat], [hz, hb], stateSpace, params)
if __name__ == "__main__":
    solution = Parallel(n_jobs=16)(delayed(one_ell)(ell, parameters) for ell in ells)
end = time.time()
print("time:{:.5f}s".format(end - start))

for ell, solu in zip(ells, solution):
    solution_20_200_200_v2[ell] = solu
# restore results
dataFile = '../data/solution/solu_modified_50*200*500_40_0204'
with open(dataFile, 'wb') as handle:
    pickle.dump(solution_20_200_200_v2, handle, protocol=pickle.HIGHEST_PROTOCOL)
