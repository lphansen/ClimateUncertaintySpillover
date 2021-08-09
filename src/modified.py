"""
module for modified model
"""
import time
import numpy as np
import SolveLinSys
from supportfunctions import *
######################global variable
delta = .01
eta = .032
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
xi_m = .00256

# state variable
numz = 40
z2_min = 1e-5
z2_max = 2
hz = (z2_max - z2_min)/numz
z = np.linspace(z2_min, z2_max, num=numz)

numr = 200
r_min = 0
r_max = 3000
hr = (r_max - r_min)/numr
r = np.linspace(r_min, r_max, num=numr)

(z_mat, r_mat) = np.meshgrid(z, r, indexing = 'ij')
stateSpace = np.hstack([z_mat.reshape(-1,1, order='F'), r_mat.reshape(-1,1,order='F')])

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

logFile = open('40*200.log','a')
# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-10
epsilon = .5
while episode == 0 or FC_Err > tol:
    print('Episode:{:d}'.format(episode), file = logFile)
    if episode ==0:
        v0 = delta*eta*r_mat
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
    v0_dr = finiteDiff(v0,1,1,hr)
    v0_drr = finiteDiff(v0,1,2,hr)
    print(v0_dr, file = logFile)
    # updating controls
    Converged = 0
    nums = 0
    e =  delta * eta / ( tau*z_mat + v0_dr)
    e[e<0] = 1e-300
    h2 = - v0_dz*np.sqrt(z_mat)*sigma2_100/xi_m
    print(np.min(e), file = logFile)
    # HJB coefficient
    A = -delta * np.ones(z_mat.shape)
    B_z = - rho*(z_mat - mu2)  + np.sqrt(z_mat)*sigma2_100*h2
    B_r = - e
    C_zz = z_mat*sigma2_100**2/2
    C_rr = np.zeros(z_mat.shape)
    D = delta * eta * np.log(e) - tau*z_mat*e + xi_m*h2**2/2

    print('here')
    out = PDESolver_2d(stateSpace, A, B_z, B_r, C_zz, C_rr, D, v0,
                       epsilon, solverType = 'False Transient')

    out_comp = out[2].reshape(v0.shape,order = "F")

    PDE_rhs = A*v0 + B_z*v0_dz + B_r*v0_dr + C_zz*v0_dzz + C_rr*v0_drr + D

    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - v0)))
    #     if episode % 1 == 0:
    print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode,
          PDE_Err, FC_Err, out[0], out[1]), file = logFile)
    episode += 1

    v0 = out_comp

print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]), file = logFile)
print("--- %s seconds ---" % (time.time() - start_time), file=logFile)

solu_40_200 = dict(e = e, phi = v0)
# restore results
import pickle
dataFile = '../data/solution/solu_modified_40*200_0900'
with open(dataFile, 'wb') as handle:
    pickle.dump(solu_40_200, handle, protocol=pickle.HIGHEST_PROTOCOL)
