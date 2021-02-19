"""
module for modified model
"""
import time
import numpy as np
import SolveLinSys
from supportfunctions import *
from global_parameters import *
######################global variable
delta = .01
eta = .032
mu2 = 1.86/1000
sigma_z_50 = .30
sigma_z_100 = .21/1000
sigma_z_200 = .16
gamma_base = .018
gamma_low = .012
gamma_high = .024
rho = .9

def compute_sigma2(rho, sigma_z, mu_2):
    """
    compute_sigma2
    """
    return np.sqrt(2*sigma_z**2*rho/mu_2)


sigma2_100 = compute_sigma2(rho, sigma_z_100, mu2)
tau = .00175*gamma_base
xi_m = .00256

# state variable
numz = N_Z
z2_min = Z_MIN
z2_max = Z_MAX
hz = (z2_max - z2_min)/numz
z = np.linspace(z2_min, z2_max, num=numz)

numb = 200
b_min = 1e-6
b_max = 1
hb = (b_max - b_min)/numb
b = np.linspace(b_min, b_max, num=numb)


gamma_1 = 0.00017675
gamma_2 = 2*.0022
gamma_bar = 2
gamma_2_plus = 0
numy = N_Y
y_min = Y_MIN
y_max = Y_MAX
y = Y_GRID
hy = HY


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

logFile = open('20*100_v6.log','a')
solution_20_100_v2 = dict()
# solving the PDE
start_time = time.time()
episode = 0
tol = 1e-8
epsilon = .3

ells = np.linspace(1e-5, 100, num=5)
# ells = ELLS
ells = [0]
for i in range(1):
    ell = ells[i]
    while episode == 0 or FC_Err > tol:
        print('Episode:{:d}'.format(episode), file = logFile)
        if episode ==0:
            v0 = - delta*eta*y_mat
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
        # With only v0_dFF[v0_dFF < 1e-16] = 0
        v0_dy = finiteDiff(v0,1,1,hy)
        v0_dyy = finiteDiff(v0,1,2,hy)
        print(v0_dy)
        # updating controls
        Converged = 0
        nums = 0
        e =  - delta*eta/(v0_dy+(eta-1)*(gamma_1 + gamma_2*y_mat*z_mat + gamma_2_plus*(y_mat*z_mat -gamma_bar)*(y_mat*z_mat>gamma_2))*z_mat)
        e[e<0] = 1e-300
        h2 = - (v0_dz*np.sqrt(z_mat)*sigma2_100+ (eta -1 )*(gamma_1 + gamma_2*y_mat*z_mat + gamma_2_plus*(y_mat*z_mat -gamma_bar)*(y_mat*z_mat>gamma_2))*y_mat*np.sqrt(z_mat)*sigma2_100)/xi_m
        print(np.min(e))
        # HJB coefficient
        A =  - delta*np.ones(y_mat.shape)
        B_z = - rho*(z_mat - mu2) + np.sqrt(z_mat)*sigma2_100*h2
        B_y = e
        C_zz = z_mat*sigma2_100**2/2
        C_yy = np.zeros(z_mat.shape)
        D =  delta*eta*np.log(e) - (1 - eta)*((gamma_1 + gamma_2*y_mat*z_mat +  gamma_2_plus*(y_mat*z_mat - gamma_bar)*(y_mat*z_mat > gamma_bar))*(z_mat*e + y_mat*(-rho*(z_mat - mu2)) + y_mat*np.sqrt(z_mat)*sigma2_100*h2) +(gamma_2 + gamma_2_plus*(y_mat*z_mat>gamma_bar))*z_mat*y_mat**2*sigma2_100**2) + xi_m*h2**2/2

        print('here')
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

    print("Episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}; Iterations: {:d}; CG Error: {:.10f}".format(episode, PDE_Err, FC_Err, out[0], out[1]), file = logFile)
    print("--- %s seconds ---" % (time.time() - start_time))

    solution_20_100_v2[ell] = dict(e = e, psi = v0)

# restore results
import pickle
from datetime import datetime
nowtime = datetime.now()
time_store = nowtime.strftime("%d%m-%H:%M")
dataFile = '../data/solution/solu_modified_{}*{}_{}'.format(numz, numy, time_store)
with open(dataFile, 'wb') as handle:
    pickle.dump(solution_20_100_v2, handle, protocol=pickle.HIGHEST_PROTOCOL)
