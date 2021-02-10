"""
module for global constants
"""
import numpy as np
##
DELTA = .01
ETA = .032
MEDIAN = .00175
GAMMA_BASE = .018
XI_M = .00256/10
# compute tau
TAU = MEDIAN*GAMMA_BASE
# setup for z_2 process
MU_2 = 1.86/1000
RHO = .9
SIGMA_Z = .42/1000

# grid setting
## z grid
N_Z = 10
Z_MIN = MU_2 - 4*SIGMA_Z
Z_MAX = MU_2 + 4*SIGMA_Z
Z_GRID = np.linspace(Z_MIN, Z_MAX, N_Z)
HZ = Z_GRID[1] - Z_GRID[0]
## b grid
N_B = 10
B_MIN = 1e-6
B_MAX = 1.
B_GRID = np.linspace(B_MIN, B_MAX, N_B)
HB = B_GRID[1] - B_GRID[0]
## y grid
N_Y = 20
Y_MIN = 100
Y_MAX = 1000
Y_GRID = np.linspace(Y_MIN, Y_MAX, N_Y)
HY = Y_GRID[1] - Y_GRID[0]
## ell grid
N_ELL = 50
LOG_ELL_MIN = -13
LOG_ELL_MAX = -5
LOG_ELL = np.linspace(LOG_ELL_MIN, LOG_ELL_MAX, N_ELL)
ELLS = np.zeros(N_ELL*2)
ELL_STEP = 1e-7
for i in range(N_ELL):
    ell = np.exp(LOG_ELL[i])
    ELLS[2*i] = ell
    ELLS[2*i + 1] = ell + ELL_STEP
