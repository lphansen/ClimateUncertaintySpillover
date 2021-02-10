"""
module for solve PDE, adding state variable version
"""
import sys, os
sys.path.append(os.getcwd()+ '/source')
import numpy as np
# import numba
import constants
import pickle
# Start of the program
# global constants
delta = constants.DELTA
eta = constants.ETA
median = constants.MEDIAN
gamma_base = constants.GAMMA_BASE
xi_m = constants.XI_m
tau = median*gamma_base
# compute
# @numba.jit(nopython=True)
def compute_psi(ell, z_2,  b_size=10, z_size = 20, args = (delta, eta, tau)):
    """
    compute psi(b, ell, z2)
    """
    b_step = 1./b_size
    b_grid = np.linspace(b_step, 1., num = b_size)
    # z_grid = np.linspace(start = 1e-5, end = 2., num = z_size)
    # for e computation
    A = np.zeros((b_size, b_size))
    B = np.zeros(b_size)
    for i in range(b_size):
        b = b_grid[i]
        e_star = b*delta*eta/(b*tau*z_2 + ell)
        B[i] = b*(delta*eta*np.log(e_star) - tau*z_2*e_star) - ell*e_star
        if i == 0:
            A[i, i+1] = delta*b/b_step
        elif i  == b_size - 1:
            A[i, i-1] = - delta*b/b_step
            A[i, i] = delta*b/b_step
        else:
            A[i, i-1] = - delta*b/(2*b_step)
            A[i, i+1] = delta*b/(2*b_step)
    # solve for the linear system
    psi_grid = np.linalg.solve(A, B)
    return psi_grid

# @numba.jit
def compute_dpsi(ell, z_2, d_ell = 1e-9, b_size=10, z_size=20, args = (delta, eta, tau)):
    """
    Compute derivative of psi
    """
    psi_0 = compute_psi(ell, z_2, b_size, z_size, args)[-1]
    psi_1 = compute_psi(ell+d_ell, z_2, b_size, z_size, args)[-1]
    dpsi_dell = (psi_1 - psi_0)/d_ell
    r = - dpsi_dell
    phi = psi_0 + ell*r
    return r, phi

# @numba.jit
def solve_phi(log_ell_min = -20, log_ell_max = 1, grid_size = 5,
              b_size=10, z_size=20, d_ell = 1e-9, z_2 = 2.,
              args = (delta, eta, tau) ):
    """
    solve phi grid and r grid, z grid
    """
    log_ell = np.linspace(log_ell_min, log_ell_max, grid_size)
    ell_grid = np.exp(log_ell)
    r_grid = np.zeros_like(ell_grid)
    z_grid = np.zeros_like(ell_grid)
    phi_grid = np.zeros_like(ell_grid)
    for i in range(grid_size):
        ell = ell_grid[i]
        r_grid[i], phi_grid[i] = compute_dpsi(ell, z_2, d_ell, b_size, z_size, args)
    index_sorted = np.argsort(r_grid)
    r_grid_sorted = r_grid[index_sorted]
    phi_grid_sorted = phi_grid[index_sorted]
    return r_grid_sorted, phi_grid_sorted

grid_size = 1000
numz = 20
z_grid = np.linspace(1e-5, 10, num = numz)
solution = np.zeros((numz, grid_size, 2))
print("solving.....")
for i in range(numz):
    z = z_grid[i]
    r_grid, phi_grid = solve_phi(-30, 10, grid_size,
                                 b_size = 100, z_size = 20, z_2 = z )
    solution[i] = np.hstack([r_grid.reshape(-1,1,order="F"), phi_grid.reshape(-1,1,order="F")])
solu_dir = '../data/solu_0128_z1-10_1749'
with open(solu_dir, 'wb') as handle:
    pickle.dump(solution, handle, protocol = pickle.HIGHEST_PROTOCOL)

