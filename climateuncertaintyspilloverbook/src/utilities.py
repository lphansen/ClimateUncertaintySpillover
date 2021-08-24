# -*- coding: utf-8 -*-
"""
Functions that facilitate computation.

"""
import numpy as np
from numba import njit
import numba
from multiprocessing import Pool

@njit
def find_nearest_value(array, value):
    r"""Return the index of the element in ``array`` that is closest to ``value``.
    """
    loc = np.abs(array - value).argmin()
    return loc


@njit
def compute_derivatives(data, order, dlt):
    r"""Compute the derivatives of a function.

    Parameters
    ----------
    data : (N, ) ndarrays
        The function whose derivatives to be computed.
    order : int::
        1: first order derivative,

        2: second order derivative.
    dlt : float
        Grid step.

    Returns
    -------
    res : (N, ) ndarrays
        Deriative array.

    Notes
    -----
    First order derivatives are computed one-sidedly.
    Specifically,

    .. math::
        \frac{d f(x_0)}{dx} \approx & \frac{f(x_1) - f(x_0)}{\Delta x} \\
        \frac{d f(x_i)}{dx} \approx & \frac{f(x_i) - f(x_{i-1})}{\Delta x}, i = 1,2,\dots,N
    """
    res = np.zeros_like(data)
    if order == 1:
        res[1:] = (1 / dlt) * (data[1:] - data[:-1])
        res[0] = (1 / dlt) * (data[1] - data[0])
    elif order == 2:
        res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
        res[-1] = (1 / dlt ** 2) * (data[-1] + data[-3] - 2 * data[-2])
        res[0] = (1 / dlt ** 2) * (data[2] + data[0] - 2 * data[1])
    return res


def ode_y_parallel(func, args_list):
    with Pool() as p:
        res_list = p.starmap(func, args_list)
    return res_list


def solve_post_jump(y_grid, γ_3, func, args_list):
    res_list = ode_y_parallel(func, args_list)
    ϕ_list = np.zeros((len(γ_3), len(y_grid)))
    ems_list = np.zeros((len(γ_3), len(y_grid)))
    for j in range(len(γ_3)):
        ϕ_list[j] = res_list[j]['v']
        ems_list[j] = res_list[j]['e_tilde']
    return ϕ_list, ems_list

def dLambda(y_mat, z_mat, gamma1, gamma2, gamma2p, gammaBar):
    """compute first derivative of Lambda, aka log damage function
    :returns:
    dlambda: (numz, numy) ndarray
        first derivative of Lambda

    """
    dlambda = gamma1 + gamma2*y_mat*z_mat + gamma2p*(y_mat*z_mat - gammaBar)*(y_mat*z_mat>=gammaBar)
    return dlambda

def J(y_arr, y_underline=1.5):
    r1 = 1.5
    r2 = 2.5
    return r1*(np.exp(r2/2*(y_arr - y_underline)**2) - 1) * (y_arr >= y_underline)
