# -*- coding: utf-8 -*-
"""
Utility functions to facilitate computation.

"""
import numpy as np
from numba import njit


# @njit
def find_nearest_value(array, value):
    """
    Find nearest value for

    """
    loc = np.abs(array - value).argmin()
    return loc

def compute_h_hat(emission, γ, ξ, arg = (1.75/1000, 1.2)):
    """
    compute h hat

    Parameters
    ----------
    emission: array
        simulated emission sequence
    γ: float
        damage model parameter
    ξ: float
        model misspecification parameter;
        smaller the value, greater the concern for model misspecification

    Returns
    -------
    h_hat, or drift distortion
    """
    median, σ_n = arg
    gamma = γ
    xi = ξ
    h_hat = emission*median*gamma*σ_n/xi
    h_hat = h_hat*median*1000*σ_n
    return h_hat

def compute_std(emission, time, arg = (1.75/1000, 1.2)):
    """
    compute standard deviation in table 1

    Parameters
    ----------
    emission: array
        simulated emission path
    time: int
        time span during which the standard deviation is considered

    Returns
    -------
    implied standard deviation
    """
    median, σ_n = arg
    emission_selected = emission[:time]
    std = np.sqrt(np.sum(emission_selected**2))/emission_selected.sum()*σ_n*median*1000
    return std

def dLambda(y_mat, z_mat, gamma1, gamma2, gamma2p, gammaBar):
    """compute first derivative of Lambda, aka log damage function
    :returns:
    dlambda: (numz, numy) ndarray
        first derivative of Lambda

    """
    dlambda = gamma1 + gamma2*y_mat*z_mat + gamma2p*(y_mat*z_mat - gammaBar)*(y_mat*z_mat>=gammaBar)
    return dlambda

def ddLambda(y_mat, z_mat, gamma2, gamma2p, gammaBar):
    """compute second derivative of Lambda function

    :gamma2: TODO
    :gamma2p: TODO
    :gammaBar: TODO
    :returns: TODO
    ddlambda: (numz, numy) ndarray
        second derivative

    """
    ddlambda = gamma2 + gamma2p*(y_mat*z_mat>=gammaBar)
    return ddlambda

# @njit
def weightOfPi(y_mat, z_mat, e, prior, gamma1, gamma2, gamma2p, gammaBar, xi_a, eta, rho, mu_2, sigma2, h2):
    """compute weight on posterior

    :y_mat: TODO
    :z_mat: TODO
    :PILast: TODO
    :gamma1: TODO
    :gamma2: TODO
    :gamma2p: TODO
    :gammaBar: TODO
    :returns: TODO

    """
    numDmg, numz, numy = prior.shape
    PIThis = np.zeros(prior.shape)
    weight = np.zeros(prior.shape)
    for i in range(numDmg):
        weight[i] = - (eta-1)/xi_a*gamma2p[i]*(y_mat>=gammaBar)*(y_mat-gammaBar)*z_mat*e
    weight = weight - np.max(weight, axis=0)
    weight_of_pi = prior*np.exp(weight)

    PIThis = weight_of_pi/np.sum(weight_of_pi, axis=0)
    return PIThis

@njit
def relativeEntropy(PIThis, PILast):
    """compute relative entropy

    :PIThis: TODO
    :PILast: TODO
    :xi_a: TODO
    :returns: TODO

    """
    numDmg, _, _ = PIThis.shape
    entrpy = np.zeros(PIThis.shape)
    for i in range(numDmg):
        entrpy[i] = PIThis[i]*(np.log(PIThis[i]) - np.log(PILast[i]))
    return np.sum(entrpy, axis=0)

# weight of pi when climate models considered
def weightPI(y_mat, z_mat, e, prior, modelParam, v0_dz, rho, gammaBar, v_n, xi_a, sigma2):
    """compute pi j star with mu_2 ambiguity
    :returns: TODO

    """
    numModel, _ = modelParam.shape
    numz, numy = y_mat.shape
    weight = np.zeros((numModel, numz, numy))
    for i in range(numModel):
        mu2, gamma2p = modelParam[i, :]
        temp = (y_mat*z_mat - gammaBar)*(z_mat*e-y_mat*(z_mat-mu2))
        temp += 1/2*z_mat*y_mat**2*sigma2**2
        temp *= v_n*gamma2p*(y_mat*z_mat>=gammaBar)
        temp += v0_dz*rho*mu2
        temp *= -1/xi_a
        weight[i] = temp
    weight = weight - np.max(weight, axis=0)
    weight = prior*np.exp(weight)
    PIThis = weight/np.sum(weight, axis=0)
    return PIThis

def weightPITemp(y_mat, z_mat, e, prior, modelParam, v0_dz, rho, gammaBar, v_n, xi_a, sigma2):
    """compute pi j star with mu_2 ambiguity
    :returns: TODO

    """
    numModel, _ = modelParam.shape
    numz, numy = y_mat.shape
    weight = np.zeros((numModel, numz, numy))
    for i in range(numModel):
        mu2, gamma2p = modelParam[i, :]
        temp = (y_mat - gammaBar)*z_mat*e
        temp *= v_n*gamma2p*(y_mat>=gammaBar)
        temp += v0_dz*rho*mu2
        temp *= -1/xi_a
        weight[i] = temp
    weight = weight - np.max(weight, axis=0)
    weight = prior*np.exp(weight)
    PIThis = weight/np.sum(weight, axis=0)
    return PIThis

@njit
def damageDriftSingle(y_mat, z_mat, e, mu2, gamma2p, rho, gamma1, gamma2, gammaBar, sigma2):
    """TODO: Docstring for damageDrift.

    :y_mat: TODO
    :z_mat: TODO
    :e: TODO
    :modelParam: TODO
    :returns: TODO

    """
    temp = z_mat*e
    temp *= gamma1 + gamma2*y_mat + gamma2p*(y_mat-gammaBar)*(y_mat>=gammaBar)
    return temp

@njit
def damageDrift(y_mat, z_mat, e, modelParam, gamma1, gamma2, gammaBar, rho, sigma2):
    """compute damage drift weighted by posteriors

    :y_mat: TODO
    :z_mat: TODO
    :e: TODO
    :modelParam: TODO
    :gamma1: TODO
    :gamma2: TODO
    :gammaBar: TODO
    :rho: TODO
    :returns: TODO

    """
    numModel,_ = modelParam.shape
    numz, numy = z_mat.shape
    driftMat = np.zeros((numModel, numz, numy))
    for i in range(numModel):
        mu2, gamma2p = modelParam[i,:]
        driftMat[i] = damageDriftSingle(y_mat, z_mat, e, mu2, gamma2p, rho, gamma1, gamma2, gammaBar, sigma2)

    return driftMat

@njit
def zDrift(z_mat, modelParam, rho):
    """computed drift term for v_dz

    :y_mat: TODO
    :z_mat: TODO
    :modelParam: TODO
    :v0_dz: TODO
    :rho: TODO
    :returns: TODO

    """
    numModel, _ = modelParam.shape
    numz, numy = z_mat.shape
    zDriftMat = np.zeros((numModel, numz, numy))
    for i in range(numModel):
        mu2 = modelParam[i,0]
        zDriftMat[i] = -rho*(z_mat - mu2)
    return zDriftMat
