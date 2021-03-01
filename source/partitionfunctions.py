#!/bin/env python
# packages
import numpy as np
from utilities import dLambda
####################################################3
def par_index_1type(source, params_dim):
    """
    return a 1-dimension array for each row: the other parameters that in the same partition.
    ___
    source: 'damage'(default), 'temp' or 'carbon';
    params_dim: (n_t, n_c, n_d), number of temperature, carbon, and damage mdoels.
    """

    n_t, n_c, n_d = params_dim
    num = n_t*n_c*n_d
    if source == "damage":
        d_idx = np.zeros((n_d, n_t*n_c), dtype = int)
        for i in range(n_d):
            d_idx[i] = np.arange(i, num+i, n_d)

        return d_idx

    elif source == "carbon":
        c_idx = np.zeros((n_c, n_t*n_d), dtype = int)
        for i in range(n_c):
            idx = []
            for j in range(n_d):
                idx_d = np.arange(n_d*i + j, num+n_d*i+j, n_c*n_d)
                idx = np.append(idx, idx_d)
            idx = np.sort(idx, axis = None)
            c_idx[i] = np.array(idx)

        return c_idx

    elif source == "temp":
        t_idx = np.zeros((n_t, n_c*n_d), dtype=int)
        for i in range(n_t):
            t_idx[i] = np.arange(n_c*n_d*i, n_c*n_d*i + n_c*n_d, 1)

        return t_idx

    else:
        raise Exception("Wrong one type partition. Check for your input. You entered {}".format(source))


def par_index_2type(source, params_dim):
    """
    return a 1-dimension array for each row: the other parameters that in the same partition.
    ___
    source: 'temp and carbon'(default), 'temp and damage' or 'carbon and damage';
    params_dim: (n_t, n_c, n_d), number of temperature, carbon, and damage mdoels.
    """

    n_t, n_c, n_d = params_dim
    num = n_t*n_c*n_d
    if source == "temp and carbon":
        n_b = n_t*n_c
        tc_idx = np.zeros((n_b, n_d), dtype = int)
        for i in range(n_b):
            tc_idx[i] = np.arange(n_d*i, n_d*i + n_d, 1)

        return tc_idx

    elif source == "carbon and damage":
        cd_idx = np.zeros((n_c*n_d, n_t), dtype = int)
        count = 0
        for i in range(n_c):
            for j in range(n_d):
                cd_idx[count] = np.arange(n_d*i+j, num+n_d*i+j, n_c*n_d)
                count += 1

        return cd_idx

    elif source == "temp and damage":
        td_idx = np.zeros((n_t*n_d, n_c), dtype=int)
        count =0
        for i in range(n_t):
            for j in range(n_d):
                td_idx[count] = np.arange(n_c*n_d*i + j, n_c*n_d*i + j + n_c*n_d, n_d)
                count += 1

        return td_idx

    else:
        raise Exception("Wrong two type partition. Check for your input. You entered {}".format(source))

def partition_func(
        args,
        params_dist,
        params_dim,
        source = "damage",
        ):

    """
    return U_y (used to solve ODE), IRE (relative entropy), scc1 (first part of the decomposition), firstpart (of the objective function).

    Parameters
    --------------------------------------------
    args: tuple (E, F, xi_a, xi_d);
    params_dist: 3 column array documenting parameter values of their corresponding probability;
    params_dim: (n_t, n_c, n_d), tuple documenting the models in the parameters. n_t, number of temperature models. n_c, number of carbon models. n_d, number of damage functions;
    source: string. "damage"(default), "carbon", "temp", "temp and carbon", "temp and damage" or "carbon and damage". It indicates the kind of partition that you are interested in.
    construction: "plus"(default) or "minus". The decomposition construction (refer to shortpaper).
    """

    ems, y_mat, z_mat, xi_a, v_n = args
    numz, numy = z_mat.shape
    n_t, n_c, n_d = params_dim
    num = len(params_dist) # number of total parameters
    n_b = n_t*n_c # number of mu2s
    params = params_dist[:,[0,1]]
    prior = params_dist[:, 2]
    mu2_list = params[np.arange(0, num, n_d),0]
    gamma2p_list = params[:n_d,1]
    global U_y, IRE, scc1


    if source == "damage":
        # construct pi_l_0_bar

        d_idx = par_index_1type("damage", params_dim)

        pi_l_0 = np.zeros((n_d, ems.shape[0]))
        for i in range(n_d):
            idx = d_idx[i]
            pi_l_0_i = prior[idx]
            pi_l_0[i] = np.sum(pi_l_0_i, axis=0)

        mu_n = np.zeros((n_d, ems.shape[0]))
        for i in range(n_d):
            dmg_id = d_idx[i]
            mu_n_i = np.zeros((len(dmg_id), numz, numy))
            for j , beta_f, gamma_2_plus, p in enumerate(params_dist[dmg_id]):
                dlambda = dLambda(y_mat, 1, gamma_1, gamma_2, gamma_2_plus, gamma_bar)
                mu_n_i[j] = p*dlambda*z_mat*ems

            mu_n[i] = np.array(mu_n_j)/pi_l_0[i]

        pi_l = np.zeros((n_d, ems.shape[0]))
        for i in range(n_d):
            weight = -1/xi_a*xi_d*mu_n[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        pi_l = pi_l/np.sum(pi_l, axis = 0)

        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior

        pi_l_minus = np.zeros((num, ems.shape[0]))
        for i in range(n_d):
            idx = np.arange(i, num+i, n_d)
            pi_l_minus[idx] = prior_d[idx]*pi_l[i]/pi_l_0[i]


    elif source == "carbon":
        prior = params_dist[:,2]
        c_idx = par_index_1type("carbon", params_dim)

        pi_l_0 = np.zeros((n_c, ems.shape[0]))
        for i in range(n_c):
            idx = c_idx[i]
            pi_l_0_i = prior[idx]
            pi_l_0[i] = np.sum(pi_l_0_i, axis=0)


        carb = np.zeros((n_c, ems.shape[0]))
        for i in range(len(carb)):
            mu_n_j = 0
            carb_bp = params_dist[c_idx[i]]
            for beta_f, gamma_2_plus, p in carb_bp:
                mu_n_j += p*( gamma_1 + gamma_2*beta_f*y_mat
                             + gamma_2_plus*(beta_f*y_mat - gamma_bar)*((beta_f*y_mat - gamma_bar) > 0))*beta_f*ems

            carb[i] = np.array(mu_n_j)/pi_l_0[i]

        pi_l = np.zeros((len(carb), ems.shape[0]))
        for i in range(len(pi_l)):
            weight = -1/xi_a*xi_d*carb[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        pi_l = pi_l/np.sum(pi_l, axis = 0)

        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior

        pi_l_minus = np.zeros((num, ems.shape[0]))
        for i in range(n_c):
            idx = c_idx[i]
            pi_l_minus[idx] = prior_d[idx]*pi_l[i]/pi_l_0[i]



    elif source == "temp":
        t_idx = par_index_1type("temp", params_dim)

        pi_l_0 = np.zeros((n_t, ems.shape[0]))
        for i in range(n_t):
            idx = t_idx[i]
            pi_l_0_i = prior[idx]
            pi_l_0[i] = np.sum(pi_l_0_i, axis=0)

        temp = np.zeros((n_t, ems.shape[0]))
        for i in range(n_t):
            mu_n_j = 0
            temp_bp = params_dist[t_idx[i]]
            for beta_f, gamma_2_plus, p in temp_bp:
                mu_n_j += p*( gamma_1 + gamma_2*beta_f*y_mat
                             + gamma_2_plus*(beta_f*y_mat - gamma_bar)*((beta_f*y_mat - gamma_bar) > 0))*beta_f*ems

            temp[i] = np.array(mu_n_j)/pi_l_0[i]

        pi_l = np.zeros((n_t, ems.shape[0]))
        for i in range(n_t):
            weight = -1/xi_a*xi_d*temp[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        pi_l = pi_l/np.sum(pi_l, axis = 0)

        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior

        pi_l_minus = np.zeros((num, ems.shape[0]))
        for i in range(n_t):
            idx = np.arange(n_c*n_d*i,  n_c*n_d+n_c*n_d*i, 1)
            for j in idx:
                j = int(j)
                pi_l_minus[j] = (prior_d[j]/pi_l_0[i])*pi_l[i]


    elif source == "temp and carbon":
        tc_idx = par_index_2type("temp and carbon", params_dim)
        pi_l_0 = np.zeros(n_b)
        for i in range(n_b):
            idx = tc_idx[i]
            pi_l_0[i] = np.sum(prior[idx], axis=0)

        temp_carb = np.zeros((n_t*n_c, ems.shape[0]))
        for i in range(n_b):
            idx = tc_idx[i]
            params_dist_i = params_dist[idx]
            mu_n_j = 0
            for beta_dist in params_dist_i:
                beta_f, gamma_2_plus, p = beta_dist
                mu_n_j += p*( gamma_1 + gamma_2*beta_f*y_mat
                             + gamma_2_plus*(beta_f*y_mat - gamma_bar)*((beta_f*y_mat - gamma_bar) > 0) )*beta_f*ems

            temp_carb[i] = np.array(mu_n_j)/pi_l_0[i]


        # pi_l_bar construction
        pi_l = np.zeros((n_b, ems.shape[0]))
        for i in range(n_b):
            weight = -1/xi_a*xi_d*temp_carb[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        pi_l = pi_l/np.sum(pi_l, axis=0)

        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior

        pi_l_minus = np.zeros((num, ems.shape[0]))
        for i in range(n_b):
            idx = np.arange(n_d*i, n_d*i+n_d, 1)
            pi_l_minus[idx] = prior_d[idx]*pi_l[i]/pi_l_0[i]


    elif source == "carbon and damage":
        prior = params_dist[:, 2]

        cd_idx = par_index_2type(source, params_dim)
        pi_l_0 = np.zeros(n_c*n_d)
        for i in range(n_c*n_d):
            idx = cd_idx[i]
            pi_l_0_i = prior[idx]
            pi_l_0[i] = np.sum(pi_l_0_i, axis=0)

        carb_damage = np.zeros((n_c*n_d, ems.shape[0]))
        for i in range(n_c*n_d):
            idx = cd_idx[i]
            beta_temp = params_dist[idx]
            mu_n_j = 0
            for temp_dist in beta_temp:
                beta_f, gamma_2_plus, p = temp_dist
                mu_n_j += p*( gamma_1 + gamma_2*beta_f*y_mat
                        + gamma_2_plus*(beta_f*y_mat - gamma_bar)*(beta_f*y_mat > gamma_bar))*beta_f*ems

            carb_damage[i] = np.array(mu_n_j)/pi_l_0[i]

        pi_l = np.zeros((len(carb_damage), ems.shape[0]))
        for i in range(len(pi_l)):
            weight = -1/xi_a*xi_d*carb_damage[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        pi_l = pi_l/np.sum(pi_l, axis = 0)

        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior


        pi_l_minus = np.zeros((num, ems.shape[0]))
        count = 0
        for i in range(n_c):
            for j in range(n_d):
                idx = np.arange(n_d*i+j, num+n_d*i+j, n_c*n_d)
                pi_l_minus[idx] = prior_d[idx]*pi_l[count]/pi_l_0[count]
                count += 1



    elif source == "temp and damage":
        prior = params_dist[:,2]

        td_idx = par_index_2type("temp and damage", params_dim)
        pi_l_0 = np.zeros((n_t*n_d, ems.shape[0]))
        for i in range(n_t*n_d):
            idx = td_idx[i]
            pi_l_0_i = prior[idx]
            pi_l_0[i] = np.sum(pi_l_0_i, axis=0)

        temp_damage = np.zeros((n_t*n_d, ems.shape[0]))
        for i in range(n_t*n_d):
            idx = td_idx[i]
            beta_carb = params_dist[idx]
            mu_n_j = 0
            for beta_f, gamma_2_plus, p in beta_carb:
                mu_n_j += p*( gamma_1 + gamma_2*beta_f*y_mat
                        + gamma_2_plus*(beta_f*y_mat - gamma_bar)*((beta_f*y_mat - gamma_bar) > 0))*beta_f*ems

            temp_damage[i] = np.array(mu_n_j)/pi_l_0[i]

        pi_l = np.zeros((len(temp_damage), ems.shape[0]))
        for i in range(len(pi_l)):
            weight = -1/xi_a*xi_d*temp_damage[i]
            pi_l[i] = pi_l_0[i]*np.exp(weight)

        denominator = 0
        for i in range(len(pi_l)):
            denominator += pi_l[i]

        pi_l = pi_l/denominator


        prior_d = np.ones((num, ems.shape[0]))
        for i in range(ems.shape[0]):
            prior_d[:, i] = prior


        pi_l_minus = np.zeros((num, ems.shape[0]))
        count = 0
        for i in range(n_t):
            for j in range(n_d):
                idx = np.arange(n_c*n_d*i+j, n_c*n_d+n_c*n_d*i+j, n_d)
                pi_l_minus[idx] = prior_d[idx]*pi_l[count]/pi_l_0[count]
                count += 1


    else:
        raise Exception("Wrong partition. You entered {}".format(partition))

    U_y = np.zeros(ems.shape)
    IRE = np.zeros(ems.shape)
    scc1 = np.zeros(ems.shape)

    for i in range(num):
        beta_f, gamma_2_plus = params[i]
        U_y += pi_l_minus[i]*((gamma_2*beta_f + gamma_2_plus*beta_f * ((beta_f*y_mat - gamma_bar) > 0))* beta_f * ems)
        IRE += (np.log(pi_l_minus[i]) - np.log(prior_d[i])) * pi_l_minus[i]
        scc1 += pi_l_minus[i]*xi_d*(gamma_1 + gamma_2*beta_f*y_mat + gamma_2_plus * (y_mat*beta_f - gamma_bar) * ((y_mat*beta_f - gamma_bar) > 0))*beta_f


    firstpart = scc1*ems
    return U_y, IRE, scc1, firstpart
