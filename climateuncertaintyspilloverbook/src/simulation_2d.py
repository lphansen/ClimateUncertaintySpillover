import numpy as np
from scipy.interpolate import interp2d


def damage_intensity(y, y_bar_lower):
    r1 = 1.5
    r2 = 2.5
    return r1 * (np.exp(r2/2 * (y - y_bar_lower)**2) - 1) * (y >= y_bar_lower)


class EvolutionState:
    DAMAGE_MODEL_NUM = 10
    DAMAGE_PROB = np.ones(10) / 10

    def __init__(self, t, prob, damage_jump_state, damage_jump_loc, tech_jump_state, variables,
                 tech_intensity_first, tech_intensity_second, y_bar_lower):
        self.t = t
        self.prob = prob
        self.damage_jump_state = damage_jump_state
        self.damage_jump_loc = damage_jump_loc
        self.tech_jump_state = tech_jump_state
        self.variables = variables
        self.y_bar_lower = y_bar_lower

        self.TECH_INTENSITY_FIRST = tech_intensity_first
        self.TECH_INTENSITY_SECOND = tech_intensity_second

    def copy(self):
        return EvolutionState(self.t,
                              self.prob,
                              self.damage_jump_state,
                              self.damage_jump_loc,
                              self.tech_jump_state,
                              self.variables,
                              self.TECH_INTENSITY_FIRST,
                              self.TECH_INTENSITY_SECOND,
                              self.y_bar_lower)

    def evolve(self, sim_args, fun_args):
        κ, μ_k, σ_k, θ_mean, α, theta = sim_args
        e_fun_pre_damage_pre_tech, e_fun_pre_damage_post_first_tech, e_fun_pre_damage_post_second_tech,\
            e_fun_post_damage_pre_tech, e_fun_post_damage_post_first_tech, e_fun_post_damage_post_second_tech,\
            i_fun_pre_damage_pre_tech, i_fun_pre_damage_post_first_tech, i_fun_pre_damage_post_second_tech,\
            i_fun_post_damage_pre_tech, i_fun_post_damage_post_first_tech, i_fun_post_damage_post_second_tech\
                = fun_args
        e, k, y, i = self.variables

        # Compute variables at t+1
        if self.damage_jump_state == 'pre':
            if self.tech_jump_state == 'pre':
                e_fun = e_fun_pre_damage_pre_tech
                i_fun = i_fun_pre_damage_pre_tech
            elif self.tech_jump_state == 'post_first':
                e_fun = e_fun_pre_damage_post_first_tech
                i_fun = i_fun_pre_damage_post_first_tech
            elif self.tech_jump_state == 'post_second':
                e_fun = e_fun_pre_damage_post_second_tech
                i_fun = i_fun_pre_damage_post_second_tech
            else:
                raise ValueError('Invalid tech jump state. Should be one of [pre, post_first, post_second]')
        elif self.damage_jump_state == 'post':
            if self.tech_jump_state == 'pre':
                e_fun = e_fun_post_damage_pre_tech[self.damage_jump_loc]
                i_fun = i_fun_post_damage_pre_tech[self.damage_jump_loc]              
            elif self.tech_jump_state == 'post_first':
                e_fun = e_fun_post_damage_post_first_tech[self.damage_jump_loc]
                i_fun = i_fun_post_damage_post_first_tech[self.damage_jump_loc]
            elif self.tech_jump_state == 'post_second':
                e_fun = e_fun_post_damage_post_second_tech[self.damage_jump_loc]
                i_fun = i_fun_post_damage_post_second_tech[self.damage_jump_loc]
            else:
                raise ValueError('Invalid tech jump state. Should be one of [pre, post_first, post_second]')
        else:
            raise ValueError('Invalid tech jump state. Should be one of [pre, post]')

        e_new = e_fun(k, y)[0]
        i_new = i_fun(k, y)[0]
        y_new = y + e_new * θ_mean
        k_new = k + i_new - κ / 2 * i_new**2 + μ_k - .5 * σ_k**2

        variables_new = (e_new, k_new, y_new, i_new)
        res_template = EvolutionState(self.t+1,
                                   self.prob,
                                   self.damage_jump_state,
                                   self.damage_jump_loc,
                                   self.tech_jump_state,
                                   variables_new,
                                   self.TECH_INTENSITY_FIRST,
                                   self.TECH_INTENSITY_SECOND,
                                   self.y_bar_lower)
        states_new = []

        # Update probabilities
        temp = damage_intensity(y_new, self.y_bar_lower)
        if temp > 1:
            temp = 1

        damage_jump_prob = temp * 1
        tech_jump_first_prob = self.TECH_INTENSITY_FIRST * 1
        tech_jump_second_prob = self.TECH_INTENSITY_SECOND * 1

        if self.damage_jump_state == 'pre' and damage_jump_prob != 0:
            if self.tech_jump_state == 'pre':
                # Damage jumps, tech jumps
                for i in range(self.DAMAGE_MODEL_NUM):
                    temp = res_template.copy()
                    temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob * tech_jump_first_prob
                    temp.damage_jump_state = 'post'
                    temp.damage_jump_loc = i
                    temp.tech_jump_state = 'post_first'
                    states_new.append(temp)

                # Damage jumps, tech does not jump
                for i in range(self.DAMAGE_MODEL_NUM):
                    temp = res_template.copy()
                    temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob * (1 - tech_jump_first_prob)
                    temp.damage_jump_state = 'post'
                    temp.damage_jump_loc = i
                    temp.tech_jump_state = 'pre'
                    states_new.append(temp)
                
                # Damage does not jump, tech jumps
                temp = res_template.copy()
                temp.prob *= (1 - damage_jump_prob) * tech_jump_first_prob
                temp.damage_jump_state = 'pre'
                temp.tech_jump_state = 'post_first'
                states_new.append(temp)
                
                # Damage does not jump, tech does not jump
                temp = res_template.copy()
                temp.prob *= (1 - damage_jump_prob) * (1 - tech_jump_first_prob)
                temp.damage_jump_state = 'pre'
                temp.tech_jump_state = 'pre'
                states_new.append(temp)

            elif self.tech_jump_state == 'post_first':
                # Damage jumps, tech jumps
                for i in range(self.DAMAGE_MODEL_NUM):
                    temp = res_template.copy()
                    temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob * tech_jump_second_prob
                    temp.damage_jump_state = 'post'
                    temp.damage_jump_loc = i
                    temp.tech_jump_state = 'post_second'
                    states_new.append(temp)

                # Damage jumps, tech does not jump
                for i in range(self.DAMAGE_MODEL_NUM):
                    temp = res_template.copy()
                    temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob * (1 - tech_jump_second_prob)
                    temp.damage_jump_state = 'post'
                    temp.damage_jump_loc = i
                    temp.tech_jump_state = 'post_first'
                    states_new.append(temp)

                # Damage does not jump, tech jumps
                temp = res_template.copy()
                temp.prob *= (1 - damage_jump_prob) * tech_jump_second_prob
                temp.damage_jump_state = 'pre'
                temp.tech_jump_state = 'post_second'
                states_new.append(temp)

                # Damage does not jump, tech does not jump
                temp = res_template.copy()
                temp.prob *= (1 - damage_jump_prob) * (1 - tech_jump_second_prob)
                temp.damage_jump_state = 'pre'
                temp.tech_jump_state = 'post_first'
                states_new.append(temp)                

            elif self.tech_jump_state == 'post_second':
                # Damage jumps
                for i in range(self.DAMAGE_MODEL_NUM):
                    temp = res_template.copy()
                    temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob
                    temp.damage_jump_state = 'post'
                    temp.damage_jump_loc = i
                    states_new.append(temp)

                # Damage does not jump
                temp = res_template.copy()
                temp.prob *= (1 - damage_jump_prob)
                temp.damage_jump_state = 'pre'
                states_new.append(temp)             

            else:
                raise ValueError('Invalid tech jump state. Should be one of [pre, post_first, post_second]')
        else:
            if self.tech_jump_state == 'pre':
                # Tech jumps
                temp = res_template.copy()
                temp.prob *= tech_jump_first_prob
                temp.tech_jump_state = 'post_first'
                states_new.append(temp)

                # Tech does not jump
                temp = res_template.copy()
                temp.prob *= (1 - tech_jump_first_prob)
                temp.tech_jump_state = 'pre'
                states_new.append(temp)

            elif self.tech_jump_state == 'post_first':
                # Tech jumps
                temp = res_template.copy()
                temp.prob *= tech_jump_second_prob
                temp.tech_jump_state = 'post_second'
                states_new.append(temp)

                # Tech does not jump
                temp = res_template.copy()
                temp.prob *= (1 - tech_jump_second_prob)
                temp.tech_jump_state = 'post_first'
                states_new.append(temp)

            else:
                temp = res_template.copy()
                states_new.append(temp)

        return states_new



def simulation_dice_prob(sim_args, k_grid, y_grid, e_mat, i_mat, g_mat, h_mat, πc_mat, K0=80/0.115, y0=1.1, T=100):
    κ, μ_k, σ_k, θ_mean = sim_args
    e_fun = interp2d(k_grid, y_grid, e_mat.T)
    i_fun = interp2d(k_grid, y_grid, i_mat.T)
    h_fun = interp2d(k_grid, y_grid, h_mat.T)
    g_fun_list = []
    for i in range(len(g_mat)):
        g_fun = interp2d(k_grid, y_grid, g_mat[i].T)
        g_fun_list.append(g_fun)
    πc_fun_list = []
    for i in range(len(πc_mat)):
        πc_fun = interp2d(k_grid, y_grid, πc_mat[i].T)
        πc_fun_list.append(πc_fun)

    et = np.zeros(T+1)
    it = np.zeros(T+1)
    kt = np.zeros(T+1)
    yt = np.zeros(T+1)
    ht = np.zeros(T+1)
    gt = np.zeros((len(g_fun_list), T+1))
    πct = np.zeros((len(πc_fun_list), T+1))
    k0 = np.log(K0)

    for i in range(T+1):
        et[i] = e_fun(k0, y0)
        it[i] = i_fun(k0, y0)
        ht[i] = h_fun(k0, y0)
        for j in range(len(g_mat)):
            gt[j, i] = g_fun_list[j](k0, y0)
        for j in range(len(πc_mat)):
            πct[j, i] = πc_fun_list[j](k0, y0)
        kt[i] = k0
        yt[i] = y0
        y0 = y0 + et[i] * θ_mean
        k0 = k0 + it[i] - κ / 2 * it[i]**2 + μ_k - .5 * σ_k**2

    return et, kt, yt, it, gt, πct, ht
