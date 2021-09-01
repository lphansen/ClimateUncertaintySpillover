# -*- coding: utf-8 -*-
# +
"""
module for simulation
"""
import numpy as np
import pandas as pd
import ray
from scipy import interpolate
from .utilities import J

# function claim
# -


def simulate_jump(model_res, θ_list, ME=None,  y_start=1,  T=100, dt=1):
    """
    Simulate temperature anomaly, emission, distorted probabilities of climate models,
    distorted probabilities of damage functions, and drift distortion.
    When ME is asigned value, it will also simulate paths for marginal value of emission

    Parameters
    ----------
    model_res : dict
        A dictionary storing solution with misspecified jump process.
        See :func:`~source.model.solve_hjb_y_jump` for detail.
    θ_list : (N,) ndarray::
        A list of matthew coefficients. Unit: celsius/gigaton of carbon.
    ME : (N,) ndarray
        Marginal value of emission as a function of y.
    y_start : float, default=1
        Initial value of y.
    T : int, default=100
        Time span of simulation.
    dt : float, default=1
        Time interval of simulation.

    Returns
    -------
    simulation_res: dict of ndarrays
        dict: {
            yt : (T,) ndarray
                Temperature anomaly trajectories.
            et : (T,) ndarray
                Emission trajectories.
            πct : (T, L) ndarray
                Trajectories for distorted probabilities of climate models.
            πdt : (T, M) ndarray
                Trajectories for distorted probabilities of damage functions.
            ht : (T,) ndarray
                Trajectories for drift distortion.
            if ME is not None, the dictionary will also include
                me_t : (T,) ndarray
                    Trajectories for marginal value of emission.
        }
    """
    y_grid = model_res["y"]
    ems = model_res["e_tilde"]
    πc = model_res["πc"]
    πd = model_res["πd"]
    h = model_res["h"]
    periods = int(T/dt)
    et = np.zeros(periods)
    yt = np.zeros(periods)
    πct = np.zeros((periods, len(θ_list)))
    πdt = np.zeros((periods, len(πd)))
    ht = np.zeros(periods)
    if ME is not None:
        me_t = np.zeros(periods)
    # interpolate
    get_πd = interpolate.interp1d(y_grid, πd)
    get_πc = interpolate.interp1d(y_grid, πc)
#     y = np.mean(θ_list)*290
    y = y_start
    for t in range(periods):
        if y > np.max(y_grid):
            break
        else:
            ems_point = np.interp(y, y_grid, ems)
            πd_list = get_πd(y)
            πc_list = get_πc(y)
            h_point = np.interp(y, y_grid, h)
            if ME is not None:
                me_point = np.interp(y, y_grid, ME)
                me_t[t] = me_point
            et[t] = ems_point
            πdt[t] = πd_list
            πct[t] = πc_list
            ht[t] = h_point
            yt[t] = y
            dy = ems_point*np.mean(θ_list)*dt
            y = dy + y
    if ME is not None:
        simulation_res = dict(yt=yt, et=et, πct=πct, πdt=πdt, ht=ht, me_t=me_t)
    else:
        simulation_res = dict(yt=yt, et=et, πct=πct, πdt=πdt, ht=ht)
    return simulation_res


def simulate_me(y_grid, e_grid, ratio_grid, θ=1.86/1000., y_start=1, T=100, dt=1):
    """
    simulate trajectories of uncertainty decomposition

    .. math::

        \\log(\\frac{ME_{new}}{ME_{baseline}})\\times 1000.

    Parameters
    ----------
    y_grid : (N, ) ndarray
        Grid of y.
    e_grid : (N, ) ndarray
        Corresponding :math:`\\tilde{e}` on the grid of y.
    ratio_grid : (N, ) ndarray::
        Corresponding :math:`\\log(\\frac{ME_{new}}{ME_{baseline}})\\times 1000` on the grid of y.
    θ : float, default=1.86/1000
        Coefficient used for simulation.
    y_start : floatsimulation
        Initial value of y.
    T : int, default=100
        Time span of simulation.
    dt : float, default=1
        Time interval of simulation. Default=1 indicates yearly simulation.

    Returns
    -------
    Et : (T, ) ndarray
        Emission trajectory.
    yt : (T, ) ndarray
        Temperature anomaly trajectories.
    ratio_t : (T, ) ndarray
        Uncertainty decomposition ratio trajectories.
    """
    periods = int(T/dt)
    Et = np.zeros(periods+1)
    yt = np.zeros(periods+1)
    ratio_t = np.zeros(periods+1)
    for i in range(periods+1):
        Et[i] = np.interp(y_start, y_grid, e_grid)
        ratio_t[i] = np.interp(y_start, y_grid, ratio_grid)
        yt[i] = y_start
        y_start = y_start + Et[i]*θ
    return Et, yt, ratio_t


def no_jump_simulation(
    model_res,
    y_start=1.1,
    T=130,
    dt=1,
):
    y = y_start
    periods = int(T / dt)
    e_tilde = model_res["e_tilde"]
    y_grid_short = model_res["y"]
    h = model_res["h"]
    πc = model_res["πc"]
    πd = model_res["πd"]
    y_bar = model_res["model_args"][4]
    θ_list = model_res["model_args"][8]
    et = np.zeros(periods)
    yt = np.zeros(periods)
    ht = np.zeros(periods)
    probt = np.zeros(periods)
    πdt = np.zeros((periods, len(πd)))
    πct = np.zeros((periods, len(πc)))

    get_d = interpolate.interp1d(y_grid_short, πd)
    get_c = interpolate.interp1d(y_grid_short, πc)
    for t in range(periods):
        if y <= y_bar:
            e_i = np.interp(y, y_grid_short, e_tilde)
            h_i = np.interp(y, y_grid_short, h)
            intensity = J(y)
            et[t] = e_i
            ht[t] = h_i
            probt[t] = intensity * dt
            yt[t] = y
            πct[t] = get_c(y)
            πdt[t] = get_d(y)
            y = y + e_i * np.mean(θ_list) * dt
        else:
            break
    yt = yt[np.nonzero(yt)]
    et = et[np.nonzero(et)]
    ht = ht[np.nonzero(ht)]
    probt = probt[:len(yt)]
    πdt = πdt[:len(yt)]
    πct = πct[:len(yt)]

    res = {
        "et": et,
        "yt": yt,
        "probt": probt,
        "πct": πct,
        "πdt": πdt,
        "ht": ht,
    }

    return res


def damage_intensity(y, y_underline):
    r1 = 1.5
    r2 = 2.5
    return r1 * (np.exp(r2/2 * (y - y_underline)**2) - 1) * (y >= y_underline)


class EvolutionState:
    DAMAGE_MODEL_NUM = 20
    DAMAGE_PROB = np.ones(20) / 20
    dt = 1/4

    def __init__(self, t, prob, damage_jump_state, damage_jump_loc, variables, y_underline, y_overline):
        self.t = t
        self.prob = prob
        self.damage_jump_state = damage_jump_state
        self.damage_jump_loc = damage_jump_loc
        self.variables = variables
        self.y_underline = y_underline
        self.y_overline = y_overline

    def set_damage(self, damage_model_num):
        """set damage model number
        """
        self.DAMAGE_MODEL_NUM = damage_model_num
        self.DAMAGE_PROB = np.ones(
            self.DAMAGE_MODEL_NUM) / self.DAMAGE_MODEL_NUM

    def set_time_step(self, dt):
        self.dt = dt

    def copy(self):
        return EvolutionState(self.t,
                              self.prob,
                              self.damage_jump_state,
                              self.damage_jump_loc,
                              self.variables,
                              self.y_underline,
                              self.y_overline)

    def evolve(self, θ_mean, fun_args):

        e_fun_pre_damage, e_fun_post_damage = fun_args
        [e, y, temp_anol] = self.variables

        # Compute variables at t+1
        if self.damage_jump_state == 'pre':
            e_fun = e_fun_pre_damage
        elif self.damage_jump_state == 'post':
            e_fun = e_fun_post_damage[self.damage_jump_loc]
        else:
            raise ValueError(
                'Invalid damage jump state. Should be one of [pre, post]')

        e_new = e_fun(y)
        y_new = y + e_new * θ_mean * self.dt
        temp_anol_new = temp_anol + e_new * θ_mean * self.dt
        variables_new = [e_new, y_new, temp_anol_new]
        res_template = EvolutionState(self.t+ self.dt,
                                      self.prob,
                                      self.damage_jump_state,
                                      self.damage_jump_loc,
                                      variables_new,
                                      self.y_underline,
                                      self.y_overline)

        states_new = []

        # Update probabilities
        temp = damage_intensity(y_new, self.y_underline)

        damage_jump_prob = temp * self.dt
        if damage_jump_prob > 1:
            damage_jump_prob = 1
        # damage has not jumped
        if self.damage_jump_state == 'pre' and damage_jump_prob != 0:
            # Damage jumps
            for i in range(self.DAMAGE_MODEL_NUM):
                temp = res_template.copy()
                temp.prob *= self.DAMAGE_PROB[i] * damage_jump_prob
                temp.damage_jump_state = 'post'
                temp.damage_jump_loc = i
                temp.variables[1] = self.y_overline
                states_new.append(temp)

            # Damage does not jump
            temp = res_template.copy()
            temp.prob *= (1 - damage_jump_prob)
            temp.damage_jump_state = 'pre'
            states_new.append(temp)
        # damage has jumped
        else:
            temp = res_template.copy()
            temp.prob *= 1
            states_new.append(temp)

        return states_new


"""
module for simulation
"""
def simulate_jump_2(model_res_pre, model_res_post, y_upper, θ_list, ME=None,  y_start=1.1,  T=100, dt=1):
    """
    Simulate temperature anomaly, emission, distorted probabilities of climate models,
    distorted probabilities of damage functions, and drift distortion.
    When ME is asigned value, it will also simulate paths for marginal value of emission

    Parameters
    ----------
    model_res : dict
        A dictionary storing solution with misspecified jump process. 
        See :func:`~source.model.solve_hjb_y_jump` for detail.
    θ_list : (N,) ndarray::
        A list of matthew coefficients. Unit: celsius/gigaton of carbon.
    ME : (N,) ndarray
        Marginal value of emission as a function of y.
    y_start : float, default=1
        Initial value of y.
    T : int, default=100
        Time span of simulation.
    dt : float, default=1
        Time interval of simulation.

    Returns
    -------
    simulation_res: dict of ndarrays
        dict: {
            yt : (T,) ndarray
                Temperature anomaly trajectories.
            et : (T,) ndarray
                Emission trajectories.
            πct : (T, L) ndarray
                Trajectories for distorted probabilities of climate models.
            πdt : (T, M) ndarray
                Trajectories for distorted probabilities of damage functions.
            ht : (T,) ndarray
                Trajectories for drift distortion.
            if ME is not None, the dictionary will also include
                me_t : (T,) ndarray
                    Trajectories for marginal value of emission.
        }
    """
    y_grid = model_res_pre["y"]
    ems = model_res_pre["e_tilde"]
    πc = model_res_pre["πc"]
    πd = model_res_pre["πd"]
    h = model_res_pre["h"]
    periods = int(T/dt)
    et = np.zeros(periods)
    yt = np.zeros(periods)
    πct = np.zeros((periods, len(θ_list)))
    πdt = np.zeros((periods, len(πd)))
    ht = np.zeros(periods)
    if ME is not None:
        me_t = np.zeros(periods)
    # interpolate
    get_πd = interpolate.interp1d(y_grid, πd)
    get_πc = interpolate.interp1d(y_grid, πc)
#     y = np.mean(θ_list)*290
    y = y_start
    kkkk=0
    threshold = 0
    for t in range(periods):
            if y_upper >= y:
                ems_point = np.interp(y, y_grid, ems)
                πd_list = get_πd(y)
                πc_list = get_πc(y)
                h_point = np.interp(y, y_grid, h)
                if ME is not None:
                    me_point = np.interp(y, y_grid, ME)
                    me_t[t] = me_point
                et[t] = ems_point
                πdt[t] = πd_list
                πct[t] = πc_list
                ht[t] = h_point
                yt[t] = y
                dy = ems_point*np.mean(θ_list)*dt
                y = dy + y
                K=t
            else:
                if kkkk==0:
                    threshold = K
                    y_grid = model_res_post["y"]
                    ems    = model_res_post["e_tilde"]
                    πc     = model_res_post["πc"]
                    h      = model_res_post["h"]
                    get_πc = interpolate.interp1d(y_grid, πc)
                    kkkk=1
                ems_point = np.interp(y, y_grid, ems)
                πᶜ_list = get_πᶜ(y)
                et[t] = ems_point
                πᶜt[t] = πᶜ_list
                ht[t] = h_point
                yt[t] = y
                dy = ems_point*np.mean(θ_list)*dt
                y = dy + y
                K=t
    if ME is not None:
        simulation_res = dict(yt=yt, et=et, πct=πct, πdt=πdt, ht=ht, me_t=me_t)
    else:
        simulation_res = dict(yt=yt[0:K], et=et[0:K], πct=πct[0:K], πdt=πdt[0:K], ht=ht[0:K], threshold = threshold)
    return simulation_res


# +
def Γ(y, γ_3, γ_1=0.00017675, γ_2=2*0.0022, y_overline=2.):
    logN = γ_1 * y + γ_2/2 * y**2 + γ_3/2 * (y - y_overline)**2 * (y > y_overline)
    return logN

def Intensity(y, r1=1.5, r2=2.5, y_underline=1.5):
    return r1 * (np.exp(r2 / 2 *
                       (y - y_underline)**2) - 1) * (y >= y_underline)






# +
def one_path(Et, θ,args=(), dt=1, Y0=1.1):
    γ_1, γ_2, γ_3_list = args
    T = len(Et)
    Yt = np.zeros(T+1)
    logNt = np.zeros(T+1)
    Yt[0] = Y0
    logNt[0] = Γ(Yt[0], 0)
    jumped = False
    γ_3_j = 0
    rng = np.random.default_rng()
    for i in range(T):
        J_y = Intensity(Yt[i])
        jump_prob = J_y * dt
        jump_prob = jump_prob * (jump_prob <= 1) + (jump_prob > 1)
        if jump_prob > 0 and jumped == False:
            # could jump:
            jump_bool = rng.choice([False,True], size=1, p=[1 - jump_prob, jump_prob])
            if jump_bool:
                # jump occurs
                # to one of the damage functions
                γ_3_j =rng.choice(γ_3_list)
                logNt[i] = Γ(Yt[i], γ_3_j)
                Yt[i+1] = 2
                jumped = True
            else:
                logNt[i] = Γ(Yt[i], γ_3_j)
                Yt[i+1] = Yt[i] + θ * Et[i] *dt
        else:
            logNt[i] = Γ(Yt[i], γ_3_j)
            Yt[i+1] = Yt[i] + θ * Et[i] *dt
    logNt[i+1] = Γ(Yt[i+1], γ_3_j)
    return Yt, logNt



# -

def simulation_jump_exo(Et, θ, k, args=(), dt=1, Y0=1.1):
    γ_1, γ_2, γ_3_list = args
    T = len(Et)
    yt = np.zeros(T+1 )
    Damage_func = np.zeros(T+1 )
    r1 = 1.5
    r2 = 2.
    γ3= 0
    Jump = 0
    y_lower = 1.5
    y_upper= 2.0
    γ3=0
    yt[0] = Y0
    Damage_func[0] = Γ(yt[0],γ3)
    rng = np.random.default_rng(k)
    for i in range(T):
        jump_prob = Intensity(yt[i]) * dt
        jump_prob = jump_prob * (jump_prob <= 1) + (jump_prob > 1)
        if jump_prob > 0 and Jump == 0:
            Jump      = rng.choice([0,1], size=1, p=[1 - jump_prob, jump_prob])
            if Jump ==0:
                Damage_func[i] = Γ(yt[i],γ3)
                yt[i+1] = yt[i]+ θ * Et[i] *dt
                else_loop = 0
            elif Jump >= 1:
                if else_loop == 0:
                    Jump = 1
                    γ3 = rng.choice(γ_3_list)
                    Damage_func[i] = Γ(yt[i],γ3)
                    yt[i+1] = 2
                    else_loop = 1
        else: 
            Damage_func[i] = Γ(yt[i],γ3)
            yt[i+1]= yt[i] + Et[i] * θ * dt
    Damage_func[T] =Γ(yt[i],γ3)
    result = dict(  Et=Et, yt=yt, Damages=Damage_func, γ3=γ3)

    return (result)


