import os
import sys

from pandas.core.accessor import delegate_names
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pandas as pd
from scipy import interpolate
import pickle
from simulation import EvolutionState
import matplotlib.pyplot as plt

θ = pd.read_csv("data/model144.csv", header=None).to_numpy()[:, 0] / 1000.
y_grid_short = np.arange(0, 2.1 + 0.01, 0.01)
γ_3 = np.linspace(0, 1./3, 20)


pre_jump_15 = pickle.load(open("pre_jump_15", "rb"))
pre_jump_175 = pickle.load(open("pre_jump_175", "rb"))

v_list = pickle.load(open("v_list", "rb"))
e_tilde_list = pickle.load(open("e_tilde_list", "rb"))
v175_list = pickle.load(open("v175_list", "rb"))
e175_list = pickle.load(open("e175_tilde_list", "rb"))


e_grid_1 = pre_jump_15[1]["model_res"]["e_tilde"]
e_func_pre_damage = interpolate.interp1d(y_grid_short, e_grid_1)

y_grid_long = np.arange(0, 5., 0.01)
e_grid_long_1 = e_tilde_list[1]
e_func_post_damage = [interpolate.interp1d(y_grid_long, e_grid_long_1[i]) for i in range(len(γ_3))]

# start simulation
e0 = 0
k0 = np.log(85/0.115)
y0 = 1.1
temp_anol0 = 1.1
y_underline = 1.5
y_overline = 2.
initial_state = EvolutionState(t=0,
                               prob=1,
                               damage_jump_state='pre',
                               damage_jump_loc=None,
                               variables=[e0, y0, temp_anol0],
                               y_underline=y_underline,
                               y_overline=y_overline)

fun_args = (e_func_pre_damage, e_func_post_damage)

T = 400
sim_res = []
ys = []
probs = []
damage_locs = []
sim_res.append([initial_state])
for i in range(T):
    if i == 0:
        states = initial_state.evolve(np.mean(θ), fun_args)
    else:
        temp = []
        for state in states:
            temp += state.evolve(np.mean(θ), fun_args)
        states = temp
    ys_t = []
    probs_t = []
    damage_loc_t = []
    for state in states:
        ys_t.append( state.variables[1] )
        probs_t.append( state.prob )
        damage_loc_t.append( state.damage_jump_loc )

    ys.append(ys_t)
    probs.append(probs_t)
    damage_locs.append(damage_loc_t)
    sim_res.append(states)
    print(i+1, len(states))

state_T = np.array(damage_locs[-1])
probs_T = np.array(probs[-1])
ys_T = np.array(ys[-1])
plt.hist(ys_T[state_T != None], weights=probs_T[state_T != None]/np.sum(probs_T[state_T != None]), bins=40)
plt.show()
