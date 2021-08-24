#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

from numba import njit
from src.model_tech_dice import hjb_post_damage_post_tech, hjb_post_damage_pre_tech
from src.model_tech_dice import hjb_pre_damage_post_tech, hjb_pre_damage_pre_tech
from src.utilities import find_nearest_value
from src.simulation_2d import simulation_dice_prob
# In[2]:


##################
ξ_a = 2./100
ξ_p = 7.5
arrival = 100_000
τ = 2.
y_bar_lower = 1.5
n_model = 20
##################

I_g_first = 1./arrival
I_g_second = 1./arrival
ξ_g_first = 100_000
ξ_g_second = 100_000
ξ_b = ξ_p

# Model parameters
δ = 0.01
α = 0.115
κ = 6.667
μ_k = -0.043
σ_k = np.sqrt(0.0087**2 + 0.0038**2)

theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

γ_1 = 1.7675/10000
γ_2 = .0022*2

γ_3_lower = 0.
γ_3_upper = 1./3

# Compute γ_3 for the n models
def Γ(y, y_bar, γ_1, γ_2, γ_3):
    return γ_1 * y + γ_2 / 2 * y ** 2 + γ_3 / 2 * (y > y_bar) * (y - y_bar) ** 2

prop_damage_lower = np.exp(-Γ(2.5, 2., γ_1, γ_2, γ_3_upper))
prop_damage_upper = np.exp(-Γ(2.5, 2., γ_1, γ_2, γ_3_lower))
γ_3 = (-np.log(np.linspace(prop_damage_lower, prop_damage_upper, n_model)) - γ_1 * 2.5 - γ_2 / 2 * 2.5**2) / .5**2 * 2
γ_3.sort()
γ_3[0] = 0
πd_o = np.ones(n_model)/n_model

θ = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
πc_o = np.ones_like(θ)/len(θ)
σ_y = 1.2*np.mean(θ)

# Grid setting
k_step = .1
k_grid = np.arange(0, 10+k_step, k_step)

y_step = .1
y_grid_long = np.arange(0., 5.+y_step, y_step)
y_grid_short = np.arange(0., 2.5+y_step, y_step)
n_bar = find_nearest_value(y_grid_long, τ) + 1

# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.


# In[3]:


# Solve post damage, post second tech jump models
model_post_damage= []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, γ_1, γ_2, γ_3_i, τ, theta, lambda_bar, vartheta_bar)
    if i == 0:
        v_guess = None
    else:
        v_guess = model_res['v']
    model_res = hjb_post_damage_post_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=1., fraction=.05,
                                          tol=1e-6, max_iter=2000, print_iteration=False)
    model_post_damage.append(model_res)


e_post_damage = np.array([model_post_damage[i]['e'] for i in range(len(γ_3))])
np.save(f'e_post_damage_{ξ_p}_no_tech', e_post_damage)

# In[6]:


# Solve pre damage, post second tech models
v_i_short = []
for model in model_post_damage:
    temp = np.zeros((len(k_grid), len(y_grid_short)))
    for i in range(temp.shape[1]):
        temp[:, i] = model['v'][:, n_bar-1]
    v_i_short.append(temp)
v_i_short = np.array(v_i_short)

model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_p, πd_o, v_i_short, γ_1, γ_2, theta, lambda_bar, vartheta_bar, y_bar_lower)
model_pre_damage = hjb_pre_damage_post_tech(k_grid, y_grid_short, model_args=model_args, v0=np.mean(v_i_short, axis=0),
                                                             ϵ=1., fraction=.05, tol=1e-6, max_iter=2_000, print_iteration=True)

np.save(f'e_pre_damage_{ξ_p}_no_tech', model_pre_damage['e'])




