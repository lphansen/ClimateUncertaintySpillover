#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

from numba import njit
from src.model_tech_dice import hjb_post_damage_post_tech, hjb_post_damage_pre_tech
from src.model_tech_dice import hjb_pre_damage_post_tech, hjb_pre_damage_pre_tech
from src.utilities import find_nearest_value
from src.simulation_2d import EvolutionState, simulation_dice_prob


# In[ ]:


##################
ξ_a = 2./100
ξ_p = 100000.
arrival = 20
τ = 2.
y_bar_lower = 1.5
n_model = 20
##################

I_g_first = 1./arrival
I_g_second = 1./arrival
ξ_g_first = ξ_p
ξ_g_second = ξ_p
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


# In[ ]:


# Solve post damage, post second tech jump models
model_post_damage_post_second_tech = []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, γ_1, γ_2, γ_3_i, τ, theta, lambda_bar_second, vartheta_bar_second)
    if i == 0:
        v_guess = None
    else:
        v_guess = model_res['v']
    model_res = hjb_post_damage_post_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=1., fraction=.05,
                                          tol=1e-6, max_iter=2000, print_iteration=False)
    model_post_damage_post_second_tech.append(model_res)


# In[ ]:


# Solve post damage, post first tech jump models
model_post_damage_post_first_tech = []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g_second, I_g_second,
                  model_post_damage_post_second_tech[i]['v'], γ_1, γ_2,
                  γ_3_i, τ, theta, lambda_bar_first, vartheta_bar_first)
    if i == 0:
        v_guess = model_res['v']
    else:
        v_guess = model_res['v']
    model_res = hjb_post_damage_pre_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=1., fraction=.05,
                                         tol=1e-6, max_iter=2000, print_iteration=False)
    model_post_damage_post_first_tech.append(model_res)


# In[ ]:


# Solve post damage pre tech jump models
model_post_damage_pre_tech = []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g_first, I_g_first,
                  model_post_damage_post_first_tech[i]['v'], γ_1, γ_2,
                  γ_3_i, τ, theta, lambda_bar, vartheta_bar)
    if i == 0:
        v_guess = model_res['v']
    else:
        v_guess = model_res['v']
    model_res = hjb_post_damage_pre_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=1., fraction=.05,
                                         tol=1e-6, max_iter=2000, print_iteration=False)
    model_post_damage_pre_tech.append(model_res)


# In[ ]:


# Solve pre damage, post second tech models
v_i_short = []
for model in model_post_damage_post_second_tech:
    temp = np.zeros((len(k_grid), len(y_grid_short)))
    for i in range(temp.shape[1]):
        temp[:, i] = model['v'][:, n_bar-1]
    v_i_short.append(temp)
v_i_short = np.array(v_i_short)

model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_p, πd_o, v_i_short, γ_1, γ_2, theta, lambda_bar_second, vartheta_bar_second, y_bar_lower)
model_pre_damage_post_second_tech = hjb_pre_damage_post_tech(k_grid, y_grid_short, model_args=model_args, v0=np.mean(v_i_short, axis=0),
                                                             ϵ=1., fraction=.05, tol=1e-6, max_iter=2_000, print_iteration=True)


# In[ ]:


# Solve pre damage, post first tech models
v_i_short = []
for model in model_post_damage_post_first_tech:
    temp = np.zeros((len(k_grid), len(y_grid_short)))
    for i in range(temp.shape[1]):
        temp[:, i] = model['v'][:, n_bar-1]
    v_i_short.append(temp)
v_i_short = np.array(v_i_short)

model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g_second, ξ_p,
              πd_o, v_i_short, I_g_second, model_pre_damage_post_second_tech['v'], γ_1, γ_2,
              theta, lambda_bar_first, vartheta_bar_first, y_bar_lower)

model_pre_damage_post_first_tech = hjb_pre_damage_pre_tech(k_grid, y_grid_short, model_args=model_args, v0=np.mean(v_i_short, axis=0),
                                                           ϵ=.1, fraction=.05, tol=1e-6, max_iter=2_000, print_iteration=False)


# In[ ]:


# Solve pre damage, pre tech models
v_i_short = []
for model in model_post_damage_pre_tech:
    temp = np.zeros((len(k_grid), len(y_grid_short)))
    for i in range(temp.shape[1]):
        temp[:, i] = model['v'][:, n_bar-1]
    v_i_short.append(temp)
v_i_short = np.array(v_i_short)

model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g_first, ξ_p,
              πd_o, v_i_short, I_g_first, model_pre_damage_post_first_tech['v'], γ_1, γ_2,
              theta, lambda_bar, vartheta_bar, y_bar_lower)

model_pre_damage_pre_tech = hjb_pre_damage_pre_tech(k_grid, y_grid_short, model_args=model_args, v0=np.mean(v_i_short, axis=0),
                                                    ϵ=.1, fraction=.05, tol=1e-6, max_iter=2_000, print_iteration=False)


# ## Distortion

# In[ ]:


# Case 1) : damage jump intensity & probability (no tech jump, no damage jump)
T_plots = 50
sim_args = (κ, μ_k, σ_k, np.mean(θ))
et, kt, yt, _, gt, πct, ht = simulation_dice_prob(sim_args, k_grid, y_grid_short,
                                      model_pre_damage_pre_tech['e'],
                                      model_pre_damage_pre_tech['i'],
                                      model_pre_damage_pre_tech['g'],
                                      model_pre_damage_pre_tech['h'],
                                      model_pre_damage_pre_tech['πc'],
                                      K0=85/0.115, y0=1.1, T=T_plots)

def damage_intensity(y, y_bar_lower):
    r1 = 1.5
    r2 = 2.5
    return r1 * (np.exp(r2/2 * (y - y_bar_lower)**2) - 1) * (y >= y_bar_lower)

intensity_dmg = damage_intensity(yt, y_bar_lower)
intensity_distortion = np.mean(gt, axis=0)
distorted_damage_probs = gt / np.mean(gt, axis=0) / n_model
np.save('et_nodmg_no_tech_baseline', et)

# In[ ]:


np.save('new_intensity_dmg_baseline.npy', intensity_dmg)
np.save(f'new_dmg_intensity_distort_baseline.npy', intensity_distortion)


# In[ ]:


# Case 2) : tech jump intensity & probability (no tech jump, no damage jump)
T_plots = 50
sim_args = (κ, μ_k, σ_k, np.mean(θ))
_, kt, yt, _, gt_tech, _, _ = simulation_dice_prob(sim_args, k_grid, y_grid_short,
                                      model_pre_damage_pre_tech['e'],
                                      model_pre_damage_pre_tech['i'],
                                      [model_pre_damage_pre_tech['g_tech']],
                                      model_pre_damage_pre_tech['h'],
                                      model_pre_damage_pre_tech['πc'],
                                      K0=85/0.115, y0=1.1, T=T_plots)

# Case 3) : tech jump intensity (tech jumped once, no damage jump)
sim_args = (κ, μ_k, σ_k, np.mean(θ))
_, kt_new, yt_new, _, gt_tech_new, _, _ = simulation_dice_prob(sim_args, k_grid, y_grid_short,
                                      model_pre_damage_post_first_tech['e'],
                                      model_pre_damage_post_first_tech['i'],
                                      [model_pre_damage_post_first_tech['g_tech']],
                                      model_pre_damage_post_first_tech['h'],
                                      model_pre_damage_post_first_tech['πc'],
                                      K0=np.exp(kt[arrival]), y0=yt[arrival], T=T_plots)


# In[ ]:


# Collect results
np.save('new_gt_tech_baseline.npy', gt_tech[0])
np.save('new_gt_tech_new_baseline.npy', gt_tech_new[0])

