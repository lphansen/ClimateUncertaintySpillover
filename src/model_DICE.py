#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pandas as pd

import model_tech_dice as mtd
import utilities as utls
import simulation_2d as s2d
import argparse

parser = argparse.ArgumentParser(description="Solve and simulate DICE model in section 9")
parser.add_argument("-a", "--xi_a", type=float, help="Set ξ_a value for the model")
parser.add_argument("-r", "--xi_r", type=float, help="Set ξ_r value for the model")
args = parser.parse_args()

##################
ξ_a = args.xi_a
ξ_p = args.xi_r

if ξ_a == 100_000 and ξ_p == 100_000:
    model = "baseline"
else:
    model = str(ξ_p).replace(".", "p")

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

θ = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
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
print("===========Solving post damage post second tech jump models==============")
model_post_damage_post_second_tech = []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, γ_1, γ_2, γ_3_i, τ, theta, lambda_bar_second, vartheta_bar_second)
    if i == 0:
        v_guess = None
    else:
        v_guess = model_res['v']
    model_res = mtd.hjb_post_damage_post_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=1., fraction=.05,
                                          tol=1e-6, max_iter=2000, print_iteration=False)
    model_post_damage_post_second_tech.append(model_res)


# In[4]:


# Solve post damage, post first tech jump models
print("====================Solving post damage post first tech jump models==================")
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



# Solve post damage pre tech jump models
print("==================Solving post damage pre tech jump models============================")
model_post_damage_pre_tech = []
for i, γ_3_i in enumerate(γ_3):
    model_args = (δ, α, κ, μ_k, σ_k, θ, πc_o, σ_y, ξ_a, ξ_b, ξ_g_first, I_g_first,
                  model_post_damage_post_first_tech[i]['v'], γ_1, γ_2,
                  γ_3_i, τ, theta, lambda_bar, vartheta_bar)
    v_guess = model_post_damage_post_first_tech[i]['v']
    model_res = hjb_post_damage_pre_tech(k_grid, y_grid_long, model_args, v0=v_guess, ϵ=.2, fraction=.01,
                                         tol=5e-3, max_iter=1000, print_iteration=False)
    model_post_damage_pre_tech.append(model_res)


# In[6]:


# Solve pre damage, post second tech models
print("=====================Solving pre damage post second tech models==========================")
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


# In[7]:


# Solve pre damage, post first tech models
print("=====================Solving pre damage post first tech models=============================")
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


# In[8]:


# Solve pre damage, pre tech models
print("=======================Solving pre damage pre tech models====================================")
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


# ## Simulation

# In[9]:


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


# In[10]:

file_intensity = open(f'data/new_intensity_dmg_{ξ_p}.npy', "wb")
np.save(file_intensity, intensity_dmg)
file_dmg_intensity_distort = open(f'data/new_dmg_intensity_distort_{ξ_p}.npy', "wb")
np.save(file_dmg_intensity_distort, intensity_distortion)


# In[11]:


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


# In[12]:

file_gt_tech = open(f'../data/new_gt_tech_{ξ_p}.npy',"wb")
np.save(file_gt_tech, gt_tech[0])
file_gt_tech_new = open(f'../data/new_gt_tech_new_{ξ_p}.npy', "wb")
np.save(file_gt_tech_new, gt_tech_new[0])

