# %%
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
import numpy as np
import pandas as pd
from model_2state import solve_prep, solve_pre_jump_2state
from multiprocessing import Pool
import argparse
#%%
parser = argparse.ArgumentParser(description="λ value")
parser.add_argument("-L", "--lam", type=float, help="set half life parameter")
args = parser.parse_args()
λ = args.lam
# %%
# parameters
δ = 0.01
η = 0.032
ξa = 0.01
θ_list = pd.read_csv("../data/model144.csv", header=None)[0].to_numpy()
θ_list = θ_list/1000
θ = np.mean(θ_list)
σy = 1.2*θ
# damage function
y_bar = 2
γ1 = 0.00017675
γ2 = 2*0.0022
γ3_list = np.linspace(0., 1./3, 20)
# %%
# y_grid
y1_step = .04
y1_grid = np.arange(0., 4., y1_step)

y2_step = .001
y2_grid = np.arange(0., .05, y2_step)

(y1_mat, y2_mat) = np.meshgrid(y1_grid, y2_grid, indexing = 'ij')

args_list  = []
for γ3_i in γ3_list:
    args_iter = (y1_grid, y2_grid, γ3_i, θ_list, (δ, η, γ1, γ2, y_bar, λ, ξa), 1e-6, 1., 1000, 0.05)
    args_list.append(args_iter)
# # #%%


# if not os.path.exists(f"./data/res_list_{λ}"):
if __name__ == "__main__":
    with Pool() as p:
        res_list = p.starmap(solve_prep, args_list)

    with open(f"../data/res_list_{λ}", "wb") as handle:
        pickle.dump(res_list, handle)

# step II HJB:
with open(f"../data/res_list_{λ}", "rb") as file:
    res_list = pickle.load(file)

ξp = 1
args_pre_jump = (δ, η, θ_list,  γ1, γ2, γ3_list, ξa, ξp)
res = solve_pre_jump_2state(res_list, args_pre_jump, ε=0.1)

with open(f"../data/res_{λ}_{ξp}", "wb") as handle:
    pickle.dump(res, handle)
