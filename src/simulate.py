"""
This script is used to simualte emission for high and low damage.
"""
# packages
import os, sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')

import numpy as np
import matplotlib.pyplot as plt
from simulation import simulate_emission_quadratic, simulate_log_damage
import pickle
import time
##################start of the simulation########################

# model parameters
δ = 0.01
η = 0.032
median = 1.75/1000
h_hat = 0.15
σ_n = 1.2
γ_low = .012
γ_high = .024
γ_base = .018
ξ = σ_n/h_hat*δ*η

# simulation settings: initial reserve, simulation timespan and function arguments
r_start = 1500
T = 500
args_trace_ϕ = (-20, -3, 5000, 1e-9, 1e-3)

e_simul = {}
for γ, dmgspec in [[γ_high, "high"], [γ_low, "low"], [γ_base, "base"]]:
    print("Simulate for {dmg} damage".format(dmg = dmgspec))
    e_simul[dmgspec] = {}
    for a, b in  [[1, 0], [1, 1], [.5, 1]]:
        start = time.time()
        e, _, _, _ = simulate_emission_quadratic(δ, η, median*γ, a*ξ, b*σ_n,
                                                 args_trace_ϕ=args_trace_ϕ,
                                                 r_start=r_start, T=T)
        end = time.time()
        if b == 0:
            key = "tau0"
        else:
            if a == 1:
                key = 'xi1'
            elif a == .5:
                key = "xi05"

        e_simul[dmgspec][key] = e
        print("- Simulate for " + key + ": {time:.3f}s".format(time=end-start))

family_dir = os.path.dirname(os.path.getcwd())
filename = family_dir + '/data'
with open(filename, "wb") as handle:
    pickle.dump(e_simul, handle)
