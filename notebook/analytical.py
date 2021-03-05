# -*- coding: utf-8 -*-
# +
from IPython.display import display
import sympy as sy
from sympy import log, Piecewise, Max, Derivative
from sympy.solvers.ode import dsolve
import matplotlib.pyplot as plt
import numpy as np

sy.init_printing()  # LaTeX like pretty printing for IPython


y = sy.symbols("y", real=True, positive=True)
f = sy.Function("f", function=True)(y)
e = sy.symbols("e", positive=True)
dΛ = sy.symbols('dΛ')
μ = sy.symbols("μ")
τ_1 = sy.symbols('τ_1')
τ_2 =  sy.symbols('τ_2')
τ_2p =  sy.symbols('τ_2p')
τ_bar = sy.symbols('τ_bar', positive=True)
δ = sy.symbols('δ')
η = sy.symbols('η')

dΛ = τ_1 + τ_2*y + τ_2p*Max(y-τ_bar, 0)
e = -δ*η/(Derivative(f, y)*μ + (η -1)*dΛ*μ) 
eq1 = sy.Eq(-δ*f + Derivative(f, y)*μ*e + δ*η*log(e) + (η-1)*dΛ*μ*e,0)# the equation 
sls = dsolve(eq1, f)  # solvde ODE
# -

print("For ode")
display(eq1)
print("the solutions are:")
for s in sls:
    display(s)
