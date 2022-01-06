
.. admonition:: \   

    This notebook can be accessed through mybinder.org. Click to load the project: |binder|

.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/lphansen/ClimateUncertaintySpillover.git/macroAnnual_v2?filepath=appendixC.ipynb

Appendix C: when :math:`Y` has two states:
==========================================

Recall in remark 1 in :ref:`section
2 <remark2p1>`, we capture the initial rise
in the emission responses by the following two-dimensional
specification:

.. math::


   dY_t^1  = Y_t^2 dt

.. math::


   dY_t^2 =   - \lambda Y_t^2 dt + \lambda \theta \mathcal{E} dt

which implies the response to a pulse is:

.. math::


   \theta \left[ 1 - \exp( - \lambda t) \right] {\mathcal E}_0

A high value of :math:`\lambda` implies more rapid convergence to the
limiting response :math:`\theta {\mathcal E}_0`. This approximation is
intended as a simple representation of the dynamics where the second
state variable can be thought of as an exponentially weighted average of
current and past
emissions.

Step I: post jump HJB:
----------------------

The post jump HJB for :math:`\phi_m(y_1, y_2)`, :math:`m = 1, \dots, M`:

.. math::


   \begin{aligned}
   0 = \max_{\mathcal{E}} \min_{\omega_\ell } & - \delta \phi_m(y_1, y_2) + \eta log(\mathcal{E}) \\
   & + \frac{\partial \phi_m}{\partial y_1} y_2 + \frac{\partial \phi_m}{\partial y_2} \lambda (- y_2 + \sum_{\ell = 1}^L 
   \omega_\ell  \theta_\ell \mathcal{E}) \\ 
   & + \frac{(\eta - 1)}{\delta} \left(\gamma_1 + \gamma_2 y_1 + \gamma_3 (y_1 - \bar y)\mathbb{I}\{y_1>\bar y\} \right) y_2 \\
   & + \xi_a \sum_{\ell = 1}^L \omega_\ell (\log \omega_\ell - \log \pi^a_\ell)
   \end{aligned}

First order condition for :math:`\omega_\ell`,
:math:`\ell = 1, \dots, L`:

.. math::


       \omega_\ell \propto \pi_\ell^a \exp\left( -\frac{1}{\xi_a} \frac{\partial \phi_m}{\partial y_2}\lambda \theta_\ell \mathcal{E} \right), for \ell = 1, \dots, L

and the first order condition for emission is:

.. math::


   \mathcal{E} = - \cfrac{\eta}{\frac{\partial \phi_m }{\partial y_2} \lambda \sum_{\ell=1}^{L} \omega_\ell \theta_\ell}

Step II: pre jump HJB:
----------------------

Given post jump value functions :math:`\phi_m(y_1, y_2)`,
:math:`m = 1, 2, M`, solve the following HJB for pre-jump value function
:math:`\Phi(y_1, y_2)`:

.. math::


   \begin{aligned}
   0 = \max_{\mathcal{E}}\min_{\omega_\ell }  & - \delta \Phi(y_1, y_2) +  \eta log(\mathcal{E}) \\
   & + \frac{\partial \Phi}{\partial y_1} y_2 + \frac{\partial \Phi}{\partial y_2} \lambda (- y_2+ \sum_{\ell = 1}^L 
   \omega_\ell  \theta_\ell \mathcal{E}) \\ 
   & + \frac{(\eta - 1)}{\delta} (\gamma_1 + \gamma_2 y_1 ) y_2 \\
   & + \xi_a \sum_{\ell = 1}^L \omega_\ell (\log \omega_\ell - \log \pi^a_\ell)\\
   & + \mathcal{J}(y_1) \sum_{m=1}^M g_m \pi_d^m ( \phi_m(\bar{y}_1, y_2) - \Phi(y_1, y_2)) \\
   & + \xi_p \mathcal{J}(y_1) \sum_{m=1}^M \pi_d^m \left(1 - g_m + g_m \log (g_m)\right)
   \end{aligned}

Or solve the following HJB with a terminal condition:

.. math::


   \begin{aligned}
   0 = \max_{\mathcal{E}}\min_{\omega_\ell }  & - \delta \Phi(y_1, y_2) +  \eta log(\mathcal{E}) \\
   & + \frac{\partial \Phi}{\partial y_1} y_2 + \frac{\partial \Phi}{\partial y_2} \lambda (- y_2+ \sum_{\ell = 1}^L 
   \omega_\ell  \theta_\ell \mathcal{E}) \\ 
   & + \frac{(\eta - 1)}{\delta} (\gamma_1 + \gamma_2 y_1 ) y_2 \\
   & + \xi_a \sum_{\ell = 1}^L \omega_\ell (\log \omega_\ell - \log \pi^a_\ell)
   \end{aligned}

.. math::


   \Phi(\bar y_1, y_2) \approx  - \xi_p \log \left (\sum_{m=1}^M \pi_m^p \exp\left[-\frac{1}{\xi_p }\phi_m(\bar y_1, y_2) \right] \right) 

In what follows, we show emission, :math:`Y` behavior before temperature
anomay reaches the upper bound of jump threshold, :math:`2^o C`,
conditioning on no jump.

The :math:`\lambda` we choose here are: - :math:`\lambda = 0.116`,
corresponds to half life of six years; - :math:`\lambda = 1, 2, 5`.
Here, we try to illustrate emission behavior as :math:`\lambda`
increase.

Example 1: Half life of six years
---------------------------------

With a half life of six years, the state evolution is as shown in the figure below.
The evolution follows the process described above with baseline probabilities.

.. toggle::

   .. code:: ipython3

       # packages
       import numpy as np
       import pandas as pd
       import pickle
       from src.model_2state import solve_prep, solve_pre_jump_2state, solve_pre_jump_2state_2
       from src.simulation_2d import simulation_2d, simulate_logkapital
       from src.plots import plot_2S_ey1y2, plot_1S_vs_2S_ems, plot_1S_vs_2S_SCC, plot_2S_ey1y2_multi_λ, plot_1S_vs_2S_ems_multi_λ, plot_1S_vs_2S_SCC_multi_λ 
       import plotly.graph_objects as go
       from plotly.subplots import make_subplots
       from scipy.interpolate import interp2d





   .. code:: ipython3

       λ = np.log(2) / 6
       # parameters
       δ = 0.01
       η = 0.032
       ξ_a = 0.01
       θ_list = pd.read_csv("./data/model144.csv", header=None)[0].to_numpy()
       θ_list = θ_list/1000
       θ = np.mean(θ_list)
       σy = 1.2*θ
       # damage function
       y_bar = 2.
       γ_1 = 0.00017675
       γ_2 = 2*0.0022
       γ_3_list = np.linspace(0., 1./3, 20)
       ξ_r = 1.
       # %%
       # y_grid
       y1_step = .04
       y1_grid = np.arange(0., 4. + y1_step, y1_step)
    
       y2_step = .001
       y2_grid = np.arange(0., .05 + y2_step, y2_step)

   .. code:: ipython3

       args_list  = []
       for γ_3_i in γ_3_list:
           args_iter = (y1_grid, y2_grid, γ_3_i, θ_list, (δ, η, γ_1, γ_2, y_bar, λ, ξ_a), 1e-6, 1., 1000, 0.05)
           args_list.append(args_iter)
    
       if not os.path.exists(f"./data/res_list_{λ}.pickle"):
           with Pool() as p:
               res_list = p.starmap(solve_prep, args_list)
    
           with open(f"./data/res_list_{λ}.pickle", "wb") as handle:
               pickle.dump(res_list, handle)
    
       # step II HJB:
       with open(f"./data/res_list_{λ}.pickle", "rb") as file:
           res_list = pickle.load(file)
    
       if not os.path.exists(f"./data/res_{λ}_{ξ_r}.pickle"):
           args_pre_jump = (δ, η, θ_list,  γ_1, γ_2, γ_3_list, ξ_a, ξ_r)
           res = solve_pre_jump_2state(res_list, args_pre_jump, ε=0.1)
           with open(f"./data/res_{λ}_{ξ_r}.pickle", "wb") as handle:
               pickle.dump(res, handle)
    
       #load results
       res = pickle.load(open(f"./data/res_{λ}_{ξ_r}.pickle", "rb"))

   .. code:: ipython3

       def simulation_2d(res, θ=1.86/1000., y1_0=1.1, y2_0=1.86/1000, T=100):
           y1_grid = res["y1"]
           y2_grid = res["y2"]
           e_grid = res["ems"]
           λ = res["λ"]
           e_fun = interp2d(y1_grid, y2_grid, e_grid.T)
           Et = np.zeros(T+1)
           y1t = np.zeros(T+1)
           y2t = np.zeros(T+1)
           for i in range(T+1):
               Et[i] = e_fun(y1_0, y2_0)
               y1t[i] = y1_0
               y2t[i] = y2_0
               y2_0 = y2_0 - λ*y2_0 + λ*θ*Et[i] 
               y1_0 = y1_0 + y2_0
           return Et, y1t, y2t

   .. code:: ipython3

       et_prejump, y1t_prejump, y2t_prejump = simulation_2d(res, θ=np.mean(θ_list), y1_0 = 1.1, y2_0=np.mean(θ_list), T=110)

   .. code:: ipython3

       simul = {
           "et": et_prejump,
           "y1t": y1t_prejump,
           "y2t": y2t_prejump,
       }
    
       pickle.dump(simul, open(f"data/simul_{λ}.pickle", "wb"))


    fig = plot_2S_ey1y2(simul)
    fig



    

.. raw:: html

    <iframe frameBorder="0" width="100%" height="400px" src="./_static/appC1.html"></iframe>

Compare the emission trajectories with the one state model:

.. raw:: html

   <iframe height="500px", width="100%" src="./_static/appC2.html" frameBorder="0"></iframe>

In the following plot, we show the social cost of carbon conditioning on
no jump happening with comparison to the model with one state.

.. toggle::

   .. code:: ipython3

       # capital related parameters
       invkap = 0.09
       α = 0.115
       αₖ = - 0.043
       σₖ = 0.0095
       κ = 6.667
       k0 = 85/α
    
       # Capital simulation
       Kt = simulate_logkapital(invkap, αₖ, σₖ, κ,  k0, T=111)
       MC = δ*(1-η)/((α - invkap)*np.exp(Kt))
       scc = η*(α - invkap)*np.exp(Kt)/(1-η)/et_prejump*1000
       scc_1 = η*(α - invkap)*np.exp(Kt[:len(et_1state)])/(1-η)/et_1state*1000
    
       # scc comparison
       fig = plot_1S_vs_2S_SCC(et_1state, scc[:101], scc_1)
       fig.update_xaxes(showline=True)
       fig.update_yaxes(rangemode="tozero")



.. raw:: html

   <iframe height="500px", width="100%" src="./_static/appC3.html" frameBorder="0"></iframe>

Example 2: increasing :math:`\lambda`
-------------------------------------

For the purpose of illustration, we choose :math:`\lambda = 1, 2, 5` to
show what happens if the half life decreases (click to see the detailed code).

.. toggle::

   .. code:: ipython3

       y2_step = .001
       y2_grid = np.arange(0., .05, y2_step) 
       number_of_cpu = joblib.cpu_count()
       v_dict = {}
       e_dict = {}
       λ_list = [1, 2, 5]
       for λ in λ_list:
           if not os.path.exists(f"data/v_list_{λ}.npy"):
               delayed_funcs = [delayed(solve_prep)(y1_grid, y2_grid, γ_3_i, θ_list, 
                                             (δ, η, γ_1, γ_2, y_bar, λ, ξ_a), 1e-6, 0.01, 10000, 0.02) for γ_3_i in γ_3_list]
               parallel_pool = Parallel(n_jobs=number_of_cpu)
               res_list = parallel_pool(delayed_funcs)
               v_list = np.zeros((len(γ_3_list), len(y1_grid), len(y2_grid)))
               for i in range(len(γ_3_list)):
                   v_list[i] = res_list[i]["v0"]
           else:
               v_list = np.load(f"data/v_list_{λ}.npy")
           if not os.path.exists(f"data/v_{λ}.npy"):
               args_pre_jump = (δ, η, θ_list,  γ_1, γ_2, γ_3_list, ξ_a, ξ_r)
               res = solve_pre_jump_2state_2(y1_grid, y2_grid, λ, v_list, args_pre_jump, ϵ=0.1, tol=1e-5, max_iter=5000)
               v_dict[λ] = res["v0"]
               e_dict[λ] = res["ems"]
           else:
               v_dict[λ] = np.load(f"data/v_{λ}.npy")
               e_dict[λ] = np.load(f"data/ems_{λ}.npy")

   .. code:: ipython3

       def simulation(y1_grid, y2_grid, e_grid, λ, θ=1.86/1000., y1_0=1.1, y2_0=1.86/1000, T=100):
           e_fun = interp2d(y1_grid, y2_grid, e_grid.T)
           Et = np.zeros(T+1)
           y1t = np.zeros(T+1)
           y2t = np.zeros(T+1)
           for i in range(T+1):
       #         y2_0 = max(y2_0, 0)
       #         y2_0 = min(y2_0, 0.05)
               Et[i] = e_fun(y1_0, y2_0)
               y1t[i] = y1_0
               y2t[i] = y2_0
               y2_0 = np.exp(-λ)*y2_0 + (1 - np.exp(-λ))*θ*Et[i] 
       #         y2_0 = max(y2_0, 0)
               y1_0 = y1_0 + y2_0
           return Et, y1t, y2t

   .. code:: ipython3

       simul_dict = {}
       for λ_i in λ_list:
           v_temp = v_dict[λ_i]
           e_temp = e_dict[λ_i]
           et_prejump, y1t_prejump, y2t_prejump = simulation(y1_grid[:len(e_temp)], y2_grid, e_temp, λ_i,
                                                         θ=np.mean(θ_list),
                                                         y1_0 = 1.1,
                                                         y2_0=np.mean(θ_list),
                                                         T=110
                                                        )
           simul = {
           "et": et_prejump,
           "y1t": y1t_prejump,
           "y2t": y2t_prejump,
           }
           simul_dict[λ_i] = simul



   .. code:: ipython3

       # plot, emission, y1, y2, button, λ = 1, 2, 5
       plot_2S_ey1y2_multi_λ(simul_dict, λ_list)



.. raw:: html

   <iframe height="500px", width="100%" src="./_static/appC4.html" frameBorder="0"></iframe>





With the same set of initial values, simulate emission trajectories conditioning on no jump happening with different values of :math:`\lambda`.
As :math:`\lambda` increases, the emission trajectories are moving towards the one state trajectoy.

.. toggle::

   .. code:: ipython3

       #  emission, λ = 1,2,5 and 1 state result
       plot_1S_vs_2S_ems_multi_λ(et_1state, simul_dict, λ_list)




.. raw:: html
   
   <iframe height="500px" width="100%" src="./_static/appC5.html"i, frameBorder="0"></iframe>




Use the above emission trajectories to produce the following trajectories of Social Cost of Carbon (SCC) conditioning on no jump happening with the set of :math:`\lambda` s.

.. toggle::

   .. code:: ipython3

       scc_list=np.zeros((3, 111))
       for i in range(len(λ_list)):
           scc_list[i] = η*(α - invkap)*np.exp(Kt)/(1-η)/simul_dict[λ_list[i]]["et"]*1000
    
       # scc λ = 1, 2, 5 and 1 state
       plot_1S_vs_2S_SCC_multi_λ(et_1state, scc_list, scc_1, λ_list)





.. raw:: html

   <iframe src="./_static/appC6.html" height="500px" width="100%" frameBorder="0"></iframe>

.. raw:: html

   <iframe src="./_static/appC7.html" height="500px" width="100%" frameBorder="0"></iframe>