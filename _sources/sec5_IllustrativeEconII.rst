
.. admonition:: \   

    This notebook presents compuations in section 8 of the paper.
    This notebook can be accessed through mybinder.org. Click to load the project: |binder|

.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/lphansen/ClimateUncertaintySpillover.git/macroAnnual_v2?filepath=sec5_IllustrativeEconII.ipynb

5 Illustrative economy II: uncertainty decomposition
====================================================

This notebook presents compuations in section 8 of the paper.

| An advantage to the more structured approach implemented as smooth
  ambiguity is that it allows us to “open the hood” so to speak on
  uncertainty. We build on the work of `Ricke and Caldeira
  (2014) <#RickeCaldeira:2014>`__ by exploring the relative
  contributions of uncertainty in the carbon dynamics versus uncertainty
  in the temperature dynamics. We depart from their analysis by studying
  the relative contributions in the context of a decision problem and we
  include robustness to model misspecification as a third source of
  uncertainty. This latter adjustment applies primarily to the damage
  function specification.
| We continue to use the social cost of carbon as a benchmark for
  assessing these contributions. We perform these computations using the
  model developed in the previous section, although the approach we
  describe is applicable more generally.
| For the uncertainty decomposition, we hold fixed the control law for
  emissions, and hence also the implied state evolution for damages, and
  explore the consequences of imposing constraints on minimization over
  the probabilities across the different models.

| Recall that we use climate sensitivity parameters from combinations of
  16 models of temperature dynamics and 9 models of carbon dynamics.
| A parameter :math:`\theta` corresponds to climate-temperature model
  pair. Let :math:`\Theta` denote the full set of :math:`L = 144` pairs,
  and let :math:`P_{j}` for :math:`j = 1,2,... J` be a partition of the
  positive integers up to :math:`L`. The integer :math:`J` is set to 9
  or 16 depending on whether we target the temperature models or the
  carbon.
| For any given such partition, we solve a constrained version of the
  minimization problem in `section 4 <sec4_IllustrativeEconIa.ipynb>`__
  by targeting the probabilities assigned to partitions while imposing
  the benchmark probabilities conditioned on each partition.

.. math::


   \begin{align*}
   \min_{{\overline \omega}_j, j=1,2,..., J} &
   \left(\frac {\partial  V}{\partial  x }\right) \cdot
   \sum_{j=1}^J {\overline \omega}_j \sum_{\ell \in P_j}  \left( {\frac {
   \pi_\ell}  {\sum_{\ell \in P_j} \pi_\ell}} \right) \mu(x, a \mid \theta_\ell) \cr
   &  + \xi_a \sum_{j=1}^J {\overline \omega}_j \left(\log {\overline \omega}_j - \log {\overline \pi}_j\right)
   \end{align*}

where: :math:`{\overline \pi}_j = {\sum_{\ell \in P_j} \pi_\ell}` and

.. math::


   \frac {\pi_\ell}  {{\overline \pi}_\ell }  \hspace{.5cm} \ell \in P_j

are the baseline conditional probabilities for partition :math:`j`. We
only minimize the probabilities across partitions while imposing the
baseline conditional probabilities within a partition.

| We impose :math:`\xi_r = \infty` when performing this minimization and
  let :math:`\xi_a = .01` as in `section
  4 <sec4_IllustrativeEconIa.ipynb>`__.
| We perform additional calculations where we let :math:`\xi_r=1` and
  :math:`\xi_a = \infty` in order to target damage function uncertainty
  rather than temperature or climate dynamics uncertainty[^1]. The two
  states in our problem are :math:`x = (y,n)`, and we look for a value
  function of the form
  :math:`V(y,n) = \phi(y) + \frac{(\eta - 1)}{\delta} n` while imposing
  that :math:`{\tilde e} = \epsilon(y)`. For each partition of interest,
  we construct the corresponding HJB equation that supports this
  minimization.

Since we are imposing the control law for emissions but constraining the
minimization, the first-order conditions for emissions will no longer be
satisfied. Recall formula for :math:`\log SCC` from `section
4 <sec4_IllustrativeEconIa.ipynb>`__ with adjustments for uncertainty.
In the absence of optimality, the net benefit measure :math:`MV(x)` is
not zero with the minimization constraints imposed. Consistent with the
SCC computation from the previous section, we use

.. math::


   \begin{align*} 
    - \frac {\partial V}{\partial x} (x) \cdot {\frac {\partial \mu}{\partial e}} \left[x, \phi(x) \right]  -  {\frac 1 2}  {\rm trace} \left[  \frac {\partial^2 V}{\partial x \partial x'} (x) \frac \partial  {\partial e} \Sigma \left[x, \phi(x)  \right] \right].
   \end{align*}

for our cost contributions in the SCC decomposition.

| We obtain the smallest cost measure when we preclude minimization
  altogether while solving for the value function and the largest one
  when we allow for full minimization with :math:`\xi_r = 1` and
  :math:`\xi_a = .01.` We have three intermediate cases corresponding to
  temperature dynamic uncertainty, climate dynamic uncertainty and
  damage function uncertainty. The smallest of these measures
  corresponds to a full commitment to the baseline probabilities.
| We form ratios with respect to the smallest measure, take logarithms
  and multiply by 100 to convert the numbers to percentages.
  Importantly, we change both probabilities and value functions in this
  computation.

We report the results in the figure below. From this figure, we see that
the uncertainty adjustments in valuation account for twenty to thirty
percent of the social cost of carbon. The contributions from temperature
and carbon are essentially constant over time with the temperature
uncertainty contribution being substantially larger. The damage
contribution is initially below half the total uncertainty, but this
changes to more than half by the time the temperature anomaly reaches
the lower threshold of 1.5 degrees Celsius.

For our uncertainty decomposition, we compute the logarithm of this
expression for alternative partitions of the models. We start by
activating separately uncertainty aversion over



.. raw:: html

   <ol style="list-style-type:lower-roman">

.. raw:: html

   <li>

models of carbon dynamics,

.. raw:: html

   </li>

.. raw:: html

   <li>

the models of temperature dynamics, and

.. raw:: html

   </li>

.. raw:: html

   <li>

the models or economic damages

.. raw:: html

   </li>

.. raw:: html

   </ol>

In each case we report the difference in logarithms between the
computation using the baseline probabilities and the solutions from the
constrained probability minimizations. Importantly, we change both
probabilities and value functions in this computation.

.. toggle::

        .. code:: ipython3

            # packages
            import pandas as pd
            import numpy as np
            from scipy import interpolate
            from src.model import solve_hjb_y, solve_hjb_y_jump, solve_baseline, minimize_g, minimize_π
            from src.utilities import find_nearest_value, solve_post_jump
            from src.simulation import simulate_me
            import plotly.io as pio
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            import pickle
            pyo.init_notebook_mode()




        .. code:: ipython3

            # preparation
            ξ_w = 100_000
            ξ_a = 0.01
            ξ_r = 1.
            
            ϵ = 5.
            η = .032
            δ = .01
            
            θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
            πc_o = np.ones_like(θ_list)/len(θ_list)
            
            σ_y = 1.2*np.mean(θ_list)
            y_underline = 1.5
            y_bar = 2.
            γ_1 = 1.7675/10000
            γ_2 = .0022*2
            γ_3 = np.linspace(0., 1./3, 20)
            πd_o = np.ones_like(γ_3)/len(γ_3)
            
            y_step = .01
            y_grid_long = np.arange(0., 5., y_step)
            y_grid_short = np.arange(0., y_bar+y_step, y_step)
            n_bar = find_nearest_value(y_grid_long, y_bar) 
            
            # Uncertainty decomposition
            n_temp = 16
            n_carb = 9
            θ_reshape = θ_list.reshape(n_temp, n_carb)
            θ_temp = np.mean(θ_reshape, axis=1)
            θ_carb = np.mean(θ_reshape, axis=0)
            πc_o_temp = np.ones_like(θ_temp)/len(θ_temp)
            πc_o_carb = np.ones_like(θ_carb)/len(θ_carb)

The results are reported below. For comparison we include the analogous
computation when we activate an aversion to all three sources of
uncertainty.

With the penalties, :math:`\xi_r = 1` and :math:`\xi_a = 0.01`, the
contributions from temperature are essentially constant with the
temperature uncertainty contribution being substantially larger. The
damage contribution is initially well below half the total uncertainty,
but this changes to be about a half after sixty years when temperature
anomaly reaches the lower bound of jump threshold, :math:`1.5`. It is
important to remember that these computations are performed while
imposing the planner’s solution for emissions and damages. So called
“business-as-usual” simulations would change substantially this
uncertainty accounting.

Since the uncertainty components are not “additive,” we explore the
joint impacts by partitioning the uncertainty using the three different
pairings of contributions. The results are reported in Figure 13(b). Not
surprisingly, the combination of temperature and damage uncertainty has
the biggest impact accounting for about three-fourths of the total
uncertainty. In contrast, the combination of temperature and carbon
uncertainty accounts for somewhere between one-half and one-third of the
total uncertainty depending on how many years in the future we look at.

The quantitative importance of damages will increase as we reduce
:math:`\xi_r`. We see the :math:`\xi_r` setting as dictating how much
wiggle room a decision maker wants to entertain for the weighting of the
alternative damage model specifications. With this change, minimizing
probabilities are shifted almost entirely to the “extreme damage”
specification, given us effectly an upper bound on the uncertainty
contribution to the social cost of carbon. Now the overall uncertainty
contribution ranges from thirty to sixty percent as shown in Figure
13(a) with :math:`\xi_r = 1` and :math:`\xi_a = 0.01`. The damage
uncertainty contribution alone accounts for more than half of this where
as the temperature and climate contributions remain about the same as
before. Temperature and damage uncertainty taken together account for
most of the uncertainty as reflected in Figure 13(b).


.. code:: ipython3

    def simulate_ratio(y_grid, e_grid, ratio, y0=1.1, T=100, dt=1):
        et = np.zeros(T+1)
        yt = np.zeros(T+1)
        ratio_t = np.zeros((len(ratio), T+1))
        ratio_func = interpolate.interp1d(y_grid, ratio)
        e_func = interpolate.interp1d(y_grid, e_grid)
        for t in range(T):
            et[t] = e_func(y0)
            ratio_t[:, t] = ratio_func(y0)
            yt[t] = y0
            y0 = y0 + np.mean(θ_list) * et[t] * dt
        return yt, ratio_t

.. toggle::

        .. code:: ipython3

            ξ_a = 0.01
            ξ_r = 1.
            # load solutions
            pre_jump_res = pickle.load(open(f"data/pre_jump_res_{ξ_a}.pickle", "rb"))
            v_list = pickle.load(open(f"data/v_list_{ξ_a}.pickle", "rb"))
            e_tilde_list = pickle.load(open(f"data/e_tilde_list_{ξ_a}.pickle", "rb"))
            # compute total ME 
            ems_star = pre_jump_res[1]["model_res"]['e_tilde']
            ME_total = η / ems_star
            
            # baseline
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, 100_000, 100_000, σ_y, y_underline)
            ME_base, ratio_base = solve_baseline(y_grid_long,
                                                 n_bar,
                                                 ems_star[:n_bar + 1],
                                                 v_list[100_000], 
                                                 args,
                                                 ϵ=1.,
                                                 tol=1e-8,
                                                 max_iter=500)
            
            # carbon
            print("--------------Carbon-----------------")
            args_list_carb = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_carb, πc_o_carb, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_carb.append(args_iter)
            
            ϕ_list_carb, ems_list_carb = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_carb)
            args = (δ, η, θ_carb, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_carb, ratiocarb = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_carb, args)
            
            # temperature
            print("-------------Temperature--------------")
            args_list_temp = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_temp, πc_o_temp, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_temp.append(args_iter)
            
            ϕ_list_temp, ems_list_temp = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_temp)
            args = (δ, η, θ_temp, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_temp, ratiotemp = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_temp, args)
            
            # damage
            print("-------------------Damage-----------------")
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 1, 100_000, 100_000, σ_y, y_underline)
            ME_dmg, ratiotemp = minimize_g(y_grid_long, n_bar, ems_star[:n_bar + 1], v_list[100_000], args)
            
            
            # two type partition
            # carbon and damage
            print("----------------Carbon and damage---------------")
            args_list_carbdmg = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_carb, πc_o_carb, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 0.5, 1e-8, 5_000, False)
                args_list_carbdmg.append(args_iter)
            ϕ_list_carbdmg, ems_list_carbdmg = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_carbdmg)
            args = (δ, η, θ_carb, γ_1, γ_2, γ_3, y_bar, πd_o, 1., ξ_a, 100_000, σ_y, y_underline)
            ME_carbdmg, ratiocarbdmg = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_carbdmg, args, True)
            
            # temperature and damage
            print("----------------Temperature and damage--------------")
            args_list_tempdmg = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_temp, πc_o_temp, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 0.5, 1e-8, 5_000, False)
                args_list_tempdmg.append(args_iter)
            ϕ_list_tempdmg, ems_list_tempdmg = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_tempdmg)
            args = (δ, η, θ_temp, γ_1, γ_2, γ_3, y_bar, πd_o, 1., ξ_a, 100_000, σ_y, y_underline)
            ME_tempdmg, ratiotempdmg = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_tempdmg, args, True)
            
            # temperature and carbon
            print("----------------Temperature and carbon-----------------")
            args_list_tempcarb = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_list, πc_o, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_tempcarb.append(args_iter)
            ϕ_list_tempcarb, ems_list_tempcarb = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_tempcarb)
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_tempcarb, ratiotempcarb = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_tempcarb, args)
            
            # ratio list
            ratios = np.array([
                np.log(ME_total[:len(ME_base)]  / ME_base ) * 100,
                np.log(ME_dmg  / ME_base ) * 100,
                np.log(ME_temp  / ME_base ) * 100,
                np.log(ME_carb  / ME_base ) * 100,
                np.log(ME_carbdmg  / ME_base ) * 100,
                np.log(ME_tempdmg  / ME_base ) * 100,
                np.log(ME_tempcarb  / ME_base ) * 100,
            ])
            
            # simulate for temperature anomaly and ratios
            yt, ratios_t = simulate_ratio(y_grid_short, ems_star[:len(y_grid_short)], ratios)

Here, we repeat the computation for a different ambiguity aversion
parameter, :math:`\xi_a = 0.005` and compare it with the decompositions
with :math:`\xi_a = 0.01` (click to see code detail).

.. toggle::

        .. code:: ipython3

            ξ_a = 0.005
            pre_jump_res = pickle.load(open(f"data/pre_jump_res_{ξ_a}.pickle", "rb"))
            v_list = pickle.load(open(f"data/v_list_{ξ_a}.pickle", "rb"))
            e_tilde_list = pickle.load(open(f"data/e_tilde_list_{ξ_a}.pickle", "rb"))
            ems_star = pre_jump_res[1]["model_res"]['e_tilde']
            ME_total = η / ems_star
            # perform uncertainty decomposition
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, 100_000, 100_000, σ_y, y_underline)
            ME_base, ratio_base = solve_baseline(y_grid_long,
                                                 n_bar,
                                                 ems_star[:n_bar + 1],
                                                 v_list[100_000], 
                                                 args,
                                                 ϵ=1.,
                                                 tol=1e-8,
                                                 max_iter=1_000)
            
            # carbon
            print("--------------------Carbon-----------------")
            args_list_carb = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_carb, πc_o_carb, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_carb.append(args_iter)
            
            ϕ_list_carb, ems_list_carb = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_carb)
            args = (δ, η, θ_carb, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_carb, ratiocarb = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_carb, args)
            
            # temperature
            print("---------------------Temperature--------------")
            args_list_temp = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_temp, πc_o_temp, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_temp.append(args_iter)
            
            ϕ_list_temp, ems_list_temp = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_temp)
            args = (δ, η, θ_temp, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_temp, ratiotemp = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_temp, args)
            
            # damage
            print("-------------------Damage-----------------")
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 1, 100_000, 100_000, σ_y, y_underline)
            ME_dmg, ratiotemp = minimize_g(y_grid_long, n_bar, ems_star[:n_bar + 1], v_list[100_000], args, ϵ=0.5)
            
            # two type partition
            # carbon and damage
            print("----------------Carbon and damage---------------")
            args_list_carbdmg = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_carb, πc_o_carb, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 0.5, 1e-8, 5_000, False)
                args_list_carbdmg.append(args_iter)
            ϕ_list_carbdmg, ems_list_carbdmg = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_carbdmg)
            args = (δ, η, θ_carb, γ_1, γ_2, γ_3, y_bar, πd_o, 1., ξ_a, 100_000, σ_y, y_underline)
            ME_carbdmg, ratiocarbdmg = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_carbdmg, args, True)
            
            # temperature and damage
            print("----------------Temperature and damage--------------")
            args_list_tempdmg = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_temp, πc_o_temp, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 0.5, 1e-8, 5_000, False)
                args_list_tempdmg.append(args_iter)
            ϕ_list_tempdmg, ems_list_tempdmg = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_tempdmg)
            args = (δ, η, θ_temp, γ_1, γ_2, γ_3, y_bar, πd_o, 1., ξ_a, 100_000, σ_y, y_underline)
            ME_tempdmg, ratiotempdmg = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_tempdmg, args, True)
            
            # temperature and carbon
            print("----------------Temperature and carbon-----------------")
            args_list_tempcarb = []
            for γ_3_m in γ_3:
                args_func = (η, δ, σ_y, y_bar, γ_1, γ_2, γ_3_m, θ_list, πc_o, 100_000, ξ_a)
                args_iter = (y_grid_long, args_func, None, 1., 1e-8, 5_000, False)
                args_list_tempcarb.append(args_iter)
            ϕ_list_tempcarb, ems_list_tempcarb = solve_post_jump(y_grid_long, γ_3, solve_hjb_y, args_list_tempcarb)
            args = (δ, η, θ_list, γ_1, γ_2, γ_3, y_bar, πd_o, 100_000, ξ_a, 100_000, σ_y, y_underline)
            ME_tempcarb, ratiotempcarb = minimize_π(y_grid_long, n_bar, ems_star[:n_bar + 1], ϕ_list_tempcarb, args)
            
            # list of ratios
            ratios_0p005 = np.array([
                np.log(ME_total[:len(ME_base)]  / ME_base ) * 100,
                np.log(ME_dmg  / ME_base ) * 100,
                np.log(ME_temp  / ME_base ) * 100,
                np.log(ME_carb  / ME_base ) * 100,
                np.log(ME_carbdmg  / ME_base ) * 100,
                np.log(ME_tempdmg  / ME_base ) * 100,
                np.log(ME_tempcarb  / ME_base ) * 100,
            ])
            
            yt_0p005, ratios_t_0p005 = simulate_ratio(y_grid_short, ems_star[:len(y_grid_short)], ratios_0p005)






.. raw :: html

   <iframe frameBorder="0", height="600px", width="100%", src="./_static/fig13a.html"></iframe>



The plot below shows uncertainty decomposition with two sources considered together.

.. raw:: html

   <iframe height="600px" width="100%" src="./_static/fig13b.html" frameBorder="0"></iframe>
