.. admonition:: \   

    This notebook can be accessed through mybinder.org. Click to load the project: |binder|

.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/lphansen/ClimateUncertaintySpillover.git/macroAnnual_v2?filepath=sec6_IllustrativeEconIII.ipynb
7 Illustrative economy IV: tail-end damages
===========================================

In this section, we consider a model that damage function would only
jump to a steep damage specifications.

Different from the model in `section
4 <sec4_IllustrativeEconIa.ipynb>`__ and `section
5 <sec5_IllustrativeEconII.ipynb>`__, we assume here that there is no
uncertainty about the jump threshold and jump intensity is highly
concentrated in the neighborhood of threshold. Here, we explore emission
trajectories when damage function jumps at :math:`1.5 ^o C` and
:math:`2 ^o C`.

Following the same method, we first solve the post jump HJB with a large
value of :math:`\gamma_3`:

.. math::


   \begin{align*}
   0 = \max_{\tilde e} \min_{\omega_\ell} \min_{h} 
   &- \delta \phi(y)    +  \eta \log \tilde e   + \frac{\xi_r}{2} h^2\cr 
   & + \frac {d \phi(y)}{d y} ( \sum_{\ell = 1}^L \omega_\ell \theta_\ell {\tilde e} + \varsigma \tilde{e} h )  + {\frac 1 2} \frac {d^2 \phi(y)}{(dy)^2} |\varsigma|^2 \tilde e^2  \cr
   &+ \frac{(\eta - 1)}{\delta} \left(  \left[ \gamma_1 + \gamma_2 y + \gamma_3 (y - {\bar y})  \right]  ( \sum_{\ell = 1}^L \omega_\ell \theta_\ell {\tilde e} + \varsigma \tilde{e} h) + 
   + {\frac 1 2}(\gamma_2 + \gamma_3)   \mid\varsigma\mid^2 \tilde e^2  \right)
   \end{align*} 

Given the post jump value function :math:`\phi(y)`, we solve for
pre-jump value function :math:`\Phi(y)`:

.. math::

    
   \begin{align*} 
    0 = \max_{\tilde e} \min_{\omega_\ell} \min_{h}\min_{g} & - \delta \Phi(y) + \eta \log \tilde e \cr & + \frac {d \Phi(y)}{d y} \sum_{i=1}^n \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} \frac {d^2 \Phi(y)}{(dy)^2} \mid\varsigma\mid^2 \tilde e^2 \cr &+ \frac{(\eta - 1)}{\delta} \left[ \left( \gamma_1 + \gamma_2 y \right) \sum_{\ell = 1}^L \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} \gamma_2 \mid\varsigma\mid^2 \tilde e^2 \right]\cr 
    & + {\mathcal I}(y) \left( \phi - \Phi \right) \cr
    & + \xi_r \mathcal{J}(y) (1 - g + g \log(g))
    \end{align*}

The first order condition for :math:`g`:

.. math::


       g = \exp \left(\frac{1}{\xi_r} (\Phi - \phi)\right)

As intensity :math:`\mathcal{J}(y)` is highly concentrated in the
neighborhood of :math:`\bar{y}`. The pre-jump HJB is equivalent to:

.. math::

    
   \begin{align*} 
    0 = \max_{\tilde e} \min_{\omega_\ell} \min_{h} & - \delta \Phi(y) + \eta \log \tilde e \cr 
    & + \frac {d \Phi(y)}{d y} \sum_{\ell = 1}^L \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} \frac {d^2 \Phi(y)}{(dy)^2} \mid\varsigma\mid^2 \tilde e^2 \cr 
    & + \frac{(\eta - 1)}{\delta} \left[ \left( \gamma_1 + \gamma_2 y \right) \sum_{\ell = 1}^L \omega_\ell \theta_\ell {\tilde e} + {\frac 1 2} \gamma_2 \mid\varsigma\mid^2 \tilde e^2 \right]\\
    s.t. & \quad \Phi(\bar{y}) = \phi(\bar{y})
   \end{align*}

7.1 Steepest damage function, :math:`\gamma_3 = \frac{1}{3}`
------------------------------------------------------------

.. toggle ::

   .. code:: ipython3

       # packages
       import numpy as np
       import pandas as pd
       from src.model import solve_hjb_y, solve_hjb_y_jump_old
       from src.utilities import find_nearest_value
       from src.simulation import simulate_jump_2
       from src.plots import plot_basic_ems, plot_basic_y, plot_basic_DMG
       import plotly.graph_objects as go
       import plotly.offline as pyo
       pyo.init_notebook_mode()
    
       # Uncertainty parameters
       ξ_w = 1.
       ξ_a = 1/100
       ξ_p = 1.
    
       # Preference
       η = .032
       δ = .01
    
       # Climate sensitivity
       θ_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
       πc_o = np.ones_like(θ_list)/len(θ_list)
    
       # Damage function
       σ_y = 1.2*np.mean(θ_list)
       y_bar_high = 2.0
       y_bar_low = 1.5
       γ_1 = 1.7675/10000
       γ_2 = .0022*2
       γ_2p = np.array([1./3])
       πd_o = np.array([1])
    
       y_step = .01
       y_grid_long = np.arange(0., 4., y_step)
       y_grid_short_high = np.arange(0., y_bar_high+y_step, y_step)
       y_grid_short_low  = np.arange(0., y_bar_low+y_step, y_step)
       n_bar_high = find_nearest_value(y_grid_long, y_bar_high) + 1
       n_bar_low = find_nearest_value(y_grid_long, y_bar_low) + 1






Recall that :math:`\log N_t` is expressed as:

.. math::


   \log N_t = \gamma_1 Y_t + \frac{\gamma_2}{2} Y_t ^2 + \frac{\gamma_3}{2} (Y_t - \bar{y})^2 \mathbb{I}\{Y_t > \bar{y}\}

For different jump threshold :math:`\bar{y}`, :math:`1.5 ^o C` and
:math:`2 ^o C`, the damage function with :math:`\gamma_3 = \frac{1}{3}`
are as follows:

.. code:: ipython3

    def DamageFunc(y, y_bar, γ_3=1/3):
        logN = γ_1 * y**2 + γ_2/2 * y**2 + γ_3/2 * (y - y_bar)**2 * (y > y_bar)
        return np.exp(-logN)


.. toggle ::

   .. code:: ipython3

       fig = go.Figure()
       fig.add_trace(go.Scatter(x=y_grid_long, y=DamageFunc(y_grid_long, 1.5), 
                                name="Threshold 1.5", line=dict(color="blue"),
                                hovertemplate="Temperature anomaly: %{x} <br>damage: %{y:.2f}</br>"
                               ) )
       fig.add_trace(go.Scatter(x=y_grid_long, y=DamageFunc(y_grid_long, 2.0), 
                                name="Threshold 2.0", line=dict(color="red"),
                                hovertemplate="Temperature anomaly: %{x} <br>damage: %{y:.2f}</br>"
                               ) )
       fig.update_xaxes(range=[0,3],title="Temperature anomaly", showline=True, showspikes=True)
       fig.update_yaxes(range=[0.4, 1.05],title="Proprotional reduction to economic output", showspikes=True)



.. raw:: html

    <iframe src="./_static/Figure7-1.html" frameBorder="0" height="600px" width="100%"></iframe>


.. toggle ::

   .. code:: ipython3

       # Prepare ϕ conditional on γ_3 = 1/3
       model_res_list_high = []
       model_res_list_low  = []
       for γ_2p_i in γ_2p:
           model_args_high = (η, δ, σ_y, y_bar_high, γ_1, γ_2, γ_2p_i, θ_list, πc_o, ξ_w, ξ_a) 
           model_args_low  = (η, δ, σ_y, y_bar_low,  γ_1, γ_2, γ_2p_i, θ_list, πc_o, ξ_w, ξ_a) 
           model_res_high  = solve_hjb_y(y_grid_long, model_args_high, v0=None, ϵ=1.,
                                   tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_low   = solve_hjb_y(y_grid_long, model_args_low, v0=None, ϵ=1.,
                                   tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_list_high.append(model_res_high)
           model_res_list_low.append(model_res_low)
        
       ϕ_list_high = [res['v'] for res in model_res_list_high]
       ϕ_list_low  = [res['v'] for res in model_res_list_low]
    
       # solve for pre-jump HJB
       ξ_p_list = [1]
       solution_high = dict()
       solution_low  = dict()
    
       for ξ_p_i in ξ_p_list:
           certainty_equivalent_high = -ξ_p_i*np.log(np.average(np.exp(-1./ξ_p_i*np.array(ϕ_list_high)), axis=0, weights=πd_o))
           certainty_equivalent_low  = -ξ_p_i*np.log(np.average(np.exp(-1./ξ_p_i*np.array(ϕ_list_low)), axis=0, weights=πd_o))
           # Change grid from 0-4 to 0-2
           ϕ_i_high = np.array([temp[:n_bar_high] for temp in ϕ_list_high])
           ϕ_i_low  = np.array([temp[:n_bar_low] for temp in ϕ_list_low])
           # Compute ϕ with jump (impose boundary condition)
           model_args_high = (η, δ, σ_y, y_bar_high, γ_1, γ_2, γ_2p, θ_list, πc_o, ϕ_i_high, πd_o, ξ_w, ξ_p_i, ξ_a)
           model_args_low  = (η, δ, σ_y, y_bar_low, γ_1, γ_2, γ_2p, θ_list, πc_o, ϕ_i_low, πd_o, ξ_w, ξ_p_i, ξ_a)
        
           model_res_high_pre = solve_hjb_y_jump_old(y_grid_short_high, model_args_high, 
                                    v0=np.average(ϕ_i_high, weights=πd_o, axis=0),
                                    ϵ=1., tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_low_pre  = solve_hjb_y_jump_old(y_grid_short_low, model_args_low, 
                                    v0=np.average(ϕ_i_low, weights=πd_o, axis=0),
                                    ϵ=1., tol=1e-8, max_iter=5_000, print_iteration=False)
           simulation_res_high = simulate_jump_2(model_res_high_pre, model_res_high, y_bar_high, θ_list, ME=None,  y_start=1.1,  T=300, dt=1)
           simulation_res_low  = simulate_jump_2(model_res_low_pre,  model_res_low,  y_bar_low,  θ_list, ME=None,  y_start=1.1,  T=300, dt=1)
        
           solution_high[ξ_p_i] = dict(
               model_res_high=model_res_high, 
               simulation_res_high=simulation_res_high, 
               certainty_equivalent_high=certainty_equivalent_high
           )
           solution_high[ξ_p_i] = dict(
               model_res_low=model_res_low, 
               simulation_res_low=simulation_res_low, 
               certainty_equivalent_low=certainty_equivalent_low
           )


   .. parsed-literal::

       Converged. Total iteration: 5000;	 LHS Error: 7.356373393230253e-05;	 RHS Error 0.0007519883367738643
       Converged. Total iteration: 5000;	 LHS Error: 3.719900538468046e-05;	 RHS Error 0.0007334349625613412


The following plots show emission trajectories, temperature anomaly
trajectories and damage trajectories for different carbon budgets.

The plots show that for threshold 1.5, jump happens in 87 years, and for
2.0 jump happens in 154 years.

Emission trajectories
~~~~~~~~~~~~~~~~~~~~~

.. toggle ::

   .. code:: ipython3

       fig = plot_basic_ems(simulation_res_high, simulation_res_low, 299)
       fig.add_vline(x=87, line_width=2, line_dash="dash", line_color="blue")
       fig.add_vline(x=154, line_width=2, line_dash="dash", line_color="red")
       fig



.. raw:: html

    <iframe src="./_static/Figure7-2.html" frameBorder="0" height="600px" width="100%"></iframe>

Temperature anomalies
~~~~~~~~~~~~~~~~~~~~~

The temperature anomaly trajectories for threshold :math:`1.5` and
:math:`2` are displayed below.

.. toggle ::

   .. code:: ipython3

       fig = plot_basic_y(simulation_res_high, simulation_res_low, 299)
       fig.add_shape(type="line",
           x0=87, y0=1, x1=87, y1=1.5,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=1.5, x1=87, y1=1.5,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=153, y0=1, x1=153, y1=2.0,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=2.0, x1=153, y1=2.0,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )



.. raw:: html

    <iframe src="./_static/Figure7-3.html" frameBorder="0" height="600px" width="100%"></iframe>

Damages
~~~~~~~

The following plot shows the corresponding damage as proportional
reduction in economic output.

.. toggle ::

   .. code:: ipython3

       fig = plot_basic_DMG(simulation_res_high, simulation_res_low, 299,
                      y_bar_high, y_bar_low, γ_1, γ_2, γ_2p)
       fig.add_shape(type="line",
           x0=87, y0=0.96, x1=87, y1=0.9947,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=0.9947, x1=87, y1=0.9947,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=153, y0=0.96, x1=153, y1=0.99,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=0.9908, x1=153, y1=0.9908,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )



.. raw:: html

    <iframe src="./_static/Figure7-4.html" frameBorder="0" height="600px" width="100%"></iframe>


7.2 Steeper damage function, :math:`\gamma_3 = \frac{2}{3}`
-----------------------------------------------------------

We now explore with a steeper damage function with
:math:`\gamma_3 = \frac{2}{3}`. For the same two choice of
:math:`\bar{y}`, the damage functions with more curvature are as
follows:

.. toggle ::
    
   .. code:: ipython3

       fig = go.Figure()
       fig.add_trace(go.Scatter(x=y_grid_long, y=DamageFunc(y_grid_long, 1.5, γ_3=2./3), name="Threshold 1.5", 
                                line=dict(color="blue"),
                                hovertemplate="Temperature anomaly: %{x} <br>Damage: %{y:.4f}</br>"
                               ) )
       fig.add_trace(go.Scatter(x=y_grid_long, y=DamageFunc(y_grid_long, 2.0, γ_3=2./3), name="Threshold 2.0", 
                                line=dict(color="red"),
                                hovertemplate="Temperature anomaly: %{x} <br>Damage: %{y:.4f}</br>"
                               ) )
       fig.update_xaxes(range=[0, 3], title="Temperature anomaly", showline=True, showspikes=True)
       fig.update_yaxes(range=[0.4, 1.05], title="Proprotional reduction to economic output", showspikes=True)



.. raw:: html

    <iframe src="./_static/Figure7-5.html" height="600px" width="100%" frameBorder="0"></iframe>


.. toggle ::

   .. code:: ipython3

       # Prepare ϕ conditional on γ_3 = 2/3
       γ_2p = np.array([2./3])
       model_res_list_high = []
       model_res_list_low  = []
       for γ_2p_i in γ_2p:
           model_args_high = (η, δ, σ_y, y_bar_high, γ_1, γ_2, γ_2p_i, θ_list, πc_o, ξ_w, ξ_a) 
           model_args_low  = (η, δ, σ_y, y_bar_low,  γ_1, γ_2, γ_2p_i, θ_list, πc_o, ξ_w, ξ_a) 
           model_res_high  = solve_hjb_y(y_grid_long, model_args_high, v0=None, ϵ=1.,
                                   tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_low   = solve_hjb_y(y_grid_long, model_args_low, v0=None, ϵ=1.,
                                   tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_list_high.append(model_res_high)
           model_res_list_low.append(model_res_low)
        
       ϕ_list_high = [res['v'] for res in model_res_list_high]
       ϕ_list_low  = [res['v'] for res in model_res_list_low]
    
       # solve for 
       ξ_p_list = [1]
       solution_high = dict()
       solution_low  = dict()
    
       for ξ_p_i in ξ_p_list:
           certainty_equivalent_high = -ξ_p_i*np.log(np.average(np.exp(-1./ξ_p_i*np.array(ϕ_list_high)), axis=0, weights=πd_o))
           certainty_equivalent_low  = -ξ_p_i*np.log(np.average(np.exp(-1./ξ_p_i*np.array(ϕ_list_low)), axis=0, weights=πd_o))
           # Change grid from 0-4 to 0-2
           ϕ_i_high = np.array([temp[:n_bar_high] for temp in ϕ_list_high])
           ϕ_i_low  = np.array([temp[:n_bar_low] for temp in ϕ_list_low])
           # Compute ϕ with jump (impose boundary condition)
           model_args_high = (η, δ, σ_y, y_bar_high, γ_1, γ_2, γ_2p, θ_list, πc_o, ϕ_i_high, πd_o, ξ_w, ξ_p_i, ξ_a)
           model_args_low  = (η, δ, σ_y, y_bar_low, γ_1, γ_2, γ_2p, θ_list, πc_o, ϕ_i_low, πd_o, ξ_w, ξ_p_i, ξ_a)
        
           model_res_high_pre = solve_hjb_y_jump_old(y_grid_short_high, model_args_high, 
                                    v0=np.average(ϕ_i_high, weights=πd_o, axis=0),
                                    ϵ=0.5, tol=1e-8, max_iter=5_000, print_iteration=False)
           model_res_low_pre  = solve_hjb_y_jump_old(y_grid_short_low, model_args_low, 
                                    v0=np.average(ϕ_i_low, weights=πd_o, axis=0),
                                    ϵ=0.5, tol=1e-8, max_iter=5_000, print_iteration=False)
           simulation_res_high = simulate_jump_2(model_res_high_pre, model_res_high, y_bar_high, θ_list, ME=None,  y_start=1.1,  T=300, dt=1)
           simulation_res_low  = simulate_jump_2(model_res_low_pre,  model_res_low,  y_bar_low,  θ_list, ME=None,  y_start=1.1,  T=300, dt=1)
        
           solution_high[ξ_p_i] = dict(
               model_res_high=model_res_high, 
               simulation_res_high=simulation_res_high, 
               certainty_equivalent_high=certainty_equivalent_high
           )
           solution_high[ξ_p_i] = dict(
               model_res_low=model_res_low, 
               simulation_res_low=simulation_res_low, 
               certainty_equivalent_low=certainty_equivalent_low
           )


   .. parsed-literal::

       Converged. Total iteration: 660;	 LHS Error: 9.829381752979316e-09;	 RHS Error 0.0012357112869160539
       Converged. Total iteration: 588;	 LHS Error: 9.630465314103276e-09;	 RHS Error 0.0010696696972092197


With steeper damage functions, jump happens later compared to the damage
function with the same threshold. For threshold of 1.5, jump happens in
108 years, and for threshold of 2 jump happens in 182 years.

Emission trajectories
~~~~~~~~~~~~~~~~~~~~~

Similarly, the emission trajectories with steeper damage function.

.. toggle ::

   .. code:: ipython3

       fig = plot_basic_ems(simulation_res_high, simulation_res_low, 299)
       fig.add_vline(x=108, line_width=2, line_dash="dash", line_color="blue")
       fig.add_vline(x=182, line_width=2, line_dash="dash", line_color="red")



.. raw:: html

    <iframe src="./_static/Figure7-6.html" height="600px" width="100%" frameBorder="0"></iframe>

Temperature anomalies
~~~~~~~~~~~~~~~~~~~~~

.. toggle ::

   .. code:: ipython3

       fig = plot_basic_y(simulation_res_high, simulation_res_low, 299)
       fig.add_shape(type="line",
           x0=108, y0=1, x1=108, y1=1.5,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=1.5, x1=108, y1=1.5,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=182, y0=1, x1=182, y1=2.0,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=2.0, x1=182, y1=2.0,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )



.. raw:: html

    <iframe src="./_static/Figure7-7.html" frameBorder="0" height="600px" width="100%"></iframe>

Damages
~~~~~~~
.. toggle ::

   .. code:: ipython3

       fig = plot_basic_DMG(simulation_res_high, simulation_res_low, 299,
                      y_bar_high, y_bar_low, γ_1, γ_2, γ_2p)
       fig.add_shape(type="line",
           x0=108, y0=0.96, x1=108, y1=0.9947,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=0.9947, x1=108, y1=0.9947,
           line=dict(
               color="blue",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=182, y0=0.96, x1=182, y1=0.99,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )
       fig.add_shape(type="line",
           x0=0, y0=0.9908, x1=182, y1=0.9908,
           line=dict(
               color="red",
               width=2,
               dash="dash",
           )
       )



.. raw:: html

    <iframe src="./_static/Figure7-8.html" frameBorder="0" height="600px" width="100%"></iframe>