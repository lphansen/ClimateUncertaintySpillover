.. admonition:: \   

    This notebook can be accessed through mybinder.org. Click to load the project: |binder|

.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/lphansen/ClimateUncertaintySpillover.git/macroAnnual_v2?filepath=sec6_IllustrativeEconIII.ipynb

6 Illustrative economy III: carbon abatement technology
=======================================================

While the model posed in section `4 <sec4_IllustrativeEconI.rst>`__
illustrated how the unfolding of damages should alter policy, the
economic model was not designed to confront transitions to fully
carbon-neutral economies. There have been several calls for such
transitions with little regard for the role or impact of uncertainty. We
now modify the model to allow for green technology in decades to come.

We next consider a technology that is close to the Dynamic Integrated
Climate-Economy (DICE) model of :cite:t:`Nordhaus:2017`. See
also :cite:t:`CaiJuddLontzek:2017` and
:cite:t:`CaiLontzek:2019` for a stochastic extension (DSICE)
of the DICE
model. [#]_ 
For our setting, we alter the output equation from our previous
specification as follows:

.. math::


   \frac {I_t}{K_t} + \frac {C_t} {K_t}  + \frac {J_t} {K_t}  = \alpha 

where:

.. math:: :label: output_loss

    \frac {J_t} {K_t} =   \left\{ \begin{matrix}   \alpha {\vartheta_t}  \left[ 1  -  \left({\frac {{\mathcal E}_t} { \alpha \lambda_t K_t}}\right)\right]^\theta & \left({\frac {{\mathcal E}_t} {\alpha  K_t}}\right)  \le \lambda_t \cr
   0 & \left({\frac {{\mathcal E}_t} {\alpha  K_t}}\right)  \ge \lambda_t  \end{matrix} \right.
    

To motivate the term :math:`J_t`, express the emissions to capital ratio
as:

.. math::


   \frac {{\mathcal E}_t} {K_t}  = \alpha \lambda_t  ( 1 - \iota_t ) 

| where :math:`0 \le \iota_t \le 1` is *abatement* at
  date :math:`t`. The exogenously specified process :math:`\lambda`
  gives the emissions to output ratio in the absence of any abatement.
| By investing in :math:`\iota_t`, this ratio can be reduced, but there
  is a corresponding reduction in output. Specifically, the output loss
  is given by:

.. math::


   J_t  = \alpha K_t \vartheta (\iota_t)^\theta 

Equation :math:numref:`output_loss` follows by solving for
abatement :math:`\iota_t` in terms of emissions. [#]_ 


The planner’s preferences are logarithmic over damaged consumption:

.. math::


   \log {\widetilde C}_t = \log C_t  - \log N_t = (\log C_t - \log K_t) - \log N_t + \log K_t .

In contrast to the previous specification, the planner’s value function
for this model is no longer additively separable in :math:`(y,k)`,
although it remains additively separable in log damages, :math:`n`.

6.1 HJB, post damage jump value functions
-------------------------------------

Controls: :math:`(i, e)` where :math:`i` is a potential value for
:math:`\frac {I_t} {K_t}` and :math:`e` is a realization of
:math:`{\mathcal E}_t`. States are :math:`(k, d, y)` where :math:`k` is
a realization of :math:`\log K_t`, :math:`d` is a realization of
:math:`\log D_t`, and :math:`y` is the temperature anomaly. Guess a
value function of the form: :math:`\upsilon_d d + \Phi^m(k,y)`.

.. math::


   \begin{align*} 
   0 = \max_{i,e} \min_{\omega_\ell, \sum_{\ell = 1}^L \omega_\ell = 1 } & - \delta \upsilon_d d  - \delta \Phi^m(k,y) +  \log \left( \alpha - i -  \alpha \overline{\vartheta} \left[ 1 - \left(\frac {e} { \alpha \overline\lambda \exp(k)} \right) \right]^\theta \right) + k - d \cr 
   & + \frac {\partial \Phi^m(k,y)}{\partial k} 
    \left[ \mu_k    + i   -
   {\frac { \kappa} 2} i^2  -  \frac  {|\sigma_k|^2}  2 + \frac {|\sigma_k|^2} 2  \frac {\partial^2 \Phi^m(k,y)}{\partial k^2}\right]  \cr
   & + \frac {\partial  \Phi^m(k,y)}{\partial y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell {e} + {\frac 1 2} \frac {\partial^2 \Phi^m(k,y)}{\partial y^2} \mid \varsigma \mid ^2 e^2  \cr
   & + \upsilon_d \left( \left[ \gamma_1 + \gamma_2 y + \gamma_3^m (y - \overline y) \right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell { e} + {\frac 1 2} (\gamma_2 + 
   \gamma_3^m) \mid \varsigma \mid ^2  e^2 \right) \cr
   & + \xi_a \sum_{\ell = 1}^L \omega_\ell \left( \log \omega_\ell - \log \pi_\ell \right) 
   \end{align*} 

First order condition
~~~~~~~~~~~~~~~~~~~~~

Let

.. math::


   mc \doteq  \frac 1 {\left( \alpha - i -   \alpha \overline{\vartheta} \left[ 1 - \left({\frac {e} { \alpha \overline\lambda \exp(k)}}\right) \right]^\theta  \right)}

First-order conditions for :math:`i`:

.. math::


   - mc + \frac {\partial \Phi^m(k,y)}{\partial k} \left( 1 - \kappa i \right) = 0.  

| Given :math:`mc`, the first-order conditions for :math:`i` are affine.
| First-order conditions for :math:`e`:

.. math::


   \begin{align*} 
    & mc  \left( \frac{\theta {\overline \vartheta}}{ \overline \lambda} \left[1 - \left({\frac {e} { \alpha \overline\lambda \exp(k)}}\right)  \right]^{\theta - 1}\right) \exp(-k)  \cr 
    &+  \frac {\partial  \Phi^m(k,y)}{\partial y}  \sum_{\ell=1}^L \omega_\ell  \theta_\ell  + \frac {\partial^2 \Phi^m(k,y)}{\partial y^2} \mid \varsigma \mid ^2 e \cr 
    & + 
    \upsilon_d \left( \left[ \gamma_1 + \gamma_2 y + \gamma_3^m (y - \overline y) \right]   \sum_{\ell=1}^L \omega_\ell \theta_\ell  + (\gamma_2 + 
   \gamma_3^m) \mid \varsigma \mid ^2 e \right) 
   \end{align*}

Given :math:`mc` and :math:`\theta = 3`, the first-order conditions for
:math:`e` are affine. Update :math:`e` according to the formula above.


6.2 Modification: pre tech jump
-------------------------------

Add

.. math::


       \xi_r \mathcal{I}_g \left(1 - h + h  \log(h) \right) + \mathcal{I}_g h (\Phi^{\text{post tech, m}} - \Phi^m )

to the above HJB.

6.3 Modification: pre damage jump
---------------------------------

Given :math:`\Phi^m(k, y)` for :math:`m = 1, 2, \dots, M`, solve
:math:`\Phi(k, y)` with extra elements to the HJB:

.. math::


     \xi_r \mathcal{I}_g  \sum \pi_d^m (1 - g_m + g_m  \log(g_m) ) + \mathcal{I}_g\sum g_m \pi_d^m(\Phi^{\text{post tech, m}} - \Phi^m )

.. toggle ::

   .. code:: ipython3

       # packages
       import numpy as np
       import pandas as pd
       import matplotlib.pyplot as plt
       import plotly.graph_objects as go
       from plotly.subplots import make_subplots
       arrival = 20
       γ_3 = np.linspace(0, 1./3, 20)
       πd_o = np.ones_like(γ_3) / len(γ_3)
       θ = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
       πc_o = np.ones_like(θ) / len(θ)
       # load data
       gt_tech_2p5 = np.load('data/new_gt_tech_2.5.npy')
       gt_tech_5 = np.load('data/new_gt_tech_5.0.npy')
       gt_tech_7p5 = np.load('data/new_gt_tech_7.5.npy')
    
       gt_tech_new_2p5 = np.load('data/new_gt_tech_new_2.5.npy')
       gt_tech_new_5 = np.load('data/new_gt_tech_new_5.0.npy')
       gt_tech_new_7p5 = np.load('data/new_gt_tech_new_7.5.npy')
    
       dmg_intensity_distort_2p5 = np.load('data/new_dmg_intensity_distort_2.5.npy')
       dmg_intensity_distort_5 = np.load('data/new_dmg_intensity_distort_5.0.npy')
       dmg_intensity_distort_7p5 = np.load('data/new_dmg_intensity_distort_7.5.npy')
       dmg_intensity_distort_baseline = np.load('data/new_dmg_intensity_distort_100000.0.npy')
    
       intensity_dmg_2p5 = np.load('data/new_intensity_dmg_2.5.npy')
       intensity_dmg_5 = np.load('data/new_intensity_dmg_5.0.npy')
       intensity_dmg_7p5 = np.load('data/new_intensity_dmg_7.5.npy')
       intensity_dmg_baseline = np.load('data/new_intensity_dmg_100000.0.npy')
    
       πct = np.load("data/πct_5.0.npy")
       distorted_damage_probs = np.load("data/distorted_damage_probs_5.0.npy")
    
       # computed distorted probability
       distorted_tech_intensity_first_2p5 = gt_tech_2p5 * 1/arrival
       distorted_tech_intensity_first_5 = gt_tech_5 * 1/arrival
       distorted_tech_intensity_first_7p5 = gt_tech_7p5 * 1/arrival
       tech_intensity_first = np.ones_like(distorted_tech_intensity_first_2p5) * 1/arrival
    
       distorted_tech_intensity_second_2p5 = gt_tech_new_2p5 * 1/arrival
       distorted_tech_intensity_second_5 = gt_tech_new_5 * 1/arrival
       distorted_tech_intensity_second_7p5 = gt_tech_new_7p5 * 1/arrival
       tech_intensity_second = np.ones_like(distorted_tech_intensity_second_2p5) * 1/arrival
    
       distorted_dmg_intensity_2p5 = dmg_intensity_distort_2p5*intensity_dmg_2p5
       distorted_dmg_intensity_5 = dmg_intensity_distort_5*intensity_dmg_5
       distorted_dmg_intensity_7p5 = dmg_intensity_distort_7p5*intensity_dmg_7p5
       distorted_dmg_intensity_baseline = dmg_intensity_distort_baseline*intensity_dmg_baseline
    
       distorted_tech_prob_first_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_2p5, 0, 0)))[:-1]
       distorted_tech_prob_first_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_5, 0, 0)))[:-1]
       distorted_tech_prob_first_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_first_7p5, 0, 0)))[:-1]
       tech_prob_first = 1 - np.exp(-np.cumsum(np.insert(tech_intensity_first, 0, 0)))[:-1]
    
       distorted_tech_prob_second_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_2p5, 0, 0)))[:-1]
       distorted_tech_prob_second_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_5, 0, 0)))[:-1]
       distorted_tech_prob_second_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_tech_intensity_second_7p5, 0, 0)))[:-1]
       tech_prob_second = 1 - np.exp(-np.cumsum(np.insert(tech_intensity_second, 0, 0)))[:-1]
    
       distorted_dmg_prob_2p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_2p5, 0, 0)))[:-1]
       distorted_dmg_prob_5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_5, 0, 0)))[:-1]
       distorted_dmg_prob_7p5 = 1 - np.exp(-np.cumsum(np.insert(distorted_dmg_intensity_7p5, 0, 0)))[:-1]
       dmg_prob = 1 - np.exp(-np.cumsum(np.insert(intensity_dmg_baseline, 0, 0)))[:-1]
    
       # code for figure 14
       from src.plots import plot14
       plot14(
           tech_prob_first, distorted_tech_prob_first_7p5, distorted_tech_prob_first_5,distorted_tech_prob_first_2p5,
           tech_prob_second, distorted_tech_prob_second_7p5, distorted_tech_prob_second_5, distorted_tech_prob_second_2p5,
           dmg_prob, distorted_dmg_prob_7p5, distorted_dmg_prob_5, distorted_dmg_prob_2p5
       )



.. raw:: html

    <iframe src="./_static/Figure14.html" height="600px" width="100%" frameBorder="0"></iframe>


The simulation uses the planner’s optimal solution. The left panel shows
the distorted jump probabilities for the first technology jump. The
middle panel shows the distorted jump probabilities for the second
technology jump. The right panel shows the distorted jump probabilities
for the damage function curvature jump. The baseline probabilities for
the right panel are computed using the state dependent intensities when
we set :math:`\xi_a = \xi_r = \infty.`




.. raw:: html

    <iframe src="./_static/Figure15.html" height="600px" width="100%" frameBorder="0"></iframe>


Figure 15 reports the probability distortions for the damage function
and climate sensitivity models. Here, we have imposed
:math:`\xi_a = .02` and :math:`\xi_r = 5.0`. Note that the damage
function probability distortions are relatively modest, consistent with
our previous discussion. The climate model distortions, by design, are
of similar magnitude as those reported previously in Figure 5.


.. [#] Among other stochastic components, the DSICE incorporates tipping elements and characterizes the SCC as a stochastic process. From a decision theory perspective, DSICE focuses on risk aversion and intertemporal substitution under an assumption of rational expectations.

.. [#] The link to the specification used in :cite:t:`CaiLontzek:2019` is then:

.. math::


   \begin{align*}
   \sigma_t & = \lambda_t \cr
   \vartheta_t & = \theta_{1,t} \cr
   \theta & = \theta_2 \cr
   \mu_t & = \iota_t 
   \end{align*}
