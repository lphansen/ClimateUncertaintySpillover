5 Uncertainty aversion
======================

This notebook corresponds to the section 5 of the paper.


Here, we mainly discuss the **components of uncertainty**.

The model so far is one of risk as captured by the stochastic
specification of shocks. The presence of shocks opens the door to a
comprehensive assessment of uncertainty conceived more broadly.
Specification uncertainty faced by a decision maker can be discussed in
the formal stochastic structure. We analyze this uncertainty using the
formalism of decision theory under uncertainty. We apply two versions of
such theory, one comes under the heading of variational preferences and
the other under smooth ambiguity preferences. We adapt both to
continuous time specifications, which facilitates their implementation
and interpretation. We use this decision theory to reduce the
sensitivity analysis to a one or two dimensional parameterization that
locates the potential misspecification that is most consequential to a
decision maker. Our aim is to provide a more complete uncertainty
quantification within the setting of the decision problem.

5.1 Misspecified Brownian motion
--------------------------------

| The potential misspecification of a Brownian motion has a particularly
  simple form. It is known from the famed Girsanov Theorem that a change
  in distribution represented by a likelihood ratio is a drift
  distortion. Under such a change in probability distribution, the
  :math:`dW_t` is a Brownian increment with a drift that can be state
  dependent, which we denote :math:`H_t dt`. Thus we modify our
  (locally) normally distributed shocks by entertaining a possible mean
  distortion.
| We use a relative entropy penalty, which for a continuous time
  Brownian specification is the quadratic
  :math:`{\frac 1 2} |H_t|^2 dt`. This formulation leads to a
  straightforward adjustment to an HJB equation. Let :math:`\phi` denote
  a value function defined as a function of the Markov state
  :math:`X_t`. Suppose the local Brownian contribution to the state
  evolution :math:`dX_t` is :math:`\sigma_x(X_t) H_tdt`. Then
  :math:`H_tdt` contributes :math:`\sigma_x(X_t) H_t dt` to the state
  evolution.

As part of recursive robustness adjustment, we solve

.. math::


   \min_h \left(\frac {\partial \phi}{\partial x}\right)\cdot \left(\sigma_x h \right) + {\frac {\xi_b} 2} \mid h\mid^2. 

where :math:`\xi_b` is penalty parameter capturing aversion to potential
misspecification. The solution to this minimization problem is:

.. math::


   \begin{equation} \label{worst_robust}
   h^* = - \frac 1 {\xi_b} {\sigma_x}' \left(\frac {\partial \phi}{\partial x}\right)
   \end{equation}

with minimized objective:

.. math::


   - \frac {1}  {2 \xi_b} \left( \frac {\partial \phi}{\partial x}\right)' \sigma_x {\sigma_x}' \left(\frac {\partial \phi}{\partial x}\right)

The change in the local evolution for :math:`dX_t` is

.. math::


   -  \frac 1 {\xi_b} \sigma_x {\sigma_x}' \left(\frac {\partial \phi}{\partial x}\right)

We explore aversion to the misspecification of Brownian risk by
including this term in the HJB equation. Large values of
:math:`\xi_w` make this contribution less consequential. While
the direct impact is evident from the division by :math:`\xi_w`, the
value function, and hence its partial derivative, also depends on
:math:`\xi`. The partial derivative of the value function is included to
locate distortions that matter to the decision maker.

5.2 Misspecified jump process
-----------------------------

There are two ways that a jump process could be misspecified. The jump
intensity governing locally the jump probability could be wrong or the
probability distribution across the alternative states, in this case
damage function specifications, could be mistaken. We capture both forms
of misspecification by introducing positive random variables :math:`G_t^j \ge 0` for each alternative damage model :math:`j` with
local evolution:

.. math::


   {\mathcal I}(Y_t)\sum_{j=1}^m G_t^j {\pi}_j^p \left[ \phi_j  - \phi \right] 

where the implied intensity is

.. math::


   {\mathcal I}(Y_t) {\overline G}_t

for

.. math::


   {\overline G}_t \doteq \sum_{j=1}^m G_t^j {\pi}_j^p. 

The altered probabilities conditioned on a jump taking place is:

.. math::


   \frac {G_t^j {\pi}_j^p}{ {\overline G}_t}   \hspace{1cm} j=1,2,...,m.

The local relative entropy for a discrepancy in the jump process is:

.. math::


   {\mathcal I}(Y_t) \sum_{j=1}^m {\pi}_j^p\left( 1 - G_t^j + G_t^j \log G_t^j  \right) \ge 0

This measure is nonnegative because the convex function :math:`g \log g`
exceeds its gradient :math:`g - 1` evaluated at :math:`g=1`.

To determine a local contribution to an HJB equation, we solve:

.. math::


   \min_{g^j: j=1,2,...,m}    {\mathcal I}\sum_{j=1}^m g^j \pi_j^p \left( \phi_j  - \phi \right)   + \xi_p \mathcal I \sum_{j=1}^m \pi_j^p \left( 1 - g^j + g^j \log g^j  \right) 

The minimizers are:

.. math::


   g_j^* = \exp\left[  \frac 1 {\xi_p}\left( \phi - \phi_j \right) \right].  

implying a minimized objective:

.. math::


   \xi_p {\mathcal I} \sum_{j=1}^m \pi_j^p \left( 1 - \exp \left[\frac 1 {\xi_p} \left( \phi - \phi_j \right) \right]\right) = - \left(\xi_p {\mathcal I}\right) \frac {\sum_{j=1}^m \pi_j^p \exp \left(- \frac 1 {\xi_p} \phi_j\right) - \exp \left(- \frac 1 {\xi_p} \phi \right)}{\exp \left(- \frac 1 {\xi_p} \phi \right)}

5.3 Local ambiguity aversion
----------------------------

To assess the consequences of the heterogeneous responses from
alternative climate models, we use what are called smooth ambiguity
preferences. In deploying such preferences, we use a robust prior
interpretation in conjunction with the continuous time formulation of
smooth ambiguity proposed by Hansen and Miao. Suppose that we have
:math:`n` different climate model drift specifications for
:math:`\mu_x^i` for model :math:`i`. Let :math:`\pi^a_i` denote the
probability of drift model :math:`i`. Standard model averaging would
have use

.. math::


   \sum_{i=1}^n \pi^a_i \mu_x^i 

and the drift. Our decision maker is uncertain about what weights to
assign but uses an initial set of weights as a baseline. For instance,
in our computations we will treat a collection of models with equal
probability under a baseline and look for a robust adjustment to these
probabilities. Under ambiguity aversion the decision maker with value
function :math:`\phi` solves

.. math::


   \min_{\pi_i, i=1,2,..., n}\sum_{i=1}^n \pi_i \left[ \left(\frac {\partial \phi}{\partial x}\right) \cdot \mu_x^i  + \xi_a \left(\log \pi_i - \log \pi_i^a\right) \right] 

The minimizing probabilities satisfy:

.. math::


   \pi_i^* \propto \pi_i^a \exp\left[ -{\frac 1 {\xi_a}} \left(\frac {\partial \phi}{\partial x}\right) \cdot \mu_x^i \right]

with minimized objective:

.. math::


   - \xi_a \log \sum_{i=1}^n \pi_i^a \exp\left[ -{\frac 1 {\xi_a}} \left(\frac {\partial \phi}{\partial x}\right) \cdot \mu_x^i \right]

In contrast to the robustness adjustment used for model
misspecification, this approach adds more structure to the drift
distortions with the implied distortion to the evolution :math:`dX_t`

.. math::


   \sum_{i=1} \pi_j^* \mu_x^j 

| We have introduced three different parameters
  :math:`(\xi_b, \xi_p, \xi_a)` that guide our sensitivity analysis.
  Anderson, Hansen, Sargent suggest that :math:`\xi_b = \xi_p`. They
  also suggest ways to calibrate this parameter based on statistical
  detection challenges. As the smooth ambiguity model also induces a
  drift distortion, we can adjust the :math:`\xi_a` parameter to have a
  structured drift distortion of a comparable magnitude. We also are
  guided by an approach from robust Bayesian analysis attributed to Good
  that inspects the implied distortions for a *priori* plausibility.
| In our pedagogical discussion so far, we have seemingly ignored
  possible interactions between damage uncertainty and climate
  uncertainty. In fact, these interactions will be present as climate
  change uncertainty will impact the value function contributions given
  by the :math:`\phi_j`\ ’s and by :math:`\phi`.
