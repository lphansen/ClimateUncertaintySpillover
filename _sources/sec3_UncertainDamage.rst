
.. admonition:: \   

    This notebook can be accessed through mybinder.org. Click to load the project: |binder|

.. |binder| image:: https://mybinder.org/badge_logo.svg
     :target: https://mybinder.org/v2/gh/lphansen/ClimateUncertaintySpillover.git/macroAnnual_v2?filepath=sec3_UncertainDamage.ipynb

3 Uncertain environmental and economic damages
==============================================

Damage functions
----------------

| For purposes of illustration, we introduce explicitly stochastic
  models of damages. The specification includes an unknown threshold
  whereby the curvature becomes apparent. In some of our computations,
  this threshold occurs somewhere between 1.5 and 2 degrees Celsius, but
  we also explore what happens when this interval is shifted to the
  right, 1.75 and 2.25 degress Celsius in our illustration. Under a
  baseline specification, damage function curvature is realized in
  accordance with a Poisson event and an intensity that depends on the
  temperature anomaly. The event is more likely to be revealed in the
  near future when the temperature anomaly is larger. While we adopt a
  probabilistic formulation as a baseline, we will entertain ambiguity
  over damage function curvature and potential misspecification of the
  Poisson intensity. We intend our specification of the damage function
  to reflect that the value of future empiricism in the near term will
  be limited as the climate-economic system is pushed into uncharted
  territory. On the other hand, we allow for the damage function
  steepness to be revealed in the future as the climate system moves
  potentially closer to an environmental tipping point.
| Our construction of potential damage functions is similar to
  :cite:author:`BarnettBrockHansen:2020` with specifications
  motivated in part by prior contributions.

We posit a damage process, :math:`N_t = \{ N_t : t\ge 0\}` to capture
negative externalities on society imposed by carbon emissions. The
reciprocal of damages, :math:`{\frac 1 {N_t}}`, diminishes the
productive capacity of the economy because of the impact of climate
change. We follow much of climate economics literature by presuming that
the process :math:`N` reflects, in part, the outcome of a damage
function evaluated at the temperature anomaly process. Importantly, we
use a family of damage functions in place of a single function. Our
construction of the alternative damage functions is similar to
:cite:author:`BarnettBrockHansen:2020` with specifications
motivated in part by prior contributions. Importantly, we modify their
damage specifications in three ways:

-  we entertain more damage functions, including ones that are more
   extreme;

-  we allow for damage function steepness to emerge at an *ex ante*
   unknown temperature anomaly threshold;

-  we presume that *ex post* this uncertainty is resolved;

We consider a specification under which there is a temperature anomaly
threshold after which the damage function could be much more curved.
This curvature in the “tail” of the damage function is only revealed to
decision-makers when a Poisson event is triggered. As our model is
highly stylized, the damages captured by the Poisson event are meant to
capture more than just the economic consequences of a narrowly defined
temperature movements. Temperature changes are allowed to trigger other
forms of climate change that in turn can spill over into the
macroeconomy.

In our computational implementation, we use a piecewise log-quadratic
function for mapping how temperature changes induced by emissions alter
economic opportunities. The Poisson intensity governing the jump
probability is an increasing function of the temperature anomaly. We
specify it so that the Poisson event is triggered prior to the anomaly
hitting an upper threshold :math:`{\overline y}`. Construct a process

.. math::


   {\overline Y}_t = \left\{ \begin{matrix} Y_t & t < \tau \cr Y_t - Y_\tau + {\overline y} & t \ge \tau \end{matrix} \right.

where :math:`\tau` is the date of a Poisson event. Notice that
:math:`{\overline Y}_\tau = {\overline y}`. The damages are given by

.. math::


   \log N_t = \Gamma \left({\overline Y}_t  \right) + \iota_n \cdot Z_t 

where:

.. math::


   \Gamma(y) = \gamma_1y + {\frac {\gamma_2} 2} y^2  + {\frac {\gamma_3^m} 2} {\bf 1}_{y \ge {\overline y}}
   ( y- {\overline y} )^2 

| and the only component of :math:`dW` pertinent for the evolution of
  :math:`\iota_n \cdot Z_t` is :math:`dW^n_t`.
| Decision-makers do not know when the Poisson event will be triggered
  nor do they know *ex ante* what the value of :math:`\gamma_3^m` is
  prior to the realization of that event. At the time of the Poisson
  event, one of :math:`M` values of :math:`\gamma_3^m` is realized. In
  our application the coefficients :math:`\gamma_3^m` are specified so
  that the proportional damages are equally spaced after the threshold
  :math:`{\overline y}`.

In our illustration, we consider 20 damage specifications,
:math:`i.e.\ M = 20`. The :math:`\gamma_3`\ ’s are 20 equally spaced
values between 0 and 1/3.

The parameter values are as follows:

===================== ==================================================
Parameter             Value
===================== ==================================================
:math:`\underline{y}` 1.5
:math:`\bar y`        2
:math:`\gamma_1`      0.00017675
:math:`\gamma_2`      0.0044
:math:`\gamma_3^m`    :math:`\frac{0.33 (m - 1)}{19}, m=1, 2, \dots, 20`
===================== ==================================================

For motivation of choosing 1.5 and 2 degrees as thresholds, see *Remark
4.1*.

.. code:: ipython3

    from src.plots import plot4, plot3
    plot4()

.. raw:: html

    <iframe frameBorder="0" width="100%" height="550px" src="./_static/Figure4.html"></iframe>



Intensity function
------------------

| The intensity function, :math:`{\mathcal J}`, determines the
  possibility of a jump over the next small increment in time. For
  :math:`Y_t = y`, :math:`\epsilon \mathcal J (y)` is the approximate
  jump probability over small time increment :math:`\epsilon`.
  Equivalently, :math:`{\mathcal J}` is a local measure of probability
  per unit of time.
| In our computations, we use intensity function

.. math::


   {\mathcal J}(y) = \left\{ \begin{matrix} {\sf r}_1 \left( \exp \left[ {\frac {{\sf r}_2}  2} (y -{\underline y})^2\right] - 1 \right) & y \ge {\underline y} \cr
   0 & 0 \le y < {\underline y} \end{matrix} \right.

Where the values for :math:`r_1` and :math:`r_2` are as follows:

=========== =====
Parameter   Value
=========== =====
:math:`r_1` 1.5
:math:`r_2` 2.5
=========== =====

.. code:: ipython3

    # Intensity function
    plot3()



.. raw:: html

    <iframe frameBorder="0" width="100%" height="550px" src="./_static/Figure3.html"></iframe>