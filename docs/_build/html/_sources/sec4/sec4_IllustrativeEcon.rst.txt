4 Illustrative Economy
======================

To illustrate our approach to uncertainty, we deliberately use a highly
stylized economic model. Later, we will consider two alternative richer
specifications of the economic environment necessary to address some
important policy challenges.

We start by specifying the economy in the absence of environmental
damages. We pose an :math:`AK` technology for which output is
proportional to capital and can be allocated between investment and
consumption. Capital in this specification should be broadly conceived.
Suppose that there are adjustment costs to capital that are represented
as the product of capital times a quadratic function of the
investment-capital ratio. Given the output constraint and capital
evolution imposed by the :math:`AK` technology, it suffices to let the
planner choose the investment-capital ratio.

Formally, “undamaged” capital evolves as

.. math::


   d K_t =  K_t   \left[ \mu_k (Z_t) dt + \left({\frac {I_t}{K_t}} \right)dt - {\frac { \kappa} 2} \left( {\frac {I_t} {K_t}} \right)^2 dt
   + \sigma_k(Z_t) dW_t^k \right]

where :math:`K_t` is the capital stock and :math:`I_t` is investment.
The capital evolution expressed in logarithms is

.. math::


   d\log K_t =  \left[ \mu_k (Z_t)    + \left({\frac {I_t}{K_t}} \right)  -
   {\frac { \kappa} 2} \left( {\frac {I_t} {K_t}} \right)^2 \right] dt -  {\frac  {\vert \sigma_k(Z_t) \vert^2}  2}dt+ \sigma_k(Z_t) dW_t^k ,

where :math:`K_t` is the capital stock. Consumption and investment are
constrained to be:

.. math::


   C_t + I_t = \alpha K_t

where :math:`C_t` is consumption.

Next, we consider environmental damages. We suppose that temperature
shifts proportionately consumption and capital by a multiplicative
factor :math:`N_t` that captures damages to the productive capacity
induced by climate change. For instance, the damage adjusted consumption
is :math:`{\widetilde C}_t = {\frac {C_t}{N_t}}` and the damage adjusted
capital is :math:`{\widetilde K}_t = {\frac {{K}_t }{N_t}}`. Notice
that:

.. math::


   d \log {\widetilde K}_t = d \log K_t - d \log N_t

Thus, damages induce a deterioration of the capital stock.

Consumer/investor preferences are time-separable with a unitary
elasticity of substitution with an instantaneous time :math:`t`
contribution:

.. math::
   \begin{align*}
    &  (1-\eta) \log {\tilde C}_t +  \eta \log {\mathcal E}_t   \cr & = (1-\eta)( \log C_t -\log K_t ) +  (1-\eta)( \log K_t - \log N_t)   + \eta \log {\mathcal E}_t
    \end{align*}

We let :math:`\delta` be the subjective rate of discount used in
preferences.

   **Remark**

   *The model as posed has a solution that conveniently separates. We
   may solve two separate control problems i) determines “undamaged”
   consumption, investment and capital ii) determines emissions, the
   temperature anomaly and damages. It is the latter one that is of
   particular interest. Undamaged consumption, investment and capital
   are merely convenient constructs that allow us to simplify the model
   solution.*
