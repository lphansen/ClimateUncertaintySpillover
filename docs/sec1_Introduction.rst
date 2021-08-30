1 Introduction
==============

There are many calls for policy implementation to address climate change
based on confidence in our knowledge of the adverse impact of economic
activity on the climate, and conversely the negative effects of climate
change on economic outcomes. Our view is that the knowledge base to
support quantitative modeling in the realm of climate change and
elsewhere remains incomplete. While there is a substantial body of
evidence supporting the adverse human imprint on the environment,
uncertainty comes into play when we build quantitative models aimed at
capturing the dynamic transmission of human activity on the climate and
on how adaptation to climate change will play out over time. It has been
common practice to shunt uncertainty to the background when building and
using quantitative models in many policy arenas. To truly engage in
“evidence-based policy” requires that we are clear both about the
quality of the evidence and the sensitivity to the modeling inputs used
to interpret the evidence. The importance of quantifying uncertainty has
been stressed and implemented in a variety of scientific settings. Our
aim is to explore ways to incorporate this uncertainty for the purposes
of making quantitative assessments of alternative courses of action. We
see this as much more than putting standard errors on econometric
estimates, and we turn to developments in dynamic decision theory as a
guide to how we confront uncertainty in policy analysis.

In climate economics, :cite:t:`Weitzman:2012` , :cite:t:`WagnerWeitzman:2015` and
others have emphasized uncertainty in the climate system’s dynamics and
how this uncertainty could create fat-tailed distributions of potential
damages. Relatedly, :cite:t:`Pindyck:2013` and
:cite:t:`MorganVaishnavDowlatabadiAzevedo:2017` find existing
integrated assessment models in climate economics to be of little value
in the actual prudent policy. We are sympathetic to their skepticism,
and are not offering simple repairs to the existing integrated
assessment models in this area nor quick modifications to EPA postings
for the social cost of carbon. Nevertheless, we still find value in the
use of models to engage in a form of “quantitative storytelling”.
Instead of proceeding with separate analyses for each such model, we
find value in model comparisons and seek a framework for “quantitative
storytelling” with multiple models. Our aim is to explore ways to
incorporate uncertainty explicitly into policy discussions with a more
explicit accounting for the limits to our understanding. Not only is
there substantial uncertainty about the economic inputs, but also about
the geoscientific inputs.

Drawing on insights from decision theory and asset pricing, Barnett et
al (2020) proposed a framework for assessing uncertainty,
broadly-conceived, to include ambiguity over alternative models and the
potential form of the misspecification of each. In effect, they suggest
methods for conducting structured uncertainty analyses. But their
examples scratch the surface of the actual quantitative assessment of
uncertainty pertinent to the problem of climate change. In this paper,
we explore more systematically the consequences of uncertainty coming
from both geo-scientific and economic inputs.

Decision theory provides tractable ways to explore a tradeoff between
projecting the “best guess” consequences of alternative courses of
action versus “worst possible” outcomes among a set of alternative
models. Rather than focusing exclusively on these extremal points, we
allow our decision maker to take intermediate positions in accordance
with parameters that govern aversions to model ambiguity and potential
misspecification. We presume a decision maker confronts many dimensions
of uncertainty and engages in a sensitivity analysis. We use the social
planner’s decision problem to add structure to this sensitivity analysis
and reduce a potentially high-dimensional sensitivity analysis to a very
low-dimensional characterization of sensitivity parameterized by
aversion to model ambiguity and potential misspecification.

This paper takes inventory of the consequence of alternative sources of
uncertainty and provides a novel way to assess it. We consider three
specific sources:

-  *carbon dynamics* mapping carbon emissions into carbon in the
   atmosphere

-  *temperature dynamics* mapping carbon in the atmosphere into
   temperature changes

-  *economic damage functions* that depict the fraction of the
   productive capacity that is reduced by temperature changes

We necessarily adopt some stark simplifications to make this analysis
tractable. Many of the climate models are of both high dimension and
nonlinear. Rather than using those models directly, we rely on outcomes
of pulse experiments applied to the models. We then take the outcomes of
these pulse experiments as inputs into our simplified specification of
the climate dynamics inside our economic model. We follow much of the
environmental macroeconomic modeling literature in the use of
{:raw-latex:`\em ad hoc`} static damage functions, and explore the
consequences of changing the curvature in these damage functions. Even
with these simplifications, our uncertainty analysis is sufficiently
rich to show how uncertainty about the alternative channels by which
emissions induce economic damages interact in important ways. Modeling
extensions that confront heterogeneity in exposure to climate change
across regions will also open the door to the inclusion of
cross-sectional evidence for measuring potential environmental damages.

We use the social cost of carbon (SCC) as a barometer for investigating
the consequences of uncertainty for climate policy. In settings with
uncertainty, we depict this as an asset price. The social counterpart to
a cash flow is the impulse response from a marginal increase in
emissions to a marginal impact on damages induced by climate changes in
future time periods. This cash flow is discounted stochastically in ways
that account for uncertainty. This follows in part revealing discussions
in :cite:`Golosovetal:2014` and
:cite:`CaiJuddLontzek:2017` who explore some of the risk
consequences for the social cost of carbon. We extend this by taking a
broader perspective on uncertainty. The common discussion in
environmental economics about what “rate” should be used to discount
future social costs is ill-posed for the model ambiguity that we
feature. Rather than a single rate, we borrow and extend an idea from
asset pricing by representing broadly based uncertainty adjustments as a
change in probability over future outcomes for the macroeconomy.

Finally, this paper extends previous work by “opening the hood” of
climate change uncertainty and exploring which components have the
biggest impact on valuation. Rather than embrace a
“one-model-fits-all-type-of-approaches” perspective, we give three
computational examples designed to illustrate different points. The
example presented in `section 4 <sec4_IllustrativeEconI.ipynb>`__ is by
far the most ambitious and sets the stage for the other two. This first
example explores what impact of future information about environmental
and economic damages, triggered by temperature anomaly thresholds,
should have on current policy. It adds a dynamic richness missing from
other treatments of model uncertainty. The second example, presented in
`section 5 <sec5_IllustrativeEconII.ipynb>`__, implements a novel
decomposition of uncertainty assessing the relative importance of
uncertainties in carbon dynamics, temperature dynamics and damage
function uncertainty. The approach that is described and implemented in
`section 5 <sec5_IllustrativeEconIII.ipynb>`__ is more generally
applicable to other economic environments. Finally, the third example
investigates the interacting implications of the uncertainties in the
development of green technologies and in environmental damages for
prudent policy. This example is developed in `section 6 <sec6_IllustrativeEconIII.ipynb>`__.

In the next section, we elaborate on some the prior contributions that
motivate our analysis.

**To next section:**

`Section 2: Uncertainty climate
dynamics <sec2_UncertainClimateDynamics.ipynb>`__
