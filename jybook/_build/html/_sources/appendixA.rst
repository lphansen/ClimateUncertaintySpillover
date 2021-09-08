Appendix A. Complete model
==========================

A.1 Description
---------------

Suppose the value function is :math:`V`. State variable:
:math:`x = (\log k, z, y, n)` (realization of
:math:`\log K, \log N, Y, Z`), control variable : :math:`(\tilde e, i)`.
Suppose the value function is separable as follows:

.. math::


       V(\log k, n, y, z) = v_k \log k + \zeta(z) + \phi(y) + v_n n

Consider the model without jump misspecification with damage function

.. math::


   \Lambda (y)  = \gamma_1 y + \frac{\gamma_2}{2}y^2 + \frac{\gamma_3}{2} (y - \bar y)^2 \mathbb{I}\{y > \bar y\}

For models with jump misspecification, the modification can be make to
:math:`(y,n)` without affecting :math:`(\log k, z)`, as the HJB is
separable. The complete HJB is as follows:

.. math::


   \begin{aligned}
    0 = \max_{\tilde e, i }\min_{\omega_\ell:\sum_{\ell=1}^L \omega_\ell = 1} \min_{h}\quad & - \delta V + (1 - \eta) [\log(\alpha - i) + \log k - n] + \eta \log \tilde e \\ 
    & + v_k \cdot \left[\mu_k(z) + i  - \frac{\kappa}{2} i^2  - \frac{|\sigma_k(z)|^2}{2} + \sigma_k(z)' h \right] \\ 
    & + \frac{\partial \zeta }{\partial z}(z)\cdot \left[\mu_z(z) + \sigma_z(z)'h \right] + \frac{1}{2} trace\left[\sigma_z(z)' \frac{\partial^2 \zeta(z)}{\partial z\partial z'} \sigma_z(z)'\right] \\
    & + \frac{d \phi(y)}{dy} \sum_{\ell=1}^L \omega_\ell\cdot \tilde e\cdot\theta_\ell + \frac12 \frac{d^2 \phi(y)}{dy^2} (\tilde e)^2 |\varsigma|^2\\
    & + v_n \left[(\gamma_1 + \gamma_2 y + \gamma_3 (y-\bar y)\mathbb{I}\{y > \bar y\}) (\sum_{\ell=1}^L \omega_\ell \theta_\ell \tilde e  + \tilde e \varsigma' h )+ \frac12 \left(\gamma_2 + \gamma_3 \mathbb{I}\{y > \bar y\} \right)\cdot |\varsigma|^2 (\tilde e)^2 \right]  \\
    & + \frac{\xi_b}{2} h'h + \xi_a \sum_{\ell=1}^L \omega_\ell \left( \log \omega_\ell - \log \pi^a_\ell \right)
   \end{aligned}

A.2 Consumption-capital dynamics
--------------------------------

The undamaged version of consumption capital model has a straightforward
solution. The HJB equation for this component is:

.. math::


   \begin{aligned}
    0 = \max_{ i } \min_{h}\quad & - \delta \left[ v_k \log k + \zeta(z)\right] + (1 - \eta) [\log(\alpha - i) + \log k - n] +  \frac{\xi_b}{2} h'h \\ 
    & + v_k \cdot \left[\mu_k(z) + i  - \frac{\kappa}{2} i^2  - \frac{|\sigma_k(z)|^2}{2} + \sigma_k(z)' h \right] \\ 
    & + \frac{\partial \zeta }{\partial z}(z)\cdot \left[\mu_z(z) + \sigma_z(z)'h \right] + \frac{1}{2} trace\left[\sigma_z(z)' \frac{\partial^2 \zeta(z)}{\partial z\partial z'} \sigma_z(z)'\right] \\
   \end{aligned}

Coefficients of :math:`\log k` satisfy that

.. math::


   -\delta v_k + (1 - \eta) = 0 \quad \Longrightarrow \quad v_k = \frac{1 - \eta}{\delta}

The first order condition for the investment-capital ratio is

.. math::


    - (1 - \eta) \frac{1}{\alpha - i} + v_k (1 - \kappa i) = 0\quad \Longrightarrow \quad - \frac{1}{\alpha -i} + \frac{1 - \kappa i}{\delta} = 0

The first order condition for :math:`h` is:

.. math::


   \xi_b h + \sigma_k v_k + \sigma_z \frac{\partial \zeta}{\partial z} = 0

and is therefore:

.. math::


   h = - \frac{1}{\xi_b} \left[ \sigma_k v_k + \sigma_k \frac{\partial \zeta}{\partial z}\right]
