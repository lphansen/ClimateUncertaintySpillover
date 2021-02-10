# Computation Strategy

## One state variable: reserve $r$

## Two state variable: $(r, z_2)$

maximzation problem involves $e$:
$$
    \max_{e} \delta \eta \log{e} - \tau z_2 e - \frac{\partial \phi(r,z_2)}{\partial r}e   
$$

Introduce a state variable $B_t$:
$$
    dB_t = - \delta B_tdt
$$
with $B_0 = 1$.

Construct optimization problem:
$$
    \max_{E_t} \int B_t(\delta\eta \log{E_t} - \tau Z_t^2 E_t)dt - \ell E_t
$$

Functional equation:
$$
    0 = \max_{e} b(\delta\eta \log{e} - \tau z_2 e ) - \ell e - \frac{\partial \psi(b, \ell, z_2)}{\partial b}\delta b
$$

solve for $e$:
$$
    e^\star(b,\ell, z_2) = \frac{b\delta\eta }{b\tau z_2 + \ell}
$$


Suppose the size of b grid $N_b$. 
Write the linear system:


$$
    \delta b \frac{ \psi( b_1, \ell, z_2 ) - \psi ( b_0, \ell, z_2) }{ \Delta b }  = b_0 \cdot (\delta \eta \log{e^\star} - \tau z_2 e^\star) - \ell e^\star
$$


$$
   \delta b \frac{ \psi(b_{i+1}, \ell, z_2) -   \psi(b_{i-1}, \ell, z_2) }{2\Delta b} = b_i \cdot (\delta\eta\log{e^\star} - \tau z_2 e^\star) - \ell e^\star, \text{for } i = 1, \cdots, N_b - 2
$$


$$
    \delta b \frac{\psi(b_{N_b-1}, \ell, z_2) -  \psi(b_{N_b-1}, \ell, z_2)}{\Delta b} = b_{N_b} \cdot (\delta\eta\log{e^\star} - \tau z_2 e^\star) - \ell e^\star
$$

Construct $\phi(r, z_2)$:

$$
    \phi(r, z_2) = \min_{\ell \geqslant 0} \psi(1, \ell, z_2) + \ell r
$$

First order condition for $\ell$:

$$
    \frac{\partial \psi(1, \ell, z_2)}{\partial \ell} + r = 0
$$

form $\phi(r, z_2)$ numerically.
$$
    \phi(r, z_2) = \psi^\star(\ell, z_2) + \ell r
$$
