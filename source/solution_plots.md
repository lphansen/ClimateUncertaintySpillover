---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: ry38
    language: python
    name: ry38
---

# Plots for BHmodified

```python
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/source')
import pickle
import numpy as np
from global_parameters import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from solver import compute_ell_r_phi, compute_h, compute_sigma2
from derivative import compute_dphidr, compute_dphidz
from simulation import simulate_ems_h, simulate_emission

mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.autolayout'] = True

data_dir = os.path.dirname(os.getcwd()) + '/data/solution/'
figDir = os.path.dirname(os.getcwd())+ '/figures/'

sigma2 = compute_sigma2(RHO, SIGMA_Z, MU_2)
```

```python
sigma2, (RHO, SIGMA_Z, MU_2)
```

```python
solu_yy = pickle.load(open(data_dir + "solu_modified_v6_101*100_0222_16:51", "rb"))
```

```python
solu_yy['e'][50].shape
```

```python
plt.plot(np.linspace(1e-10, 2000, 100), solu_yy['e'][-1] )
plt.xlabel('y')
plt.ylabel('e')
plt.title(r'$e(y,z_2)$, with $z_2 = 1.86/1000$', )
# plt.savefig('e_y_ambiguity.png')
```

```python
Y_GRID.shape
```

```python
plt.plot(np.linspace(1e-10, 2000, 100), solu_yy["e"][0] )
# plt.plot(np.linspace(100,2000, 100), solu_yzeasy[0]['e'][25], label="yesterday")
plt.xlabel('y')
plt.ylabel('e', rotation=0, labelpad=30)
plt.title(r'$e(y,z_2)$ with $z_2 \approx 1.86/1000$')
# plt.savefig(figDir + 'e_yz.png')
```

```python
Y_GRID[0]
```

```python
Z_GRID[50]
```

```python
def simulate_ems(Y, e, T=102, dt=1/4):
    periods = int(T/dt)
    et = np.zeros(periods)
    yt = np.zeros(periods)
    y = 870-580
    yt[0]= y
    for t in range(periods):
        loc = (np.abs(Y-y)).argmin()
        et[t] = e[loc]
        y = y + et[t]*dt
        yt[t] = y
    return et, yt
```

```python
def simulate_emsn(Y, e, T=102, dt=1/4):
    periods = int(T/dt)
    et = np.zeros(periods)
    yt = np.zeros(periods)
    y = 870-580
    yt[0]= y
    for t in range(periods):
        et[t] = np.interp(y, Y, e)
#         et[t] = e[loc]
        y = y + et[t]*dt
        yt[t] = y
    return et, yt
```

```python
et, yt = simulate_emsn(np.linspace(1e-10, 2000, 100), solu_yy['e'][-1])
```

```python
et
```

```python
# 300 represents number of points to make between T.min and T.max
plt.plot(et, label="new HJB")
# plt.plot(solu_bbh[0]['y'], label="BBH")
# plt.legend()
plt.xlabel('years')
plt.ylabel('emission')
plt.xticks(np.arange(0,408,100), np.arange(0,102,25))
# plt.title(r'$e_t$ with $z_2 \approx 1.86/1000$')
# # plt.savefig(figDir + 'e_t_rough.png')
```

```python
plt.plot(et[:400] - solu_bbh[0]['y'], label="new HJB")
```

```python
et[400-1] - solu_bbh[0]['y'][400-1]
```

```python
et[0] - solu_bbh[0]['y'][0]
```

```python
with open("../data/simulation/ems_modified_highrisk", "wb") as handle:
    pickle.dump(et, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
Y = np.linspace(0, 4000, 500)
y0 = (870-580)*np.ones(20)
np.abs(y - y0).argmin()
```

```python
y[36]
```

```python
numz = 10
z_min = 1.86/1000 - 4*.42/1000
z_max = 1.86/1000 + 4*.42/1000
zs = np.linspace(z_min, z_max, numz)

n_ell = 50
log_ell = np.linspace(-13, -5, 50)
ell_step = 1e-7
ells = np.zeros(n_ell*2)
for i in range(n_ell):
    ell = np.exp(log_ell[i])
    ells[2*i] = ell
    ells[2*i + 1] = ell + ell_step 

y = np.linspace(0, 3000, 20)
```

```python
# y and emmission
plt.plot(y, solu_yy[ells[0]]["e"][2,-1])
plt.xlabel("y")
plt.ylabel('emission')
plt.title(r'$e(y,\  b=1,\  z_2 = 1.86/1000 , \ \ell = e^{-13})$')
# plt.savefig(figDir + "e_y.png")
```

```python
solu_yy[ells[0]]["e"][2,-1]
```

```python
plt.plot(y, solu_yy[ells[0]]["psi"][2,-1])
plt.xlabel("y")
plt.ylabel(r'$\phi$', rotation=0)
plt.title(r'$\phi(y,\  b=1,\  z_2 = 1.86/1000 , \ \ell = e^{-13})$')
# plt.savefig(figDir + "phi_y.png")
```

```python
#### em = np.zeros(n_ell*2)
for i in range(n_ell*2):
    em[i] = solu_yy[ells[i]]["e"][2,-1,-1]
```

```python
plt.plot(ells, em)
plt.xlabel(r"$\ell$")
plt.ylabel(r'e', rotation=0, labelpad=20)
plt.title(r'$e(y=3000,\  b=1,\  z_2 = 1.86/1000 , \ \ell )$')
# plt.savefig(figDir + "e_ell.png")
```

```python
phis = np.zeros(n_ell*2)
for i in range(n_ell*2):
    phis[i] = solu_yy[ells[i]]["psi"][2,-1,-1]
```

```python
plt.plot(ells, phis)
plt.xlabel(r"$\ell$")
plt.ylabel(r'$\phi$', rotation=0, labelpad=20)
plt.title(r'$\phi(y=3000,\  b=1,\  z_2 = 1.86/1000 , \ \ell )$')
# plt.savefig(figDir + "phi_ell.png")
plt.show()
```

```python
# compute r
numy = 20
ell_step = 1e-7
psis = np.zeros((numz, numy, n_ell*2))
r_grid = np.zeros((numz, numy, n_ell*2))
for i, ell in enumerate(ells):
    psis[:, :, i] = solu_yy[ell]['psi'][:, -1] 
```

```python
phi = np.zeros((numz, numy, n_ell))
ell_grid = np.zeros(n_ell)
r_grid = np.zeros((numz, numy, n_ell))
for i, lnell in enumerate(log_ell):
    dphi = (psis[:, :, 2*i + 1] - psis[:, :, 2*i])/ell_step
    ell_grid[i] = np.exp(lnell) + ell_step/2
    r_grid[:, :, i] = -dphi
    phi[:, :, i] = (psis[:, :, 2*i + 1] + psis[:, :, 2*i])/2
```

```python
plt.plot(r_grid[2,10], phi[2,10])
```

```python
plt.plot(ell_grid, r_grid[2,10])
```

```python
dphidz = np.zeros((numz-1, numy, n_ell))
dphidz = (phi[1:] - phi[:-1])/(zs[1] - zs[0])
z_new = (zs[1:] + zs[:-1])/2
```

```python
y[10]
```

```python
zs[4], (zs[1:] + zs[:-1])/2, z_new
```

```python
plt.plot(ell_grid, - dphidz[4, 10]*z_new[4]*sigma2**2/XI_M*1e6)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sqrt{z_2} \sigma_2 h_2$', rotation=0, labelpad=50)
plt.title(r'$z_2 = 1.86/1000, y = 1500$')
# plt.savefig(figDir + "h_ell.png")
```

```python

```

```python
ell, r, phi = compute_ell_r_phi(solu_yy, z = z)
```

```python
# import solution
data_dir = os.path.dirname(os.getcwd()) + '/data/solution/'
xi_multi_list = [1, 2, 4]
solu = dict()
for xi_multi in xi_multi_list:
    solu[xi_multi] = pickle.load(open(data_dir + "solu_modified_40*200*200_{:d}0_0203".format(xi_multi), "rb"))
```

```python
log_ell = np.linspace(-13, -5, 200)
ell_step = 1e-7
```


```python
figDir = os.path.dirname(os.getcwd())+ '/figures/'
```

```python
figDir
```

## generate $\ell$, r and $\phi$
${\frac {(z_2)^2 y^2} 2} $

```python
# compute sorted grids
Z = np.linspace(1e-5, 2, 40)
ell = dict()
r = dict()
phi = dict()
for xi_multi in xi_multi_list:
    ell[xi_multi], r[xi_multi], phi[xi_multi] = compute_ell_r_phi(solu=solu[xi_multi],
                                                                 z=Z)
```

```python
%matplotlib widget
plt.plot(r[1][:, 20], ell[1])
plt.show()
```

```python
%matplotlib widget
plt.plot(ell[1], r[1][:, 7])
plt.plot(ell[1], r[1][:, 10])
plt.plot(ell[1], r[1][:, 13])
plt.xlabel(r'$\ell$')
plt.show()
```

```python
# compute r, emission and phi
r_new = dict()
phi_new = dict()
ems = dict()
for xi_multi in xi_multi_list:
    r_new[xi_multi], phi_new[xi_multi], ems[xi_multi] = compute_dphidr(phi[xi_multi], r[xi_multi],z=Z)
```

```python
plt.plot(ems[1][:, 10])
plt.plot(ems[2][:, 10])
```

```python
ems[1][:, 10].shape
```

## Generate z and $\frac{\partial \phi}{\partial z}(r,z)$

```python
z_new = dict()
dphi_dz = dict()
for xi_multi in xi_multi_list:
    z_new[xi_multi], dphi_dz[xi_multi] = compute_dphidz(phi[xi_multi], z=Z)
```

```python
plt.plot(r[1][:, 0], dphi_dz[1][:, 0])
```

## Generate distortion
Formula:
$$
    \sqrt{z_2} \sigma_2 h_2 = - \frac{\frac{\partial \phi}{\partial z}(r,z) z_2 \sigma_2^2}{\xi_m}
$$

```python
SIGMA_2 = compute_sigma2(RHO, .21, 1)
h_z = dict()
for xi_multi in xi_multi_list:
    h_z[xi_multi] = compute_h(dphi_dz[xi_multi], z_new[xi_multi], args=(SIGMA_2, XI_m/xi_multi))
```

```python
plt.plot(ell[1], h_z[1][:, 7])
plt.plot(ell[1], h_z[1][:, 10])
plt.plot(ell[1], h_z[1][:, 13])
```

```python
fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(ell[1], h_z[1][:, 13], label=r'90 percentile of $z_2$')
plt.plot(ell[1], h_z[1][:, 10], label=r'50 percentile of $z_2$')
plt.plot(ell[1], h_z[1][:, 7], label=r"10 percentile of $z_2$")
plt.legend(title=r"with $\xi_m=.000256$")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad = 45, rotation = 0)
plt.title("implied distortion - multiplier")
# plt.savefig("../figures/h_ell_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()
```

```python
fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(ell[4], h_z[4][:, 20], label=r'$\xi_m/40$')
plt.plot(ell[2], h_z[2][:, 20], label=r"$\xi_m/20$")
plt.plot(ell[1], h_z[1][:, 20], label=r"$\xi_m /10$")
plt.legend(title=r"original $\xi_m$ = .00256, median $z_2$")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad=45, rotation=0)
plt.title("implied distortion - multiplier")
# plt.savefig("../figures/h_ell_xi_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()
```

```python
fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(r[4][:, 10], h_z[4][:, 20], label=r'$\xi_m/10$')
plt.plot(r[2][:, 10], h_z[2][:, 20], label=r'$\xi_m/20$')
plt.plot(r[1][:, 10], h_z[1][:, 20], label=r'$\xi_m/40$')
plt.legend(title=r"original $\xi_m$ = .00256, median $z_2$ ")
plt.xlabel(r"$r$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad=45, rotation=0)
plt.title("implied distortion - reserve")
# plt.savefig("../figures/h_r_0202.png", bbox_inches='tight', facecolor="white")
plt.show()
```

## Simulate emission and h

```python
ems_t = dict()
h_t = dict()
for xi_multi in xi_multi_list:
    ems_t[xi_multi], h_t[xi_multi] = simulate_ems_h(dphi_dz[xi_multi], z_new[xi_multi], 
                                                   20, r[xi_multi][:,20], ems[xi_multi][:,20],
                                                    SIGMA_2, XI_m/xi_multi,
                                                   z=Z)
```

```python
plt.plot(ems_t[1])
```

```python
ems_ = dict()
h_ = dict()
for pos in [20, 14, 26]:
    ems_[pos], h_[pos] = simulate_ems_h(dphi_dz[1], z_new[1], pos, r[1][:, pos], ems[1][:, pos],
                                sigma_2=SIGMA_2, xi=XI_m/1, z=Z)
```

```python
fig = plt.figure(figsize=(12, 8), dpi=100)
for pos, percentile in [[26, 90], [20, 50], [14, 10]]:
    plt.plot(h_[pos], label=r'{:d}th percentile of $z_2$'.format(percentile))
plt.legend(title=r"original $\xi_m$ = .000256 ")
plt.xlabel("year")
plt.ylabel("implied distortion")
plt.title("implied distortion - time")
# plt.savefig("../figures/ht_0201.png", bbox_inches='tight', facecolor = "white")
plt.show()
```

```python
fig = plt.figure(figsize = (12,8), dpi = 100)
for xi_multi in [4, 2, 1]:
    plt.plot(h_t[xi_multi], label=r'$\xi_m/{:d}$'.format(10*xi_multi))
plt.legend(title = r"original $\xi_m$ = .00256, median $z_2$ ")
plt.xlabel("year")
plt.ylabel("implied distortion")
plt.title("implied distortion - time")
# plt.savefig("../figures/ht_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()
```

```python
fig = plt.figure(figsize = (12,8), dpi = 100)
for pos, percentile in [[14, 10], [20, 50], [26, 90]]:
    plt.plot(ems_[pos], label=r'{:d}th percentile of $z_2$'.format(percentile))
plt.legend()
plt.xlabel("year")
plt.ylabel("emission")
plt.savefig("../figures/et_0201.png", bbox_inches='tight', facecolor = "white")
plt.show()
```

```python
solu_yb = pickle.load(open(data_dir + "solu_modified_5*50*200_10_0205", "rb"))
```

```python
n_ell = 200
ell_step = 1e-7
log_ell = np.linspace(-13, -5, n_ell)
ells = np.zeros(n_ell*2)
for i in range(n_ell):
    ell = np.exp(log_ell[i])
    ells[2*i] = ell
    ells[2*i + 1] = ell + ell_step 
```

```python
loc = np.abs(ells - 1e-4).argmin()
ells[loc]
```

```python
n_y = 50
ys = np.linspace(0, 3000, n_y)
```

```python
#  y and phi
plt.plot(ys, solu_yb[ells[189]]["psi"][-1])
plt.xlabel('y')
plt.ylabel(r'$\phi$', rotation = 0, labelpad = 15)
plt.title(r'$\phi(y, b=1)$ with $z_2 = 1$ and $\ell \approx 1e-4 $')
# plt.savefig("../figures/phi_y.png")
```

```python
plt.plot(ys, solu_yb[ells[189]]["e"][-1])
plt.xlabel('y')
plt.ylabel(r'$e$', rotation = 0, labelpad = 20)
plt.title(r'$e(y, b = 1)$ with $z_2 = 1$ and $\ell \approx 1e-4$')
# plt.savefig("../figures/e_y.png")
```

```python
phis = np.zeros((3, n_ell*2))
for i, y_i in enumerate([0, 25-1, -1]):
    for j in range(2*n_ell):
        phis[i,j] = solu_yb[ells[j]]["psi"][-1,y_i]
        
emss = np.zeros((3, n_ell*2))
for i, y_i in enumerate([0, 25-1, -1]):
    for j in range(2*n_ell):
        emss[i,j] = solu_yb[ells[j]]["e"][-1,y_i]
```

```python
phis[0].shape, ells.shape
```

```python
plt.plot(ells, phis[0])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\phi$', rotation = 0)
plt.title(r'$\phi(y=0,b=1; \ell)$ with $z_2 = 1$')
# plt.savefig('../figures/phi_ell.png')
```

```python
plt.plot(ells, emss[0])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$e$', rotation = 0, labelpad=20)
plt.title(r'$e(y=0,b=1; \ell)$ with $z_2 = 1$')
# plt.savefig('../figures/e_ell.png')
```

```python
solu_zyb = pickle.load(open(data_dir + 'solu_modified_20*50*50*100_0209', 'rb'))
```

```python
n_ell = 100
ell_step = 1e-7
log_ell = np.linspace(-13, -5, n_ell)
ells = np.zeros(n_ell*2)
for i in range(n_ell):
    ell = np.exp(log_ell[i])
    ells[2*i] = ell
    ells[2*i + 1] = ell + ell_step 
```

```python
n_y = 50
ys = np.linspace(0, 3000, n_y)

n_z = 20
zs = np.linspace(1e-5, 2, n_z)
```

```python
plt.plot(ys, solu_zyb[ells[0]]['psi'][2, -1])
# plt.plot(ys, solu_zyb[ells[0]]['psi'][-1, -1])
plt.xlabel('y')
plt.ylabel(r'$\phi$', rotation=0)
plt.title(r'$\phi(y, b=1, z_2=1; \ell \approx 1e-4)$')
# plt.savefig(figDir + 'phi_y_z.png')
```

```python
plt.plot(ys, solu_zyb[ells[0]]['e'][2, -1])
# plt.plot(ys, solu_zyb[ells[0]]['e'][-1, -1])
plt.xlabel('y')
plt.ylabel(r'e', rotation=0, labelpad=20)
plt.title(r'$e(y, b=1, z_2 =1; \ell \approx 1e-4)$')
# plt.savefig(figDir + 'e_y_z.png')
```

```python
phis_zyb = np.zeros(n_ell*2)
emss_zyb = np.zeros(n_ell*2)
for i in range(n_ell*2):
    phis_zyb[i] = solu_zyb[ells[i]]['psi'][2, -1, 0]
    emss_zyb[i] = solu_zyb[ells[i]]['e'][2, -1, 0]
```

```python
plt.plot(ells, phis_zyb)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\phi$', rotation=0, labelpad=20)
plt.title(r'$\phi(y=0, b=1, z_2=1; \ell)$')
# plt.savefig(figDir + 'phi_ell_z.png')
```

```python
plt.plot(ells, emss_zyb)
plt.xlabel(r'$\ell$')
plt.ylabel(r'e', rotation=0, labelpad=20)
plt.title(r'$e(y=0, b=1, z_2=1; \ell)$')
# plt.savefig(figDir + 'e_ell_z.png')
```

```python
ell, r, phi = compute_ell_r_phi(solu=solu_zyb, z=zs)
```
