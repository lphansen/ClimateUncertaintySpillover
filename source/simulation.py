import numpy as np


def simulation(y_grid, e_grid, h_grid, g_grid, θ=1.86/1000., y_start=.5394, T=100, σ_y=1.2*1.86/1000., baseline=True):
    Et = np.zeros(T+1)
    yt = np.zeros(T+1)
    ht = np.zeros(T+1)
    gt = np.zeros((len(g_grid), T+1))
    for i in range(T+1):
        Et[i] = np.interp(y_start, y_grid, e_grid)
        ht[i] = np.interp(y_start, y_grid, h_grid)
        for n in range(gt.shape[0]):
            gt[n, i] = np.interp(y_start, y_grid, g_grid[n])
        yt[i] = y_start
        if baseline:
            y_start = y_start + Et[i]*θ
        else:
            y_start = y_start + Et[i]*(θ+σ_y*ht[i])
    return Et, yt, ht, gt
