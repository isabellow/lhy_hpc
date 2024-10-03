import numpy as np
from scipy import stats

def tuning_curve_1d(x, Y, dt, b, smooth=True, l=2, SEM=False):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    l : int
        smoothness param for gaussian filter; optional, default is 2
    SEM : bool
        return SEM for FR; optional, default is False
    Returns
    -------
    firing_rate : ndarray
        trial-averaged, binned firing rate for each cell
        shape (n_bins, n_cells)
    centers : ndarray
        center of each bin
    '''
    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)
    # find FR in each bin
    firing_rate = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    spike_sem = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    for i in range(unique_bdx.shape[0]):
        spike_ct = np.sum(Y[b_idx == unique_bdx[i], :], axis=0)
        occupancy = dt * np.sum(b_idx==unique_bdx[i])
        spike_sem[i, :] = stats.sem(Y[b_idx == unique_bdx[i], :]/dt, axis=0)
        firing_rate[i, :] = spike_ct / occupancy
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, l, axis=0, mode='wrap')
        spike_sem = gaussian_filter1d(spike_sem, l, axis=0, mode='wrap')
    if SEM:
        return firing_rate, centers, spike_sem
    else:
        return firing_rate, centers


def tuning_curve_2d(X, Y, dt, b_lims, b):
    '''
    Params
    ------
    X : ndarray
        variable of interest by observation; shape (n_obs, 2)
    Y : ndarray
        spikes per observation; shape (n_cells, n_obs)
    dt : int
        time per observation in seconds
    b_lims : tuple of floats
        min/max values considered
    b : int
        bin size
    
    Returns
    -------
    firing_rate : ndarray
        average firing rate per 2d bin
        shape (n_bins, n_bins)
    centers : ndarray
        center of each bin
        shape (2, n_bins)
    bin_idx : ndarray
        index for each firing rate
        shape (2, n_bins)
    '''
    edges = np.arange(b_lims[0], b_lims[1] + b, b)
    centers = (edges[:-1] + edges[1:])/2
    
    x1_idx = np.digitize(X[:, 0], edges)
    if np.max(X[:, 0]) == edges[-1]:
        x1_idx[x1_idx==np.max(x1_idx)] = np.max(x1_idx) - 1
    x2_idx = np.digitize(X[:, 1], edges)
    if np.max(X[:, 1]) == edges[-1]:
        x2_idx[x2_idx==np.max(x2_idx)] = np.max(x2_idx) - 1
    unique_bdx_x1 = np.unique(x1_idx)
    unique_bdx_x2 = np.unique(x2_idx)
    
    n_bins = edges.shape[0] + 1
    n_cells = Y.shape[0]
    
    firing_rate = np.zeros((n_cells, n_bins, n_bins))
    for i in unique_bdx_x1:
        for j in unique_bdx_x2:
            pos_idx = (x1_idx == i) & (x2_idx == j)
            spike_ct = np.sum(Y[:, pos_idx], axis=1)
            occupancy = dt * np.sum(pos_idx)
            if occupancy < 0.1:
                firing_rate[:, i, j] = np.NaN
            else:
                firing_rate[:, i, j] = spike_ct / occupancy
    
    return firing_rate, centers