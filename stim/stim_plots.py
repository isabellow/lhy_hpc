import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


def plot_stim_trials(v, t, dt, \
                    before_t=0.01, after_t=0.03):
    '''
    Plot each stimulation trial as a heatmap

    Params
    ------
    v : binned voltage
    t : time bins (s)
    dt : time bin length (s)

    '''
    # define the time window
    b_idx = np.round(before_t / dt).astype(int)
    a_idx = np.round(after_t / dt).astype(int)
    n_bins = b_idx + a_idx

    # get the stimulation events
    min_thresh = 1.5
    stim_idx, _ = find_peaks(-v, height=min_thresh, distance=10000)
    stim_idx = stim_idx[1:-1] # TODO - update to be more general
    n_stim = stim_idx.shape[0]

    stim_events = np.zeros((n_stim, n_bins))
    for i, s_idx in enumerate(stim_idx):
        start = s_idx - b_idx
        end = s_idx + a_idx    
        stim_events[i] = v[start:end]

    # plot the heatmap
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(-stim_events, clim=[0, 0.1], 
              aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(0, n_bins+5, 100))
    ax.set_xticklabels(np.arange(-before_t*1000, (after_t*1000)+5, 10)) # ms

    ax.set_ylabel('stimulation trial')
    ax.set_xlabel('time (ms)')

    return f, ax

def plot_ex_traces(voltage, timestamps, \
                    ex_stim=np.asarray([]), \
                    before_t=0.01, after_t=0.03):
    '''
    Plot traces from example stimulation trials

    Params
    ------
    voltage : raw voltage
    timestamps : raw time points
    ex_stim : array of indices for example stim events

    '''
    # get the stimulation events
    min_thresh = 1.5
    stim_idx, _ = find_peaks(-voltage, height=min_thresh, distance=10000)
    stim_idx = stim_idx[1:-1]
    n_stim = stim_idx.shape[0]
    if ex_stim.shape[0]==0:
        ex_stim = np.arange(n_stim)
    else:
        n_stim = stim_idx[ex_stim].shape[0]

    f, ax = plt.subplots(1, 1, figsize=(8, n_stim))
    for i, s_idx in enumerate(stim_idx[ex_stim]):
        # define the time window
        start_t = timestamps[s_idx] - before_t
        end_t = timestamps[s_idx] + after_t
        start_idx = np.argmin(np.abs(timestamps - start_t))
        end_idx = np.argmin(np.abs(timestamps - end_t))
        
        # align to 0
        t_pts = timestamps[start_idx:end_idx].copy()
        t_pts = t_pts - timestamps[s_idx]

        # plot each trace shifted vertically from the others
        ax.plot(t_pts, voltage[start_idx:end_idx] - i*0.5, 'k')

    # ticks, lims, labels  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim([(-n_stim + 0.5)*0.5, 0.25])

    ax.spines['bottom'].set_bounds(-before_t, after_t)
    ax.set_xticks(np.arange(-before_t, after_t + 0.005, 0.005))
    ax.set_xticklabels(np.arange(-before_t*1000, after_t*1000 + 5, 5))
    ax.set_xlabel('time (ms)')

    return f, ax

