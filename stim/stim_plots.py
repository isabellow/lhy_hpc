import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


def plot_stim_trials(v, t, stim_idx, \
                    sampling_rate=30000, \
                    before_t=0.01, after_t=0.03, \
                    v_min=0, v_max=50
                    ):
    '''
    Plot each stimulation trial as a heatmap

    Params
    ------
    v : binned voltage
    t : time bins (s)
    stim_idx : which time bins correspond to a stim event
    
    sampling_rate : n samples per second
    before_t, after_t : amount of time before/after the stim (s)
    v_min, v_max : c_lim for heatmap (microvolts)
    '''
    # define the time window
    b_idx = np.round(before_t * sampling_rate).astype(int)
    a_idx = np.round(after_t * sampling_rate).astype(int)
    n_bins = b_idx + a_idx

    # get the stimulation events
    n_stim = stim_idx.shape[0]
    stim_events = np.zeros((n_stim, n_bins))
    for i, s_idx in enumerate(stim_idx):
        start = s_idx - b_idx
        end = s_idx + a_idx    
        stim_events[i] = v[start:end]

    # plot the heatmap
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(-stim_events, clim=[v_min, v_max], 
              aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(0, n_bins+5, 150))
    ax.set_xticklabels(np.arange(-before_t*1000, (after_t*1000)+5, 5)) # ms

    ax.set_ylabel('stimulation trial')
    ax.set_xlabel('time (ms)')

    return f, ax

def plot_ex_traces(voltage, timestamps, \
                    ex_stim=np.asarray([]), \
                    before_t=0.01, after_t=0.03,\
                    scale_bar=False):
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
    stim_idx, _ = find_peaks(-voltage, height=min_thresh) #, distance=10000)
    stim_idx = stim_idx[1:-1]
    n_stim = stim_idx.shape[0]
    if ex_stim.shape[0]==0:
        ex_stim = np.arange(n_stim)
    else:
        n_stim = stim_idx[ex_stim].shape[0]

    f, ax = plt.subplots(1, 1, figsize=(10, n_stim/1.5))
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
        ax.plot(t_pts, voltage[start_idx:end_idx] - i*0.4, 'k')

    # plot scale bar or x-axis
    if scale_bar:
        ax.plot([after_t - 0.005, after_t], \
            [-(i+1)*0.4, -(i+1)*0.4], \
            'k', lw=3)
        ax.text(after_t - 0.0025, -(i+1)*0.4 - 0.15, '5ms', ha='center')
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_ylim([(-(n_stim + 1))*0.4, 0.5])
    else:
        ax.spines['bottom'].set_bounds(-before_t, after_t)
        ax.set_xticks(np.arange(-before_t, after_t + 0.005, 0.005))
        ax.set_xticklabels(np.arange(-before_t*1000, after_t*1000 + 5, 5))
        ax.set_xlabel('time (ms)')
        ax.set_ylim([(-n_stim)*0.4, 0.5])

    # hide other axes  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])

    return f, ax

