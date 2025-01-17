import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def plot_raw_ephys(ephys_data, ch_num, ex_trials=[10, 20],
                    sampling_rate=30000, t_pre=0.02, t_post=0.03):
    """
    ex_trials : [start_trial, end_trial] for example traces
    sampling_rate : int, HZ
    t_pre : float seconds collected before stim starts
    t_post :float seconds collected after stim time
    """
    # convert samples to time in seconds
    n_samples = ephys_data.shape[1]
    time_pts = np.linspace(-t_pre, t_post, num=n_samples)

    # make sure the data looks reasonable
    ch_idx = ch_num-1
    ex_channel = ephys_data[ch_idx]

    # traces
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(time_pts, ex_channel[:, ex_trials[0]:ex_trials[1]])
    ax.set_xlim([-t_pre, t_post])
    # ax.set_ylim([-200, 200])
    ax.set_ylabel('voltage (uV)')
    ax.set_xlabel('time (sec)')
    ax.set_title('raw ephys (10 stims)')
    plt.show()

    # heat map
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(-ex_channel.T, aspect='auto', cmap='viridis')
    ax.set_xticks(np.linspace(0, n_samples, 6))
    ax.set_xticklabels(np.round(np.arange(-t_pre, t_post + 0.01, 0.01), 2))
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('stim events')
    ax.set_title('raw ephys (all stims)')
    
    return f, ax

def plot_avg_stim(filt_data, start_t=-0.005, end_t=0.02,
                    sampling_rate=30000, t_pre=0.02,
                    v_min=0, v_max=10, take_median=True):
    '''
    filt_data : bandpass or highpass filtered ephys data
    start_t, end_t : window to plot, relative to stim, in seconds
    v_min, v_max : colormap limits in microvolts
    v_max = 10 # microvolts
    '''
    # avg across stim events
    if take_median:
        avg_stim = np.median(filt_data, axis=-1)
    else:
        avg_stim = np.mean(filt_data, axis=-1)

    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)

    # grab that window
    stim_events = avg_stim[:, start_idx:end_idx]
    n_bins = stim_events.shape[1]

    # plot for all channels
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(-stim_events, clim=[v_min, v_max],
                aspect='auto', cmap='Greys',
                interpolation='gaussian')

    # axes and labels
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks(np.arange(0, n_bins+1, 0.005*sampling_rate))
    ax.set_xticklabels(np.arange(start_t*1000, end_t*1000+5, 5))
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('good channels') # check dorsal vs. ventral order
    ax.set_title('median stim response across channels')

    return f, ax

def plot_avg_stim_by_shank(filt_data, ch_names,
                            start_t=0, end_t=0.015,
                            sampling_rate=30000, t_pre=0.02,
                            v_min=0, v_max=5, 
                            smooth=True, take_median=True):
    '''
    as above, but sorted by channel column and shank (if H6)
    '''
    n_channels, n_samples, n_stim = filt_data.shape

    # avg across stim events
    if take_median:
        avg_stim = np.median(filt_data, axis=-1)
    else:
        avg_stim = np.mean(filt_data, axis=-1)

    ''' organize by channel column and shank '''
    if 'B-01' in ch_names: # H6
        split_stims = np.zeros((4, n_channels//4, n_samples))
        split_stims[0] = avg_stim[:n_channels//2:2]
        split_stims[1] = avg_stim[1:n_channels//2:2]
        split_stims[2] = avg_stim[n_channels//2::2]
        split_stims[3] = avg_stim[(n_channels//2)+1::2]
        
        # rough physical layout
        microns = np.arange(25, 425, 25)  
    else: # H5
        split_stims = np.zeros((2, n_channels//2, n_samples))
        split_stims[0] = avg_stim[::2]
        split_stims[1, :24] = avg_stim[1::2]
        
        # rough physical layout
        microns = np.arange(25, 825, 25)

    ''' plot avg stim event organized by channel map - smoothed '''
    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)

    # grab that window
    stim_events = split_stims[:, :, start_idx:end_idx]
    n_bins = stim_events.shape[-1]

    # fig params
    f, ax = plt.subplots(1, 5,
                        figsize=(6, 4),
                        gridspec_kw=dict(wspace=0.1))

    # plot for all channels
    for i, stim_event in enumerate(stim_events):
        stim_event = stim_event.squeeze()
        if i < 2:
            idx = i
        else:
            idx = i+1
        ax[idx].imshow(-stim_event, clim=[v_min, v_max],
                       aspect='auto', cmap='viridis')
        ax[idx].set_xticks(np.arange(0, n_bins+1, 0.004*sampling_rate))
        ax[idx].set_xticklabels((np.arange(start_t*1000, end_t*1000+1, 4)).astype(int))
        if i==0:
            ax[idx].set_ylabel('approx. depth on probe (um)')  
            ax[idx].set_title('A shank, avg response', loc='left')
            ax[idx].set_yticks(np.arange(0, stim_event.shape[0], 2))
            ax[idx].set_yticklabels(microns[-1::-2])
        else:
            ax[idx].tick_params(labelleft=False)
        if i == 2:
            ax[idx].set_title('B shank, avg response', loc='left')
        ax[idx].set_ylim(ax[idx].get_ylim()[::-1])
        

    ax[2].set_xlabel('time post-stim (ms)', labelpad=20)
    ax[2].tick_params(labelleft=False, labelbottom=False)

    return f, ax

def plot_avg_stim_trace(filt_data, ch_names, ch_idx,
                        start_t=-0.01, end_t=0.02,
                        sampling_rate=30000, t_pre=0.02,
                        ymin=-40, ymax=40):
    # get the average response for the channel
    ch_id = ch_names[ch_idx]
    ex_channel = filt_data[ch_idx].squeeze()
    avg_response = np.median(ex_channel, axis=1)

    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)

    # grab that window
    avg_response_subset = avg_response[start_idx:end_idx]
    n_bins = avg_response_subset.shape[0]

    # plot it
    f, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(avg_response_subset, '-k', lw=2)

    # ticks and lims
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([0, n_bins])
    ax.set_xticks(np.arange(0, n_bins+1, 0.005*sampling_rate))
    ax.set_xticklabels(np.arange(start_t*1000, end_t*1000+5, 5))

    # axis params
    ax.spines['bottom'].set_bounds(0, n_bins)
    ax.spines['left'].set_bounds(ymin, ymax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # labels
    ax.set_ylabel('voltage (uV)')
    ax.set_xlabel('time from stim (ms)')
    ax.set_title(f'avg response for channel {ch_id}')

    return f, ax
        

def plot_stim_trials(filt_data, ch_names, ch_idx,
                        start_t=-0.01, end_t=0.02,
                        sampling_rate=30000, t_pre=0.02, \
                        v_min=0, v_max=50):
    '''
    Plot the stimulation trials as a heatmap for a given channel
    '''
    # choose a channel to look at
    ch_id = ch_names[ch_idx]
    ex_channel = filt_data[ch_idx].squeeze()

    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)

    # grab that window
    stim_events = ex_channel.T[:, start_idx:end_idx]
    n_bins = stim_events.shape[1]

    # plot for this channel
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(-stim_events, clim=[v_min, v_max],
              aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(0, n_bins+1, 0.005*sampling_rate))
    ax.set_xticklabels(np.arange(start_t*1000, end_t*1000+5, 5))
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('stim events') # check dorsal vs. ventral order
    ax.set_title(f'stim response for channel {ch_id}')

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

