import numpy as np
from scipy.signal import find_peaks


''' Clustering analysis based on Payne et al. 2021 - for each cell '''
def get_waveform_params(waveform_struct):
    # get avg waveforms and reshape to (n_cells, n_channels, n_timpoints)
    mean_waveforms = np.transpose(waveform_struct['waveFormsMean'], (2, 1, 0)) 
    n_cells = mean_waveforms.shape[0]

    # best channel for each waveform, pythonic indexing
    max_site = waveform_struct['max_site'] - 1

    # get the waveform properties
    fr = waveform_struct['meanRate']
    width = np.zeros(n_cells)
    assym = np.zeros(n_cells)
    for wf_idx in range(n_cells):
        best_ch = max_site[wf_idx]
        width[wf_idx] = calc_spike_width(mean_waveforms[wf_idx, best_ch])
        assym[wf_idx] = calc_amp_assym(mean_waveforms[wf_idx, best_ch])

    return  mean_waveforms, max_site

''' Waveform parameters '''
def calc_spike_width(wf, sampling_rate=30000):
    '''
    Calculate the time (ms) from the trough to the subsequent peak
    of an average waveform

    Params
    ------
    wf : average waveform for the channel with the waveform peak; shape (n_timepoints,)
    '''
    trough_idx = np.argmin(wf)
    spk_w_samples = np.argmax(wf[trough_idx:])
    spk_w = (spk_w_samples / sampling_rate)*1000
    return spk_w

def calc_amp_assym(wf):
    '''
    Calculate the relative height of the two positive peaks flanking the trough
    of an average waveform
    
    (b-a)/(b+a) where a is the first peak, b is the second
    −1 when 1st peak is there and 2nd is not
    0 when 1st = 2nd
    +1 when -1st is not there and 2nd is there

    Params
    ------
    wf : average waveform for the channel with the waveform peak; shape (n_timepoints,)
    '''
    trough_idx = np.argmin(wf)
    a = np.max(wf[:trough_idx])
    b = np.max(wf[trough_idx:])
    return (b - a) / (b + a)



''' stim-related analyses: may move or delete '''
def calc_avg_stim(filt_data, sampling_rate=30000, t_pre=0.02,
                    start_t=2e-3, end_t=16e-3):
    '''
    Get the average across all stim events for a given time winow.

    start_t, end_t : float, seconds; defines time window post stim
    '''
    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)
    total_samples = end_idx - start_idx
    t_window = np.linspace(start_t, end_t, total_samples)

    # get the avg stim event
    avg_stim = np.median(filt_data[:, start_idx:end_idx], axis=-1)
    return avg_stim, t_window


def find_spikes(avg_stim, sampling_rate=30000, 
                prominance_thresh=3, dist_thresh=2e-3):
    # data params
    n_channels = avg_stim.shape[0]
    dist_samples = np.round(dist_thresh*sampling_rate).astype(int)
    
    # get all the peaks for each channel
    spikes_idx = []
    sp_props_all = []
    total_spikes = 0
    channel_max = np.zeros(n_channels)
    ch_max_sp_idx = np.zeros(n_channels)
    for i, stim_ch in enumerate(avg_stim):
        spikes, sp_props = find_peaks(stim_ch,
                                        height=0.5,
                                        prominence=prominance_thresh,
                                        distance=dist_samples)
        total_spikes += spikes.shape[0]
        if spikes.shape[0] > 0:
            channel_max[i] = np.max(sp_props['peak_heights'])
            ch_max_sp_idx[i] = np.argmax(sp_props['peak_heights'])
        spikes_idx.append(spikes.astype(int))
        sp_props_all.append(sp_props)
    ch_max_sp_idx = ch_max_sp_idx.astype(int)
    
    return spikes_idx, sp_props_all, total_spikes, channel_max, ch_max_sp_idx


def find_isolated_spikes(avg_stim, n_templates=50,
                            sampling_rate=30000, t_pre=0.02,
                            prominance_thresh=3, dist_thresh=2e-3,
                            wf_pre=0.5e-3, wf_post=0.5e-3, n_wf_channels=7):
    '''
    Extract waveform templates from a given time window post-stim.
    Only keep templates that are not part of another template, starting with the biggest spike.

    prominance_thresh, dist_thresh : floats for scipy.signal.find_peaks
    wf_pre, wf_post : float, seconds; defines time window for waveform template centered on peak
    n_wf_channels : int; defines number of channels for waveform template, centered on peak
    '''
    n_channels, n_timepts = avg_stim.shape

    # to store variables
    templates = np.asarray([])
    all_sp_ch_idx = np.asarray([])
    all_sp_t = np.asarray([])

    # convert params to samples
    wf_pre_samples = np.round(wf_pre*sampling_rate).astype(int)
    wf_post_samples = np.round(wf_post*sampling_rate).astype(int)

    # for plotting
    wf_window_ms = np.linspace(-wf_pre*1000, wf_post*1000, wf_pre_samples+wf_post_samples)

    # get the initial spike events
    (spikes_idx, sp_props_all, total_spikes,
        channel_max, ch_max_sp_idx) = find_spikes(avg_stim,
                                                    prominance_thresh=prominance_thresh,
                                                    dist_thresh=2e-3)

    # find all spikes that aren't part of other templates
    avg_stim_residual = avg_stim.copy()
    n_timepts = avg_stim.shape[1]
    while (templates.shape[0] < n_templates) & (total_spikes > 0):
        # reset variables
        temp_template = np.zeros_like(avg_stim)

        # find the largest remaining spike
        sp_ch_idx = np.argmax(channel_max)
        max_sp_props = sp_props_all[sp_ch_idx]
        sp_t_idx = spikes_idx[sp_ch_idx][ch_max_sp_idx[sp_ch_idx]]

        # get the template window
        sp_start_ch = sp_ch_idx - n_wf_channels//2
        sp_end_ch = sp_ch_idx + n_wf_channels//2 + 1
        sp_start_t = sp_t_idx - wf_pre_samples
        sp_end_t = sp_t_idx + wf_post_samples
        
        # check edges
        if sp_start_ch < 0:
            sp_start_ch = 0
            sp_end_ch = n_wf_channels
        elif sp_end_ch > n_channels:
            sp_start_ch = n_channels - n_wf_channels - 1
            sp_end_ch = -1
        if sp_start_t < 0:
            sp_start_t = 0
            sp_end_t = wf_pre_samples + wf_post_samples
        elif sp_end_t > n_timepts:
            sp_start_t = n_timepts - (wf_pre_samples + wf_post_samples)
            sp_end_t = -1
        
        # extract the template waveform and save the indices
        new_template = avg_stim_residual[None, sp_start_ch:sp_end_ch, sp_start_t:sp_end_t]
        if templates.shape[0] == 0:
            templates = new_template
        else:
            templates = np.concatenate((templates, new_template), axis=0)
        all_sp_ch_idx = np.append(all_sp_ch_idx, sp_ch_idx)
        all_sp_t = np.append(all_sp_t, sp_t_idx)
            
        # find the residual activity
        temp_template[sp_start_ch:sp_end_ch, sp_start_t:sp_end_t] = new_template.squeeze().copy()
        avg_stim_residual = avg_stim_residual - temp_template
        
        # get remaining peaks
        spikes_idx, sp_props_all, total_spikes, channel_max, ch_max_sp_idx = find_spikes(avg_stim_residual)
        print(f"found {templates.shape[0]} templates, analyzing {total_spikes} remaining spikes")
    all_sp_ch_idx = all_sp_ch_idx.astype(int)
    all_sp_t = all_sp_t.astype(int)

    return templates, all_sp_ch_idx, all_sp_t, wf_window_ms


def trial_trial_correlations(filt_data, templates, 
                                wf_ch_idx, wf_times,
                                sampling_rate=30000, t_pre=0.02,
                                t_buffer=2e-3):
    ''' 
    Sweep the templates over the filtered data on each trial to find the best correlation. 

    Each waveform template has a channel associated with the biggest peak and spans n_wf_channels.
    Here, we search for occurances of that template across just those channels and within
    a limited time window before and after the peak time defined by t_buffer.

    Correlations are calculated separately for each channel, then summed across channels to get a
    composite correlation for all template channels over time (so stronger correlations will win out).

    Params
    ------
    filt_data : filtered ephys data; shape (n_channels, n_timepts, n_stim)
    templates : waveform templates for putative antidromic responses; 
                shape (n_wf_cells, n_wf_channels, n_wf_pts)
    wf_ch_idx : channel index for best spike in each template
    wf_times : (seconds) time post stim that the best spike in each template peaked
    sampling_rate : (Hz)
    t_pre : (seconds) param defining how much data was taken before the stim
    t_buffer : (seconds) time window before and after the template peak to search for spikes

    Returns
    -------
    all_correlations : correlation of each template with the data;
                        shape (n_wf_cells, n_samples, n_stim);
                        where n_samples is 2 * t_buffer * sampling rate
                        note that edges will be zeros to avoid edge effects
                        (see documentation for mode = 'valid' in numpy.convolve)
    t_windows : (seconds) timepoints over which the correlation was performed
                (relative to stim time)
    '''
    # data params
    n_samples = int(t_buffer*2*sampling_rate)
    n_wf_cells, n_wf_channels, n_wf_pts = templates.shape
    n_channels, n_timepts, n_stim = filt_data.shape

    # to store the correlation values and indices
    all_correlations = np.zeros((n_wf_cells, n_samples, n_stim))
    t_windows = np.zeros((n_wf_cells, n_samples))
    for i, temp in enumerate(templates):
        # get the data for this template
        sp_ch = wf_ch_idx[i]
        sp_t = wf_times[i]
        
        # set the indices
        start_ch = sp_ch - n_wf_channels//2
        end_ch = sp_ch + n_wf_channels//2 + 1
        start_t = sp_t - t_buffer
        end_t = sp_t + t_buffer
        
        # check edges
        if start_ch < 0:
            start_ch = 0
            end_ch = n_wf_channels + 1
        elif end_ch > n_channels:
            start_ch = n_channels - (n_wf_channels + 1)
            end_ch = -1
        if start_t < 0:
            start_t = 0
            end_t = wf_pre_samples + wf_post_samples + 1
        elif end_t > n_timepts:
            start_t = n_timepts - (wf_pre_samples + wf_post_samples + 1)
            end_t = -1
        
        # set the time window to look at
        start_t_adj = start_t + t_pre
        end_t_adj = end_t + t_pre
        start_idx = np.round(start_t_adj*sampling_rate).astype(int)
        end_idx = np.round(end_t_adj*sampling_rate).astype(int)
        t_windows[i] = np.linspace(start_t, end_t, end_idx - start_idx)
        
        # examine a chunk of data around the peak for best correlation   
        corr = np.zeros_like(filt_data[start_ch:end_ch, start_idx:end_idx])
        start_corr_idx = n_wf_pts//2
        end_corr_idx = (-n_wf_pts//2) + 1
        for s in range(n_stim):
            data_chunk = filt_data[start_ch:end_ch, start_idx:end_idx, s]
            for ch in range(data_chunk.shape[0]):
                corr[ch, start_corr_idx:end_corr_idx, s] = np.correlate(data_chunk[ch], 
                                                                            temp[ch], 
                                                                            mode='valid')
        corr_composite = np.sum(corr, axis=0) # combine correlations across channels
        all_correlations[i] = corr_composite

    return all_correlations, t_windows