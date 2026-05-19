import spikeinterface as si # core only
import spikeinterface.preprocessing as spre

def ks4_preprocess(recording, filt=True, cmr=True, whiten=True):
    # Bandpass / highpass
    if filt:
        recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype="float32")

    # Common median reference (similar to KS behavior)
    if cmr:
        recording = spre.common_reference(recording, reference="global", operator="median")

    # Whitening
    if whiten:
        recording = spre.whiten(recording, dtype="float32")

    return recording

def split_by_stim(recording, stim_t, pre_stim_t, post_stim_t):
    '''
    Split a recording file into spontaneous and stim periods.

    For each time point in stim_t, define a window around the stim
    event starting pre_stim samples before and ending post_stim
    samples after that time point.

    Grab these windows and concatenate them into one recording file
    of stim periods. Grab the data outside these windows and
    concatenate into one recording file of spontaneous periods.
    '''
    # recording params
    n_samples = recording.get_num_frames()
    fs = recording.get_sampling_frequency()
    pre_stim = int(pre_stim_t*fs)
    post_stim = int(post_stim_t*fs)


    # get the start and end time for each stim event + hash
    stim_windows = []
    for t in stim_t:
        start = t - pre_stim
        end = t + post_stim
        stim_windows.append([start, end])

    # combine stim windows into one recording file
    stim_recordings = []
    for start, end in stim_windows:
        snippet = recording.frame_slice(start_frame=start, end_frame=end)
        stim_recordings.append(snippet)
    recording_stim = si.concatenate_recordings(stim_recordings) 

    # get the start and end times for each spontaneous spiking period
    spont_windows = []
    post_hash = 0
    for start, end in stim_windows:
        if start > post_hash:
            spont_windows.append((post_hash, start))
        post_hash = end
    if post_hash < n_samples:
        spont_windows.append((post_hash, n_samples))

    # combine spontaneous windows into one recording file
    spont_recordings = []
    for start, end in spont_windows:
        chunk = recording.frame_slice(start_frame=start, end_frame=end)
        spont_recordings.append(chunk)
    recording_spont = si.concatenate_recordings(spont_recordings)

    # store the time windows to later reconstruct the full recording
    stim_t_map = []
    local_cursor = 0
    for start, end in stim_windows:
        dur = end - start
        stim_t_map.append({
            "local_start": local_cursor,
            "local_end": local_cursor + dur,
            "global_start": start,
            "global_end": end
        })
        local_cursor += dur

    spont_t_map = []
    local_cursor = 0
    for start, end in spont_windows:
        dur = end - start
        spont_t_map.append({
            "local_start": local_cursor,
            "local_end": local_cursor + dur,
            "global_start": start,
            "global_end": end
        })
        local_cursor += dur

    # package things up
    recordings = {}
    recordings['stim'] = recording_stim
    recordings['spont'] = recording_spont
    maps = {}
    maps['stim_t'] = stim_t_map
    maps['spont_t'] = spont_t_map

    return recordings, maps

def get_global_spks(sorting_obj, mapping):
    '''
    Converts spike times aligned to a subsampled recording (e.g. recording minus stim period)
    to spike time aligned to the full recording.

    sorting_object : dartsort sorting (or matching) object
        created by passing a recording through sorting or template matching
    mapping : list of dicts
        contains global and local start/end times for each recording snippet
    '''
    # possibly first do sorting_spont.to_numpy_sorting()?
    
    # spontaneous spike times and unit IDs (aligned to subsampled recording)
    spikes = sorting_obj.to_spike_vector()
    spike_t_local = spikes["sample_index"] # spike times
    unit_idx = spikes["unit_index"] # indices
    labels = sorting_obj.get_unit_ids()
    spike_ids = labels[spikes["unit_index"]] # cluster id for each spike
    
    # realign spike times to the full recording
    spike_t_global = np.zeros_like(spk_t_local)
    for m in mapping:
        spk_idx = (spike_t_local >= m["local_start"]) & (spike_t_local < m["local_end"])
        spike_t_global[spk_idx] = spike_t_local[spk_idx] - m["local_start"] + m["global_start"]

    return spike_t_global, spike_ids