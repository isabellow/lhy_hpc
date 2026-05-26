import numpy as np
import spikeinterface as si # core only
import spikeinterface.preprocessing as spre
import dartsort
from dataclasses import replace

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

def remove_stim_events(recording, stim_t, buffer=6, stim_dur=200e-6):
    '''
    Remove stimulation artifacts (+ buffer) from a recording file.

    stim_t : ndarray, shape (n_stims,)
        stim start times in samples relative to recording start
    buffer : int
        number of samples to exclude before and after the stim event
    stim_dur : float
        duration of the stim event in seconds

    Returns the edited recording, as well as start times and duration 
    of stim blank period in samples
    '''
    # recording params
    n_samples = recording.get_num_frames()
    fs = recording.get_sampling_frequency()
    stim_dur_samp = int(stim_dur*fs)

    # get the start/end time for each stim event
    mask_starts = stim_t - buffer
    mask_ends = stim_t + stim_dur_samp + buffer
    mask_duration = stim_dur_samp + (buffer*2)

    # define the windows for stim-free recording
    rec_starts = np.insert(mask_ends, 0, 0)
    rec_ends = np.append(mask_starts, n_samples)

    # make a recording file excluding stim events
    no_stim_recordings = []
    for start, end in zip(rec_starts, rec_ends):
        chunk = recording.frame_slice(start_frame=start, end_frame=end)
        no_stim_recordings.append(chunk)
    rec_no_stim = si.concatenate_recordings(no_stim_recordings)

    return rec_no_stim, mask_starts, mask_duration




def split_by_stim(rec_no_stim, mask_starts, mask_duration, post_mask_t):
    '''
    Split a recording file into spontaneous and stim periods.

    This should be used after removing stim events using remove_stim_events,
    which outputs mask_starts and mask_duration.

    For a recording with stim events removed starting at mask_starts and
    lasting for a fixed duration mask_duration, extract post stim events
    of duration post_mask_t. Concatenate the pre-stim windows for a 
    spontaneous-spiking-only recording and the post-stim windows for a
    recording that includes hash.

    Keep track of global time (relative to original recording) to reassemble
    the final spike trains for alignment to behavior.
    '''
    # recording params
    n_samples = rec_no_stim.get_num_frames()
    fs = rec_no_stim.get_sampling_frequency()
    post_stim_frames = int(post_mask_t*fs)
    n_stim = mask_starts.shape[0]

    # get adjusted time windows for post stim period
    mask_periods = np.full(n_stim, mask_duration)
    removed_samples = np.cumsum(np.insert(mask_periods, 0, 0))
    post_stim_start = mask_starts - removed_samples[:-1]
    post_stim_end = post_stim_start + post_stim_frames

    # chop up and concatenate post stim periods (hash)
    post_stim_recordings = []
    for start, end in zip(post_stim_start, post_stim_end):
        snippet = recording.frame_slice(start_frame=start, end_frame=end)
        post_stim_recordings.append(snippet)
    rec_post_stim = si.concatenate_recordings(post_stim_recordings)

    # save the local and global start/end times
    global_hash_ends = mask_starts + mask_duration + post_stim_frames
    post_stim_map = {}
    post_stim_map['global_starts'] = mask_starts + mask_duration
    post_stim_map['global_ends'] = global_hash_ends
    post_stim_map['snippet_dur'] = np.full(n_stim, post_stim_frames)

    # get time windows for pre stim period
    pre_stim_start = np.insert(post_stim_end, 0, 0)
    pre_stim_end = np.append(post_stim_start, n_samples)

    # chop up and concatenate pre stim periods (spontaneous)
    pre_stim_recordings = []
    for start, end in zip(pre_stim_start, pre_stim_end):
        snippet = recording.frame_slice(start_frame=start, end_frame=end)
        pre_stim_recordings.append(snippet)
    rec_pre_stim = si.concatenate_recordings(pre_stim_recordings)

    # save the local and global start/end times
    pre_stim_map = {}
    pre_stim_map['global_starts'] = np.insert(global_hash_ends, 0, 0)
    pre_stim_map['global_ends'] = np.append(mask_starts, n_samples + n_stim*mask_duration)
    pre_stim_map['snippet_dur'] = pre_stim_end - pre_stim_start

    # store the recordings
    recordings = {}
    recordings['pre_stim'] = rec_pre_stim
    recordings['post_stim'] = rec_post_stim

    # store the timing information
    maps = {}
    maps['pre_stim'] = pre_stim_map
    maps['post_stim'] = post_stim_map

    return recordings, maps

def get_global_spikes(dartsort_object, mapping):
    '''
    Converts spike times aligned to a subsampled recording (e.g. recording minus stim period)
    to spike time aligned to the full recording.

    sorting_object : dartsort sorting or matching object
        created by passing a recording through sorting or template matching
    mapping : list of dicts
        contains global and local start/end times for each recording snippet
    '''
    # spike times aligned to subsampled recording 
    spike_t_local = sorting_obj.times_samples
    
    # realign spike times to the full recording
    spike_t_global = np.zeros_like(spike_t_local)
    for m in mapping:
        spk_idx = (spike_t_local >= m["local_start"]) & (spike_t_local < m["local_end"])
        spike_t_global[spk_idx] = spike_t_local[spk_idx] - m["local_start"] + m["global_start"]
    
    return spike_t_global

def load_ks4_sorting(recording, kilosort_dir):
    times = np.load(f"{kilosort_dir}/spike_times.npy")
    labels_raw = np.load(f"{kilosort_dir}/spike_clusters.npy")
    pos = np.load(f"{kilosort_dir}/spike_positions.npy")
    amp = np.load(f"{kilosort_dir}/amplitudes.npy")

    # change labels to continuous, nonnegative
    unique_labels, labels = np.unique(labels_raw, return_inverse=True)

    # load the sorting, but use 0s for channels because KS
    # doesn't save them. we'll back them out in the next step.
    ks_st = dartsort.DARTsortSorting(
        times_samples=times,
        labels=labels,
        channels=np.zeros_like(labels),
        ephemeral_features=dict(amplitudes=amp)
    )

    # make quick templates (not suitable for matching)
    templates = dartsort.TemplateData.from_config(
        recording=recording,
        sorting=ks_st,
        template_cfg=dartsort.raw_template_cfg,
    )
    
    # set spike channels from template main channels
    main_channels = np.ptp(templates.templates, axis=1).argmax(1)
    
    # final spike train with channel info
    ks_st = dartsort.DARTsortSorting(
        times_samples=times,
        labels=labels,
        channels=main_channels[labels],
        ephemeral_features=dict(
            positions=pos,
            amplitudes=amp,
            times_seconds=recording.sample_index_to_time(times),
        ),
    )
    return ks_st


def split_templates_by_shank(ds_folder, template_data, shank_idx):
    '''
    ds_folder : string
        path to dartsort files
    shank_idx : array of bools, shape (n_channels,)
        which channels belong to each shank
    '''
    # divide channels by shank
    n_channels = shank_idx.shape[0]
    all_channels = np.arange(n_channels)
    shank0_ch = all_channels[shank_idx==0]
    shank1_ch = all_channels[shank_idx==1]

    # get the best channel for each ks unit
    with np.load(f"{ds_folder}/ks_st_realigned.npz") as data:
        ks_labels, label_idx = np.unique(data['labels'], return_index=True)
        best_ch = data['channels'][label_idx]

    # match to the dartsort templates
    with np.load(f"{ds_folder}/template_data.npz") as data:
        ds_labels = data['unit_ids']
        shared_labels, ds_idx, ks_idx = np.intersect1d(ks_labels, ds_labels, return_indices=True)
        best_ch = best_ch[ds_idx]

    # divide by shank
    shank_0_idx = np.isin(best_ch, shank0_ch)
    shank0_units = ds_labels[shank_0_idx]
    shank1_units = ds_labels[~shank_0_idx]

    # package into a new template data structure with correct units/channels
    template_data_0 = template_data[shank0_units]
    template_data_0 = replace(
        template_data_0,
        templates=template_data_0.templates[:, :, shank0_ch],
        spike_counts_by_channel=template_data_0.spike_counts_by_channel[:, shank0_ch],
        registered_geom=template_data_0.registered_geom[shank0_ch],
    )

    template_data_1 = template_data[shank1_units]
    template_data_1 = replace(
        template_data_1,
        templates=template_data_1.templates[:, :, shank1_ch],
        spike_counts_by_channel=template_data_1.spike_counts_by_channel[:, shank1_ch],
        registered_geom=template_data_1.registered_geom[shank1_ch],
    )

    return template_data_0, template_data_1