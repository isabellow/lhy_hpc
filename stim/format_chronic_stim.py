import numpy as np
from scipy.signal import butter, bessel, remez, filtfilt
import time

import sys
sys.path.append("../utils/")
import load_matlab_data
import helpers
sys.path.append("..//neural/")
from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel

import scipy.io
import os

def load_stim(data_dir, stim_pol="neg", verbose=True):
    '''
    Loads the unsorted stim data (arbitrary default channel order)
    this is an array of floats: voltage in microvolts
    shape (n_channels, n_samples, n_stim_events)
    '''
    tic = time.perf_counter()
    ephys_data = np.load(f"{data_dir}amplifier_data_by_stim_{stim_pol}.npy")
    toc = time.perf_counter()
    n_stim = ephys_data.shape[-1]
    if verbose:
        print(f'loaded {n_stim} stim events in {toc-tic} seconds')
    return ephys_data

def load_all_stim_times(data_dir):
    stim_times = np.asarray([])
    for file in sorted(os.listdir(data_dir)):
        if "stim_t" in file:
            stim_t = np.load(f'{data_dir}{file}')
            stim_t = np.squeeze(stim_t.astype(int))
            stim_times = np.append(stim_times, stim_t)
    return np.unique(stim_times).astype(int)

def sort_stim_by_channel(data_dir, ephys_data):
    '''
    Sorts the stim data by the channel order specified in the Intan header.
    '''
    # intan header
    intan_info = load_matlab_data.loadmat_sbx(f"{data_dir}intan_info.mat")
    intan_info = intan_info['header']

    # get the channel IDs
    amp_ch_info_mat = intan_info['amplifier_channels']
    ch_sort_idx = np.asarray([])
    ch_names_unsorted = []
    for amp_ch in amp_ch_info_mat:
        for strg in amp_ch._fieldnames:
            if strg == 'custom_channel_name':
                name = amp_ch.__dict__[strg]
                ch_names_unsorted.append(name)
            elif strg == 'custom_order':
                idx = amp_ch.__dict__[strg]
                ch_sort_idx = np.append(ch_sort_idx, idx)
    ch_sort_idx = ch_sort_idx.astype(int)

    # sort by channel index
    sort_idx = np.argsort(ch_sort_idx)
    ephys_data_sorted = ephys_data[sort_idx]
    ch_names = [ch_names_unsorted[i] for i in sort_idx]

    return ephys_data_sorted, ch_names

def trim_stim_data(ephys_data, ch_names, 
                    broken_ch=np.asarray([]),
                    start_idx=0, end_idx=-1):
    '''
    Trim the data as needed to remove broken channels
    and/or focus on a subset of stim events.
    '''
    n_channels = ephys_data.shape[0]
    all_ch = np.arange(n_channels)
    good_idx = np.setdiff1d(all_ch, broken_ch)
    ch_names_trimmed = [ch_names[i] for i in good_idx]
    ephys_data_trimmed = ephys_data[good_idx, :, start_idx:end_idx]
    return ephys_data_trimmed, ch_names_trimmed    

def filter_stim_data(ephys_data, sampling_rate=30000,
                        t_pre=0.02, stim_duration=200,
                        filt_cut=750, filt_kind='bessel'):
    '''
    sampling_rate : int in Hz
    t_pre : float, seconds collected before stim starts
    stim_duration : int in ms
    filt_cut : int, cut-off for the highpass filter in Hz
    filt_kind : string, type of filter to use (butter or bessel)
    '''
    # mask the stim for filtering
    buffer = 6 # samples
    stim_duration_samples = np.round(stim_duration*1e-6*sampling_rate).astype(int)
    mask_duration = stim_duration_samples + buffer
    samples_pre = int(t_pre*sampling_rate)
    mask_start = samples_pre - buffer
    mask_end = samples_pre + mask_duration

    # highpass filter, masking the stim
    filt_data = np.zeros_like(ephys_data)
    n_stim = ephys_data.shape[-1]
    for i in range(n_stim):
        filt_data_pre_stim = highpass(ephys_data[:, :mask_start, i],
                                      highcut=filt_cut,
                                      fs=sampling_rate, order=2,
                                      axis=-1, kind=filt_kind
                                     )
        filt_data_post_stim = highpass(ephys_data[:, mask_end:, i],
                                       highcut=filt_cut,
                                       fs=sampling_rate, order=2,
                                       axis=-1, kind=filt_kind
                                      )
        filt_data[:, :mask_start, i] = filt_data_pre_stim
        filt_data[:, mask_start:mask_end, i] = ephys_data[:, mask_start:mask_end, i]
        filt_data[:, mask_end:, i] = filt_data_post_stim

    return filt_data

def filter_stim_for_spikes(ephys_data, sampling_rate=30000, t_pre=0.02,
                                stim_duration=200, buffer=6):
    """
    Filters the stim data for comparison with average waveforms obtained by
    getSpikeWaveform.m, masking the stim during filtering.
    """
    # mask the stim for filtering
    stim_duration_samples = np.round(stim_duration*1e-6*sampling_rate).astype(int)
    mask_duration = stim_duration_samples + buffer
    samples_pre = int(t_pre*sampling_rate)
    mask_start = samples_pre - buffer
    mask_end = samples_pre + mask_duration

    # highpass and lowpass filter, masking the stim
    filt_data = np.zeros_like(ephys_data)
    n_stim = ephys_data.shape[-1]
    for i in range(n_stim):
        hplp_data_pre_stim = helpers.bandpass(ephys_data[:, :mask_start, i],
                                            lowcut=800, highcut=5000,
                                            fs=sampling_rate)
        hplp_data_post_stim = helpers.bandpass(ephys_data[:, mask_end:, i],
                                            lowcut=800, highcut=5000,
                                            fs=sampling_rate)
        filt_data[:, :mask_start, i] = hplp_data_pre_stim
        filt_data[:, mask_start:mask_end, i] = ephys_data[:, mask_start:mask_end, i]
        filt_data[:, mask_end:, i] = hplp_data_post_stim

    return filt_data


""" Utils """
def highpass(x, highcut, fs, order=5, axis=-1, kind='butter'):
    """
    Modified slightly from AHW.

    Parameters
    ----------
    x : ndarray
        1d time series data
    highcut : float
        Defines upper frequency cutoff (e.g. in Hz)
    fs : float
        Sampling frequency (e.g. in Hz)
    order : int
        Filter order parameter
    kind : str
        Specifies the kind of filter
        butter for butterworth; bessel for bessel
    axis : int
        Axis along which to bandpass filter data
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    if kind == "butter":
        b, a = butter(order, high, btype="high")
    elif kind == "bessel":
        b, a = bessel(order, high, btype="highpass")
    else:
        raise ValueError("Filter kind not recognized.")
    return filtfilt(b, a, x, axis=axis)


''' To sort/filter cells by stim responsiveness '''
def idx_cells_by_stim(data_dict, bird, session_id):
    '''
    Given stim responsive channels for a bird/session,
    return whether each cell is on or surrounded by stim-responsive channels
    (i.e., in the projection nucleus)
    '''
    # get each cell depth and shank
    cell_pos = data_dict[bird][session_id]['cell_pos']
    n_cells = cell_pos.shape[0]
    cell_dv = cell_pos[:, -1]
    A_idx = data_dict[bird][session_id]['shank_A_idx']
    B_idx = ~A_idx

    # situate each cell in the projection nucleus or neighboring regions
    nucleus_dvs = data_dict[bird][session_id]['nucleus_dvs']
    A_nuc_lims = nucleus_dvs[0]
    B_nuc_lims = nucleus_dvs[1]

    DL_idx = np.zeros(n_cells).astype(bool)
    DL_idx[A_idx] = cell_dv[A_idx] < A_nuc_lims[0]
    DL_idx[B_idx] = cell_dv[B_idx] < B_nuc_lims[0]

    DMZ_idx = np.zeros(n_cells).astype(bool)
    DMZ_idx[A_idx] = cell_dv[A_idx] > A_nuc_lims[1]
    DMZ_idx[B_idx] = cell_dv[B_idx] > B_nuc_lims[1]

    proj_idx = np.abs((DL_idx + DMZ_idx) - 1).astype(bool)
    if bird == 'RBY94':
        proj_idx[B_idx] = False

    return proj_idx

def chunk_cells_by_region(data_dict, bird, session_id):
    '''
    Index cells by location in the brain

    1 = putative projection nucleus
    0 = put. DL (dorsal/lateral)
    2 = put. ventral subiculum/SESN/DMZ (ventral/medial)
    '''
    # get each cell depth and shank
    cell_pos = data_dict[bird][session_id]['cell_pos']
    n_cells = cell_pos.shape[0]
    cell_dv = cell_pos[:, -1]
    A_idx = data_dict[bird][session_id]['shank_A_idx']
    B_idx = ~A_idx

    # situate each cell in the projection nucleus or neighboring regions
    nucleus_dvs = data_dict[bird][session_id]['nucleus_dvs']
    A_nuc_lims = nucleus_dvs[0]
    B_nuc_lims = nucleus_dvs[1]

    DL_idx = np.zeros(n_cells).astype(bool)
    DL_idx[A_idx] = cell_dv[A_idx] < A_nuc_lims[0]
    DL_idx[B_idx] = cell_dv[B_idx] < B_nuc_lims[0]

    DMZ_idx = np.zeros(n_cells).astype(bool)
    DMZ_idx[A_idx] = cell_dv[A_idx] > A_nuc_lims[1]
    DMZ_idx[B_idx] = cell_dv[B_idx] > B_nuc_lims[1]

    if bird == 'RBY94': # likely entire B shank was medial of the nucleus - TODO confirm this if possible with waveform props
        DMZ_idx[B_idx] = True
        DL_idx[B_idx] = False

    proj_idx = np.abs((DL_idx + DMZ_idx) - 1).astype(bool)

    cell_loc_idx = np.full(n_cells, np.nan)
    cell_loc_idx[DL_idx] = 0
    cell_loc_idx[proj_idx] = 1
    cell_loc_idx[DMZ_idx] = 2

    return cell_loc_idx