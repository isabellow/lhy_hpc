import numpy as np
import pandas as pd

import os 
import sys
sys.path.append("..//utils/")
import color_utils
from load_matlab_data import loadmat_sbx
sys.path.append("..//neural/")
from format_waveform_data import get_spike_times
from format_behavior_data import (load_behavior_data, get_cache_ints,
                                   get_retrieve_ints, spikes_by_cache)
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
'''
plot cache-aligned activity for significantly suppressed and enhanced cells
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'LMN88' # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50 # Hz
dt = 1/fps

# for filtering out cells
fr_thresh = 0.05 # Hz, threshold for excluding low firing cells

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# onset-aligned PSTH window (seconds relative to event onset)
t_on_start = -0.5
t_on_end   =  1.0
timepoints_on = np.arange(t_on_start, t_on_end, dt)
n_t_on = timepoints_on.shape[0]

# offset-aligned PSTH window (seconds relative to event offset)
t_off_start = -1.0
t_off_end   =  0.5
timepoints_off = np.arange(t_off_start, t_off_end, dt)
n_t_off = timepoints_off.shape[0]

# same windows in frames (for slicing spike_fr)
fr_on_start  = int(t_on_start  / dt)
fr_on_end    = int(t_on_end    / dt)
fr_off_start = int(t_off_start / dt)
fr_off_end   = int(t_off_end   / dt)

# raster window centered on event offset (must be an even integer)
event_window        = 10   # seconds
fr_halfwidth_raster = int((event_window // 2) / dt)  # frames on each side

# default time axis for the raster (matches spikes_by_cache output)
default_t_pts = np.arange(-event_window / 2, event_window / 2 + dt, dt)

# style
cache_color = 'xkcd:orange'
ret_color   = 'xkcd:purple'
psth_lw  = 3
event_lw = 1
time_int = 2      # x-tick spacing (s) on the rasters
title_size = 14
axis_label = 12


''' Define/create the save folder'''
save_dir = f"{save_figs_dir}/{bird}/"
if os.path.isdir(save_dir):
    print('save directory exists')
else:
    os.mkdir(save_dir)
save_folder = f"{save_dir}/cache_activity/"
if os.path.isdir(save_folder):
    print('save folder exists')
else:
    os.mkdir(save_folder)

''' Helper function - TODO move elsewhere '''
def compute_event_psth(spike_fr, event_onsets, event_offsets,
                       fr_on_start, fr_on_end, fr_off_start, fr_off_end,
                       n_t_on, n_t_off, n_frames, dt):
    '''
    Average firing rate (Hz) aligned to the onset and offset of each event.
    Trials whose time windows extend outside the session are excluded.

    Returns
    -------
    onset_psth  : (n_cells, n_t_on)   firing rate in Hz
    offset_psth : (n_cells, n_t_off)  firing rate in Hz
    n_valid     : int  — number of complete trials included
    '''
    n_cells = spike_fr.shape[0]
    onset_snippets  = []
    offset_snippets = []

    for ev_on, ev_off in zip(event_onsets.astype(int), event_offsets.astype(int)):
        on_start  = ev_on  + fr_on_start
        on_end    = ev_on  + fr_on_end
        off_start = ev_off + fr_off_start
        off_end   = ev_off + fr_off_end
        if on_start < 0 or off_end > n_frames:
            continue
        onset_snippets.append(spike_fr[:, on_start:on_end])    # (n_cells, n_t_on)
        offset_snippets.append(spike_fr[:, off_start:off_end])  # (n_cells, n_t_off)

    n_valid = len(onset_snippets)
    if n_valid == 0:
        return (np.zeros((n_cells, n_t_on)),
                np.zeros((n_cells, n_t_off)),
                0)

    onset_psth  = np.mean(np.stack(onset_snippets,  axis=0), axis=0) / dt
    offset_psth = np.mean(np.stack(offset_snippets, axis=0), axis=0) / dt
    
    return onset_psth, offset_psth, n_valid


''' Plot cache responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting cache responsive cells for {bird}_{session_id}')

    ''' Get the file params '''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}/behavior_data/"

    ''' Load and format the neural data '''
    # spikes per video frame
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy") # cells x video frames
    spike_bool = spike_fr.astype(bool)
    n_cells_raw, n_frames = spike_fr.shape

    # session average firing rate
    waveform_props = data_dict[bird][session_id]['waveform_props']
    log_fr = waveform_props[2]
    avg_firing_rate = 10**log_fr

    # filter out low-firing cells and cells not in the nucleus
    high_fr = avg_firing_rate > fr_thresh
    if 'stim_resp_idx_ch' in data_dict[bird][session_id].keys():
        stim_idx_cell = idx_cells_by_stim(data_dict, bird, session_id)
    else:
        print(f'warning! no stim data for {bird}_{session_id}')
        stim_idx_cell = np.ones(n_cells_raw).astype(bool)
    excitatory_idx = data_dict[bird][session_id]['excitatory_idx']

    cell_filt_idx = high_fr & stim_idx_cell & excitatory_idx
    
    # spike_fr = spike_fr[cell_filt_idx]
    # spike_bool = spike_bool[cell_filt_idx]
    # avg_firing_rate = avg_firing_rate[cell_filt_idx]
    n_cells = spike_fr.shape[0]

    # get cache-responsiveness
    cache_modulated_filt = data_dict[bird][session_id]['barcode_dict']['cache_modulated']
    # cache_modulated = np.ones(n_cells).astype(bool)
    # cache_modulated_filt = cache_modulated[cell_filt_idx]
    cache_up_bool = cache_modulated_filt == 1
    cache_down_bool = cache_modulated_filt == -1
    print(f'{np.sum(cache_up_bool)} cache up and {np.sum(cache_down_bool)} cache down cells')

    ''' Load and format behavior data '''
    seed_struct, count_data = load_behavior_data(data_dir)

    # event times
    cache_onsets, cache_offsets = get_cache_ints(count_data, seed_struct)
    retrieve_onsets, retrieve_offsets = get_retrieve_ints(count_data, seed_struct)
    n_cache = cache_onsets.shape[0]
    n_ret   = retrieve_onsets.shape[0]

    ''' Raster matrices (activity aligned to event offset) '''
    # only use events that fall within session bounds
    cache_in_bounds = ((cache_offsets - fr_halfwidth_raster) >= 0) & \
                      ((cache_offsets + fr_halfwidth_raster + 1) <= n_frames)
    ret_in_bounds   = ((retrieve_offsets   - fr_halfwidth_raster) >= 0) & \
                      ((retrieve_offsets   + fr_halfwidth_raster + 1) <= n_frames)

    n_cache_raster = int(np.sum(cache_in_bounds))
    n_ret_raster   = int(np.sum(ret_in_bounds))

    cache_mat, cache_t_pts, cache_ons = spikes_by_cache(
        spike_bool,
        cache_onsets[cache_in_bounds], cache_offsets[cache_in_bounds],
        cache_window=event_window, dt=dt)

    ret_mat, ret_t_pts, ret_ons = spikes_by_cache(
        spike_bool,
        retrieve_onsets[ret_in_bounds], retrieve_offsets[ret_in_bounds],
        cache_window=event_window, dt=dt)

    ''' onset and offset aligned PSTHs '''
    cache_onset_psth, cache_offset_psth, n_cache_valid = compute_event_psth(
        spike_fr, cache_onsets, cache_offsets,
        fr_on_start, fr_on_end, fr_off_start, fr_off_end,
        n_t_on, n_t_off, n_frames, dt)

    ret_onset_psth, ret_offset_psth, n_ret_valid = compute_event_psth(
        spike_fr, retrieve_onsets, retrieve_offsets,
        fr_on_start, fr_on_end, fr_off_start, fr_off_end,
        n_t_on, n_t_off, n_frames, dt)

    # smooth with a ~100 ms Gaussian kernel
    sigma = fps // 10
    cache_onset_psth_sm  = gaussian_filter1d(cache_onset_psth,  sigma, axis=1, mode='nearest')
    cache_offset_psth_sm = gaussian_filter1d(cache_offset_psth, sigma, axis=1, mode='nearest')
    ret_onset_psth_sm    = gaussian_filter1d(ret_onset_psth,    sigma, axis=1, mode='nearest')
    ret_offset_psth_sm   = gaussian_filter1d(ret_offset_psth,   sigma, axis=1, mode='nearest')

    ''' Cache tuning index '''
    cache_up_idx = np.where(cache_up_bool)[0]
    cache_down_idx = np.where(cache_down_bool)[0]
    event_tuned_idx = np.append(cache_up_idx, cache_down_idx)
    # event_tuned_idx = cache_up_idx

    ''' Plot '''
    # create figure once and reuse across cells (clear between cells)
    f, ax = plt.subplots(2, 4, figsize=(8, 5),
                         gridspec_kw=dict(width_ratios=[1, 0.2, 1, 1],
                                          wspace=0.08, hspace=0.45))

    for c_idx in event_tuned_idx:
        cell_id = c_idx

        # clear axes from previous cell
        for row in range(2):
            for col in range(4):
                ax[row, col].cla()

        # ── Cosmetics ──────────────────────────────────────────────────
        for row in range(2):
            for col in [2, 3]:
                ax[row, col].spines['top'].set_visible(False)
                ax[row, col].spines['right'].set_visible(False)
            # offset PSTH shares the y-scale with the onset PSTH
            ax[row, 3].spines['left'].set_visible(False)
            ax[row, 3].tick_params(labelleft=False)
            # spacer
            for side in ['top', 'left', 'bottom', 'right']:
                ax[row, 1].spines[side].set_visible(False)
            ax[row, 1].set_xticks([])
            ax[row, 1].set_yticks([])
            ax[row, 1].set_facecolor('none')

        # scale spike marker size to N events
        ax_h_pts    = ax[0, 0].get_position().height * f.get_figheight() * 72
        spk_s_cache = (ax_h_pts / max(n_cache_raster, 1)) ** 2
        spk_s_ret   = (ax_h_pts / max(n_ret_raster,   1)) ** 2
        on_s  = spk_s_cache / 2
        avg_fr_session = np.round(avg_firing_rate, 2)

        # ── Per-cell PSTH y-axis ceiling ───────────────────────────────
        max_cache = float(np.ceil(max(
            np.max(cache_onset_psth_sm[c_idx]),
            np.max(cache_offset_psth_sm[c_idx])))) + 1
        max_ret = float(np.ceil(max(
            np.max(ret_onset_psth_sm[c_idx]),
            np.max(ret_offset_psth_sm[c_idx])))) + 1

        # ── Cache row (row 0) ──────────────────────────────────────────
        # raster (sorted by duration, aligned to offset)
        for ev_idx in range(n_cache_raster):
            spk_idx = cache_mat[c_idx, ev_idx]
            spk_t   = cache_t_pts[spk_idx]
            ax[0, 0].scatter(spk_t, np.full(spk_t.shape[0], ev_idx),
                             color=cache_color, marker='|',
                             lw=0.6, s=spk_s_cache, alpha=1)
        # offset reference line and onset tick marks
        ax[0, 0].vlines(0, 0, n_cache_raster,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[0, 0].scatter(cache_ons, np.arange(n_cache_raster),
                         color='k', marker='|', lw=event_lw, s=on_s, zorder=2)

        # onset PSTH
        ax[0, 2].plot(timepoints_on, cache_onset_psth_sm[c_idx],
                      lw=psth_lw, color=cache_color)
        ax[0, 2].vlines(0, 0, max_cache,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[0, 2].hlines(avg_fr_session[c_idx], t_on_start, t_on_end,
                        colors='xkcd:gray', linestyles='dashed', lw=event_lw)

        # offset PSTH
        ax[0, 3].plot(timepoints_off, cache_offset_psth_sm[c_idx],
                      lw=psth_lw, color=cache_color)
        ax[0, 3].vlines(0, 0, max_cache,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[0, 3].hlines(avg_fr_session[c_idx], t_off_start, t_off_end,
                        colors='xkcd:gray', linestyles='dashed', lw=event_lw)

        # ── Retrieval row (row 1) ──────────────────────────────────────
        # raster
        for ev_idx in range(n_ret_raster):
            spk_idx = ret_mat[c_idx, ev_idx]
            spk_t   = ret_t_pts[spk_idx]
            ax[1, 0].scatter(spk_t, np.full(spk_t.shape[0], ev_idx),
                             color=ret_color, marker='|',
                             lw=0.6, s=spk_s_ret, alpha=1)
        ax[1, 0].vlines(0, 0, n_ret_raster,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[1, 0].scatter(ret_ons, np.arange(n_ret_raster),
                         color='k', marker='|', lw=event_lw, s=on_s, zorder=2)

        # onset PSTH
        ax[1, 2].plot(timepoints_on, ret_onset_psth_sm[c_idx],
                      lw=psth_lw, color=ret_color)
        ax[1, 2].vlines(0, 0, max_ret,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[1, 2].hlines(avg_fr_session[c_idx], t_on_start, t_on_end,
                        colors='xkcd:gray', linestyles='dashed', lw=event_lw)

        # offset PSTH
        ax[1, 3].plot(timepoints_off, ret_offset_psth_sm[c_idx],
                      lw=psth_lw, color=ret_color)
        ax[1, 3].vlines(0, 0, max_ret,
                        colors='k', linestyles='dashed', lw=event_lw)
        ax[1, 3].hlines(avg_fr_session[c_idx], t_off_start, t_off_end,
                        colors='xkcd:gray', linestyles='dashed', lw=event_lw)

        # ── Limits & ticks ─────────────────────────────────────────────
        # rasters
        ax[0, 0].set_xlim(cache_t_pts[0], cache_t_pts[-1])
        ax[0, 0].set_ylim(-0.5, max(n_cache_raster, 1) - 0.5)
        ax[0, 0].set_xticks(np.arange(-event_window / 2,
                                       event_window / 2 + 0.5, time_int))
        ax[1, 0].set_xlim(ret_t_pts[0], ret_t_pts[-1])
        ax[1, 0].set_ylim(-0.5, max(n_ret_raster, 1) - 0.5)
        ax[1, 0].set_xticks(np.arange(-event_window / 2,
                                       event_window / 2 + 0.5, time_int))

        # PSTHs (onset and offset share y-axis per event type)
        for row, max_fr in enumerate([max_cache, max_ret]):
            ax[row, 2].set_xlim(t_on_start, t_on_end)
            ax[row, 3].set_xlim(t_off_start, t_off_end)
            ax[row, 2].set_ylim(0, max_fr)
            ax[row, 3].set_ylim(0, max_fr)
            ax[row, 2].set_yticks([0, max_fr])
            ax[row, 3].set_yticks([])

        # ── Axis labels ────────────────────────────────────────────────
        ax[0, 0].set_ylabel(f'caches (n={n_cache_raster})', fontsize=axis_label)
        ax[1, 0].set_ylabel(f'retrievals (n={n_ret_raster})', fontsize=axis_label)
        for row in range(2):
            ax[row, 0].set_xlabel('time from event offset (s)', fontsize=axis_label)
            ax[row, 2].set_xlabel('time from onset (s)',  fontsize=axis_label)
            ax[row, 3].set_xlabel('time from offset (s)', fontsize=axis_label)
        ax[0, 2].set_ylabel('firing rate (Hz)', fontsize=axis_label)
        ax[1, 2].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        f.suptitle(
            f'{bird} {session_id}  —  cell {cell_id}  '
            f'(baseline {avg_fr_session[c_idx]} Hz)',
            fontsize=title_size, y=1.02)

        # save the figure
        if c_idx in cache_up_idx:
            save_subfolder = f'{save_folder}/cache_up/'
            os.makedirs(save_subfolder, exist_ok=True)
        else:
            save_subfolder = f'{save_folder}/cache_down/'
            os.makedirs(save_subfolder, exist_ok=True)

        f.savefig(f'{save_subfolder}/{session_id}_cache_ret_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)