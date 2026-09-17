import numpy as np

import os
import sys
import zlib
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import cluster_ids_for_session
from cell_filters import filter_cells, apply_cell_filter
from event_psth import (window_frames, build_raster, raster_scatter,
                        sort_events_by_duration, subsample_for_raster,
                        event_psth_on_off, shared_ylim, raster_marker_size,
                        plot_psth_trace)
from format_behavior_data import load_behavior_data, get_eating_bouts
from spike_amplitudes import load_spike_amplitudes, select_trials_by_drift

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

'''
Plot eating-aligned activity for single cells.

Each figure is one cell:
    raster (aligned to onset, offsets marked) | onset PSTH | offset PSTH
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'LIM64'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50  # Hz
dt = 1 / fps

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on or bounded by stim-responsive channels
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.05,       # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
# max events plotted in the raster (TC uses every selected bout)
max_events = 200

''' Per-cell trial selection (unit drift) '''
# Drop bouts where the unit looks unhealthy BEFORE the tuning curve is
# computed, so each cell gets its own set of bouts.  The raster then shows
# that cell's surviving bouts, subsampled to max_events as usual.
subsample_drift = False
subsample_metric = 'firing rate'   # 'firing rate' | 'amplitude'
subsample_thresh = 0.5             # keep bouts >= this x the session average
thresh_t_window = 60.0             # seconds centred on the bout

''' Plotting params '''
# tuning-curve windows (seconds) relative to bout onset / offset
fr_on_start, fr_on_end, timepoints_on = window_frames(-0.5, 1.0, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-1.0, 0.5, dt)

# raster window centred on bout onset
event_window = 12                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster + 1) * dt

# 40 ms Gaussian smoothing for the tuning curves
sigma_frames = fps // 25

# style
eat_color = '#006666' # TRQ
psth_lw = 3
event_lw = 1
time_int = 2        # x-tick spacing (s) on the raster
title_size = 14
axis_label = 12

''' Define/create the save folder '''
save_folder = f"{save_figs_dir}/{bird}/eating_activity/"
os.makedirs(save_folder, exist_ok=True)


''' Plot eating responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting eating-bout responses for {bird}_{session_id}')

    ''' Get the file params '''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}/behavior_data/"

    ''' Load and format the neural data '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape

    # session average firing rate (waveform_props is [asymm, width, log10 fr])
    avg_firing_rate = 10 ** data_dict[bird][session_id]['waveform_props'][2]

    ''' Label cells by their KS cluster ID '''
    all_ids = cluster_ids_for_session(data_dict, bird, session_id, root_dir)

    ''' Filter cells — one mask applied to every per-cell array '''
    keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw,
                                 **CELL_FILTERS)
    filtered = apply_cell_filter(keep_cells,
                                 spike_fr=spike_fr,
                                 avg_firing_rate=avg_firing_rate)
    spike_fr = filtered['spike_fr']
    avg_firing_rate = filtered['avg_firing_rate']
    n_cells = spike_fr.shape[0]
    cell_ids = all_ids[keep_cells]

    if n_cells == 0:
        print('  no cells survived the filters, skipping')
        continue
    cells_to_plot = np.arange(n_cells)

    ''' Load and format behavior data '''
    seed_struct, count_data = load_behavior_data(data_dir)
    eat_onsets, eat_offsets = get_eating_bouts(count_data)
    n_eat_true = eat_onsets.shape[0]
    print(f'  {n_eat_true} eating bouts')

    ''' Per-cell bout selection, before anything is averaged '''
    trial_keep = None
    if subsample_drift:
        amp_data = None
        if subsample_metric == 'amplitude':
            session_data = data_dict[bird][session_id]
            ks_dir = f"{bird}_{session_data['ephys_id']}/{session_data['ks_folder']}/"
            amp_data = load_spike_amplitudes(session_dir, data_dir, ks_dir,
                                             cell_ids, n_frames, fps=fps)
        trial_keep = select_trials_by_drift(spike_fr, eat_onsets, dt,
                                      subsample_metric, subsample_thresh,
                                      thresh_t_window, amp_data=amp_data)
        print(f'  {subsample_metric} cut keeps '
              f'{trial_keep.sum(axis=1).min()}-'
              f'{trial_keep.sum(axis=1).max()} of '
              f'{n_eat_true} bouts per cell')

    ''' Tuning curves, from each cell's selected bouts '''
    on_psth, off_psth, _, n_used = event_psth_on_off(
        spike_fr, eat_onsets, eat_offsets,
        (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
        sigma_frames=sigma_frames, keep=trial_keep)
    print(f'  tuning curves from up to {int(n_used[:, 0].max())} eating bouts '
          '(complete windows only)')

    raster_seed = zlib.crc32(f'{bird}_{session_id}_eat'.encode()) & 0xffffffff

    ''' Plot '''
    # 3 visible panels (raster | onset PSTH | offset PSTH); column 1 is a
    # spacer so the raster and tuning curves are not crowded together
    f, ax = plt.subplots(1, 4, figsize=(8, 2),
                         gridspec_kw=dict(width_ratios=[1, 0.2, 1, 1],
                                          wspace=0.08))

    avg_fr_session = np.round(avg_firing_rate, 2)

    for c_idx in cells_to_plot:
        cell_id = cell_ids[c_idx]

        for col in range(4):
            ax[col].cla()

        # ── Cosmetics ────────────────────────────────────────────────────
        for col in [2, 3]:
            ax[col].spines['top'].set_visible(False)
            ax[col].spines['right'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[3].tick_params(labelleft=False)
        for side in ['top', 'left', 'bottom', 'right']:
            ax[1].spines[side].set_visible(False)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_facecolor('none')

        # ── Shared tuning-curve ceiling across both panels ────────────────
        max_fr = shared_ylim(on_psth[c_idx, 0], off_psth[c_idx, 0])

        rng = np.random.default_rng(int(cell_id) * 7919)

        # ── Raster rows: this cell's selected bouts, then subsample ───────
        # rows are per cell because the drift cut is; the seed is not, so
        # with subsample_drift off every cell shows the same bouts
        sel = slice(None) if trial_keep is None else trial_keep[c_idx]
        r_onsets, r_offsets, _, _, _ = subsample_for_raster(
            eat_onsets[sel], eat_offsets[sel],
            max_per_group=max_events,
            rng=np.random.default_rng(raster_seed))
        n_eat_raster = r_onsets.shape[0]

        # order by duration, onset-aligned so the offset is what gets marked
        order, _ = sort_events_by_duration(r_onsets, r_offsets)
        align_frames = r_onsets[order]
        offset_ticks = np.clip((r_offsets - r_onsets)[order] * dt,
                               0, raster_t_pts[-1])

        # ── Raster, aligned to onset, offsets marked ──────────────────────
        raster = build_raster(spike_fr[c_idx], align_frames, fr_halfwidth_raster)
        spk_s = raster_marker_size(ax[0], f, n_eat_raster)
        spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)
        ax[0].scatter(spk_t, spk_row, color=eat_color, marker='|', lw=0.6, s=spk_s)
        ax[0].vlines(0, -0.5, n_eat_raster - 0.5,
                    colors='k', linestyles='dashed', lw=event_lw)
        ax[0].scatter(offset_ticks, np.arange(n_eat_raster),
                     color='k', marker='|', lw=event_lw, s=spk_s / 2, zorder=2)

        # ── Tuning curves ──────────────────────────────────────────────────
        plot_psth_trace(ax[2], timepoints_on, on_psth[c_idx, 0], dt, eat_color,
                        sigma_frames=sigma_frames, lw=psth_lw)
        plot_psth_trace(ax[3], timepoints_off, off_psth[c_idx, 0], dt, eat_color,
                        sigma_frames=sigma_frames, lw=psth_lw)
        for col, t_pts in [(2, timepoints_on), (3, timepoints_off)]:
            ax[col].vlines(0, 0, max_fr, colors='k', linestyles='dashed', lw=event_lw)
            ax[col].hlines(avg_fr_session[c_idx], t_pts[0], t_pts[-1],
                          colors='xkcd:gray', linestyles='dashed', lw=event_lw)

        # ── Limits & ticks ───────────────────────────────────────────────
        ax[0].set_xlim(raster_t_pts[0], raster_t_pts[-1])
        ax[0].set_ylim(-0.5, max(n_eat_raster, 1) - 0.5)
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[0].set_xticks(np.arange(-event_window / 2, event_window / 2 + 0.5, time_int))
        
        ax[2].set_xlim(timepoints_on[0], timepoints_on[-1])
        ax[3].set_xlim(timepoints_off[0], timepoints_off[-1])
        ax[2].set_ylim(0, max_fr)
        ax[3].set_ylim(0, max_fr)
        ax[2].set_yticks([0, max_fr])
        ax[3].set_yticks([])

        # ── Labels ─────────────────────────────────────────────────────────
        ax[0].set_ylabel(f'eating bouts (n={int(n_used[c_idx, 0])})',
                         fontsize=axis_label)
        ax[0].set_xlabel('time from bout onset (s)', fontsize=axis_label)
        ax[2].set_xlabel('time from onset (s)', fontsize=axis_label)
        ax[3].set_xlabel('time from offset (s)', fontsize=axis_label)
        ax[2].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=1.05)

        f.savefig(f'{save_folder}/{session_id}_eating_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
