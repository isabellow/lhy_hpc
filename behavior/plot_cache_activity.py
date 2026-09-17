import numpy as np

import os
import sys
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import cluster_ids_for_session
from cell_filters import filter_cells, apply_cell_filter
from event_psth import (window_frames, build_raster, raster_scatter,
                        sort_events_by_duration, event_psth_on_off,
                        shared_ylim, raster_marker_size, plot_psth_trace)
from format_behavior_data import (load_behavior_data, get_cache_ints,
                                  get_retrieve_ints)
from spike_amplitudes import load_spike_amplitudes, select_trials_by_drift

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

'''
Plot cache- and retrieval-aligned activity for single cells.

Each figure is one cell:
    row 0  caches      raster (aligned to offset) | onset PSTH | offset PSTH
    row 1  retrievals  raster (aligned to offset) | onset PSTH | offset PSTH

All four tuning curves share one y axis, so cache and retrieval responses can
be compared directly for each cell.

Uses get_cache_ints/get_retrieve_ints rather than the *_refined variants: the
refined windows are padded and truncated to give a single mean activity value
per event, whereas a PSTH needs the real interaction onset and offset.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'LMN86'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
session_list = ['260828']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on or bounded by stim-responsive channels.
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.05,       # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

# which cells get a figure:
#   'cache_modulated' — only cells the shuffle test flagged (barcode_dict)
#   'all'             — every cell that survives CELL_FILTERS
CELL_SELECTION = 'all'

''' Per-cell trial selection (unit drift) '''
# Drop events where the unit looks unhealthy BEFORE the tuning curves are
# computed, so each cell gets its own set of events, and its raster shows
# the events its curves were actually built from.
subsample_drift = False
subsample_metric = 'firing rate'   # 'firing rate' | 'amplitude'
subsample_thresh = 0.5             # keep events >= this x the session average
thresh_t_window = 60.0             # seconds centred on the event

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# tuning-curve windows (seconds) relative to event onset / offset
fr_on_start, fr_on_end, timepoints_on = window_frames(-0.5, 1.0, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-1.0, 0.5, dt)

# raster window centered on event offset
event_window = 10                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster + 1) * dt

# ~40 ms Gaussian smoothing for the tuning curves
sigma_frames = fps // 25

# style
cache_color = 'xkcd:orange'
ret_color = 'xkcd:purple'
psth_lw = 3
event_lw = 1
time_int = 2        # x-tick spacing (s) on the rasters
title_size = 14
axis_label = 12

''' Define/create the save folder '''
save_folder = f"{save_figs_dir}/{bird}/cache_activity/"
os.makedirs(save_folder, exist_ok=True)

''' Plot cache responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting cache responsive cells for {bird}_{session_id}')

    ''' Get the file params '''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}/behavior_data/"

    ''' Load and format the neural data '''
    # spikes per video frame
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape

    # session average firing rate (waveform_props is [asymm, width, log10 fr])
    avg_firing_rate = 10 ** data_dict[bird][session_id]['waveform_props'][2]

    # cache-responsiveness from the shuffle test
    barcode_dict = data_dict[bird][session_id].get('barcode_dict', {})
    cache_modulated = barcode_dict.get('cache_modulated', None)

    ''' Label cells by their KS cluster ID '''
    all_ids = cluster_ids_for_session(data_dict, bird, session_id, root_dir)

    ''' Filter cells — one mask applied to every per-cell array '''
    keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw,
                                 **CELL_FILTERS)
    filtered = apply_cell_filter(keep_cells,
                                 spike_fr=spike_fr,
                                 avg_firing_rate=avg_firing_rate,
                                 cache_modulated=cache_modulated)
    spike_fr = filtered['spike_fr']
    avg_firing_rate = filtered['avg_firing_rate']
    cache_modulated = filtered['cache_modulated']
    n_cells = spike_fr.shape[0]
    cell_ids = all_ids[keep_cells]

    if n_cells == 0:
        print('  no cells survived the filters, skipping')
        continue

    ''' Pick the cells to plot '''
    cache_up_bool = cache_modulated == 1
    cache_down_bool = cache_modulated == -1
    print(f'  {np.sum(cache_up_bool)} cache up and '
          f'{np.sum(cache_down_bool)} cache down cells')

    if CELL_SELECTION == 'cache_modulated':
        cells_to_plot = np.where(cache_up_bool | cache_down_bool)[0]
    else:
        cells_to_plot = np.arange(n_cells)
    if cells_to_plot.shape[0] == 0:
        print('  no cells to plot, skipping')
        continue

    ''' Load and format behavior data '''
    seed_struct, count_data = load_behavior_data(data_dir)

    cache_onsets, cache_offsets = get_cache_ints(count_data, seed_struct)
    retrieve_onsets, retrieve_offsets = get_retrieve_ints(count_data, seed_struct)

    n_cache = cache_onsets.shape[0]
    n_ret = retrieve_onsets.shape[0]
    print(f'  {n_cache} caches, {n_ret} retrievals')
    if n_cache == 0 or n_ret == 0:
        print('  skipping')
        continue

    ''' Per-cell event selection, before anything is averaged '''
    cache_keep = ret_keep = None
    if subsample_drift:
        amp_data = None
        if subsample_metric == 'amplitude':
            session_data = data_dict[bird][session_id]
            ks_dir = f"{bird}_{session_data['ephys_id']}/{session_data['ks_folder']}/"
            amp_data = load_spike_amplitudes(session_dir, data_dir, ks_dir,
                                             cell_ids, n_frames, fps=fps)
        cache_keep, ret_keep = [
            select_trials_by_drift(spike_fr, ons, dt, subsample_metric,
                                   subsample_thresh, thresh_t_window,
                                   amp_data=amp_data)
            for ons in (cache_onsets, retrieve_onsets)]
        print(f'  {subsample_metric} cut keeps '
              f'{cache_keep.sum(axis=1).min()}-{cache_keep.sum(axis=1).max()} '
              f'of {n_cache} caches and '
              f'{ret_keep.sum(axis=1).min()}-{ret_keep.sum(axis=1).max()} '
              f'of {n_ret} retrievals per cell')

    ''' Onset- and offset-aligned tuning curves '''
    cache_on_psth, cache_off_psth, _, n_cache_used = event_psth_on_off(
        spike_fr, cache_onsets, cache_offsets,
        (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
        sigma_frames=sigma_frames, keep=cache_keep)

    ret_on_psth, ret_off_psth, _, n_ret_used = event_psth_on_off(
        spike_fr, retrieve_onsets, retrieve_offsets,
        (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
        sigma_frames=sigma_frames, keep=ret_keep)

    print(f'  tuning curves from up to {int(n_cache_used[:, 0].max())} caches '
          f'and {int(n_ret_used[:, 0].max())} retrievals (complete windows only)')

    ''' Plot '''
    # create the figure once and clear it between cells
    f, ax = plt.subplots(2, 4, figsize=(8, 5),
                         gridspec_kw=dict(width_ratios=[1, 0.2, 1, 1],
                                          wspace=0.08, hspace=0.45))
    avg_fr_session = np.round(avg_firing_rate, 2)

    for c_idx in cells_to_plot:
        cell_id = cell_ids[c_idx]

        for row in range(2):
            for col in range(4):
                ax[row, col].cla()

        # ── Cosmetics ──────────────────────────────────────────────────
        for row in range(2):
            for col in [2, 3]:
                ax[row, col].spines['top'].set_visible(False)
                ax[row, col].spines['right'].set_visible(False)
            # the offset tuning curve shares its y scale with the onset one
            ax[row, 3].spines['left'].set_visible(False)
            ax[row, 3].tick_params(labelleft=False)
            # spacer column
            for side in ['top', 'left', 'bottom', 'right']:
                ax[row, 1].spines[side].set_visible(False)
            ax[row, 1].set_xticks([])
            ax[row, 1].set_yticks([])
            ax[row, 1].set_facecolor('none')

        # ── Shared tuning-curve ceiling across all four panels ─────────
        max_fr = shared_ylim(cache_on_psth[c_idx], cache_off_psth[c_idx],
                             ret_on_psth[c_idx], ret_off_psth[c_idx])

        # deterministic within-frame jitter, so a cell looks the same each run
        rng = np.random.default_rng(int(cell_id) * 7919)

        # ── Raster rows: this cell's selected events, ordered by duration ─
        # per cell because the drift cut is; identical for every cell when
        # subsample_drift is off
        raster_specs = []
        n_rows = {}
        for row, (ons, offs, k, color) in enumerate(
                [(cache_onsets, cache_offsets, cache_keep, cache_color),
                 (retrieve_onsets, retrieve_offsets, ret_keep, ret_color)]):
            sel = slice(None) if k is None else k[c_idx]
            c_ons, c_offs = ons[sel], offs[sel]
            order, _ = sort_events_by_duration(c_ons, c_offs)
            # onset tick position for each row, relative to the offset at t=0
            onset_ticks = np.clip(-(c_offs - c_ons)[order] * dt,
                                  raster_t_pts[0], 0)
            n_rows[row] = c_ons.shape[0]
            raster_specs.append((row, c_offs[order], onset_ticks, color,
                                 n_rows[row]))

        for row, align_frames, onset_ticks, color, n_events in raster_specs:
            if n_events == 0:
                ax[row, 0].set_xlim(raster_t_pts[0], raster_t_pts[-1])
                continue
            raster = build_raster(spike_fr[c_idx], align_frames,
                                  fr_halfwidth_raster)
            spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)
            spk_s = raster_marker_size(ax[row, 0], f, n_events)
            ax[row, 0].scatter(spk_t, spk_row, color=color, marker='|',
                               lw=0.6, s=spk_s)
            # offset reference line and per-event onset ticks
            ax[row, 0].vlines(0, -0.5, n_events - 0.5,
                              colors='k', linestyles='dashed', lw=event_lw)
            ax[row, 0].scatter(onset_ticks, np.arange(n_events),
                               color='k', marker='|', lw=event_lw,
                               s=spk_s / 2, zorder=2)

        # ── Tuning curves ──────────────────────────────────────────────
        psth_specs = [
            (0, cache_on_psth[c_idx, 0], cache_off_psth[c_idx, 0], cache_color),
            (1, ret_on_psth[c_idx, 0], ret_off_psth[c_idx, 0], ret_color),
        ]
        for row, on_trace, off_trace, color in psth_specs:
            plot_psth_trace(ax[row, 2], timepoints_on, on_trace, dt, color,
                            sigma_frames=sigma_frames, lw=psth_lw)
            plot_psth_trace(ax[row, 3], timepoints_off, off_trace, dt, color,
                            sigma_frames=sigma_frames, lw=psth_lw)
            for col, t_pts in [(2, timepoints_on), (3, timepoints_off)]:
                ax[row, col].vlines(0, 0, max_fr,
                                    colors='k', linestyles='dashed', lw=event_lw)
                ax[row, col].hlines(avg_fr_session[c_idx], t_pts[0], t_pts[-1],
                                    colors='xkcd:gray', linestyles='dashed',
                                    lw=event_lw)

        # ── Limits & ticks ─────────────────────────────────────────────
        raster_ticks = np.arange(-event_window / 2, event_window / 2 + 0.5, time_int)
        for row, n_events in [(0, n_rows[0]), (1, n_rows[1])]:
            ax[row, 0].set_xlim(raster_t_pts[0], raster_t_pts[-1])
            ax[row, 0].set_ylim(-0.5, max(n_events, 1) - 0.5)
            ax[row, 0].set_xticks(raster_ticks)
            ax[row, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

            # every tuning curve on the same scale
            ax[row, 2].set_xlim(timepoints_on[0], timepoints_on[-1])
            ax[row, 3].set_xlim(timepoints_off[0], timepoints_off[-1])
            ax[row, 2].set_ylim(0, max_fr)
            ax[row, 3].set_ylim(0, max_fr)
            ax[row, 2].set_yticks([0, max_fr])
            ax[row, 3].set_yticks([])

        # ── Axis labels ────────────────────────────────────────────────
        ax[0, 0].set_ylabel(f'caches (n={n_rows[0]})', fontsize=axis_label)
        ax[1, 0].set_ylabel(f'retrievals (n={n_rows[1]})', fontsize=axis_label)
        for row in range(2):
            ax[row, 0].set_xlabel('time from event offset (s)', fontsize=axis_label)
            ax[row, 2].set_xlabel('time from onset (s)', fontsize=axis_label)
            ax[row, 3].set_xlabel('time from offset (s)', fontsize=axis_label)
            ax[row, 2].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        f.suptitle(f'{bird} {session_id}  —   cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=1.02)

        # save into cache_up / cache_down / other by shuffle-test outcome
        if cache_up_bool[c_idx]:
            save_subfolder = f'{save_folder}/cache_up/'
        elif cache_down_bool[c_idx]:
            save_subfolder = f'{save_folder}/cache_down/'
        else:
            save_subfolder = f'{save_folder}/not_modulated/'
        os.makedirs(save_subfolder, exist_ok=True)

        f.savefig(f'{save_subfolder}/{session_id}_cache_ret_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
