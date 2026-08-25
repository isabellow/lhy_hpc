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
                        event_psth_on_off, shared_ylim, raster_marker_size)
from format_behavior_data import (load_behavior_data, get_checks_raw,
                                  get_visits_raw, get_site_occupancy)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

'''
Plot check- and visit-aligned activity for single cells, split by whether the
site held a seed at the time (occupied vs. empty).

Each figure is one cell:
    row 0  checks  raster (aligned to offset) | onset PSTH | offset PSTH
    row 1  visits  raster (aligned to offset) | onset PSTH | offset PSTH

Events come from get_checks_raw and get_visits_raw (mirroring
get_cache_ints/get_retrieve_ints): bare interaction bounds, not the
padded/truncated windows of get_checks_refined and get_visits_refined.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'TRQ82'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on or bounded by stim-responsive channels
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.05,       # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
# site interactions longer than this are probably not checks (SC, EM 2024)
max_check_dur = 1.5     # seconds

# a group needs at least this many usable events to get a tuning curve
min_events_per_group = 5

# max events plotted in the raster
max_events_per_group = 100

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# tuning-curve windows (seconds) relative to event onset / offset
fr_on_start, fr_on_end, timepoints_on = window_frames(-0.5, 0.5, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-0.5, 0.5, dt)

# raster window centered on event offset
event_window = 2                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster+1) * dt

# 40 ms Gaussian smoothing for the tuning curves
sigma_frames = fps // 25

# style — checks green, visits grey; darker shade = occupied
# group index is the occupancy flag itself: 0 = empty, 1 = occupied
check_colors = ['xkcd:light green', 'xkcd:forest green']
visit_colors = ['xkcd:grey', 'xkcd:charcoal']
group_names = ['empty', 'occupied']
psth_lw = 3
event_lw = 1
divider_lw = 0.8
time_int = 1        # x-tick spacing (s) on the rasters
time_int_tc = 0.5        # x-tick spacing (s) on the tuning curves
title_size = 14
axis_label = 12
legend_size = 8

''' Define/create the save folder '''
save_folder = f"{save_figs_dir}/{bird}/check_visit_activity/"
os.makedirs(save_folder, exist_ok=True)

''' Plot check/visit responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting check/visit responses for {bird}_{session_id}')

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

    check_onsets, check_offsets, check_site_idx = get_checks_raw(
        count_data, seed_struct, max_check_dur=max_check_dur, dt=dt)
    check_occupied = get_site_occupancy(
        count_data, seed_struct, check_onsets, check_site_idx,
        use_init_counts=True)

    visit_onsets, visit_offsets, visit_site_idx = get_visits_raw(count_data)
    visit_occupied = get_site_occupancy(
        count_data, seed_struct, visit_onsets, visit_site_idx,
        use_init_counts=False)

    events = {
        'check': {'onsets': check_onsets, 'offsets': check_offsets,
                  'occupied': check_occupied},
        'visit': {'onsets': visit_onsets, 'offsets': visit_offsets,
                  'occupied': visit_occupied},
    }
    for key, label in [('check', 'checks'), ('visit', 'visits')]:
        occ = events[key]['occupied']
        print(f'  {occ.shape[0]} {label} '
              f'({int(np.sum(occ))} occupied, {int(np.sum(~occ))} empty)')

    ''' Pre-compute per-row event ordering and tuning curves '''
    rows = []
    for key, label, colors in [('check', 'checks', check_colors),
                               ('visit', 'visits', visit_colors)]:
        onsets = events[key]['onsets'].astype(int)
        offsets = events[key]['offsets'].astype(int)
        groups = events[key]['occupied'].astype(int)   # 0 empty, 1 occupied
        n_events_true = onsets.shape[0]

        # true tallies, from every event, before any raster subsampling
        n_occ_true = int(np.sum(groups == 1))
        n_emp_true = int(np.sum(groups == 0))

        # compute tuning curves from every event
        on_psth, off_psth, group_ids, n_used = event_psth_on_off(
            spike_fr, onsets, offsets,
            (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
            groups=groups, sigma_frames=sigma_frames)

        # no tuning curve if too few events
        enough = n_used >= min_events_per_group
        for g_idx, g_id in enumerate(group_ids):
            if not enough[g_idx]:
                print(f'  excluding {group_names[g_id]} {label} from the '
                      f'tuning curves ({int(n_used[g_idx])} usable events, '
                      f'need {min_events_per_group})')

        # subsample raster if too many events
        raster_seed = zlib.crc32(f'{bird}_{session_id}_{key}'.encode()) & 0xffffffff
        raster_rng = np.random.default_rng(raster_seed)
        r_onsets, r_offsets, r_groups, _, n_total_shown = subsample_for_raster(
            onsets, offsets, groups,
            max_per_group=max_events_per_group, rng=raster_rng)
        n_events = r_onsets.shape[0]
        for g_id, n_full in n_total_shown.items():
            if n_full > max_events_per_group:
                print(f'  {label} group {group_names[g_id]}: showing '
                      f'{max_events_per_group}/{n_full} events in the raster')

        # raster order: occupancy block first, then duration within block
        order, block_edges = sort_events_by_duration(r_onsets, r_offsets, r_groups)
        durations = (r_offsets - r_onsets)[order]
        offset_ticks = np.clip(durations * dt, 0, raster_t_pts[-1])

        rows.append(dict(
            key=key, label=label, colors=colors,
            n_events=n_events,
            n_events_true=n_events_true, n_occ_true=n_occ_true, n_emp_true=n_emp_true,
            n_total_shown=n_total_shown,
            align_frames=r_onsets[order],
            groups_sorted=r_groups[order],
            offset_ticks=offset_ticks,
            block_edges=block_edges,
            on_psth=on_psth, off_psth=off_psth,
            group_ids=group_ids, n_used=n_used, enough=enough,
        ))

    ''' Plot '''
    f, ax = plt.subplots(2, 4, figsize=(8, 4),
                         gridspec_kw=dict(width_ratios=[1, 0.2, 1, 1],
                                          wspace=0.3, hspace=0.45))

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
            ax[row, 3].spines['left'].set_visible(False)
            ax[row, 3].tick_params(labelleft=False)
            for side in ['top', 'left', 'bottom', 'right']:
                ax[row, 1].spines[side].set_visible(False)
            ax[row, 1].set_xticks([])
            ax[row, 1].set_yticks([])
            ax[row, 1].set_facecolor('none')

        # ── Shared tuning-curve ceiling across all four panels ─────────
        # only groups that actually get drawn contribute to the limit
        psth_for_limit = []
        for r in rows:
            for g_idx in range(r['group_ids'].shape[0]):
                if r['enough'][g_idx]:
                    psth_for_limit.append(r['on_psth'][c_idx, g_idx])
                    psth_for_limit.append(r['off_psth'][c_idx, g_idx])
        max_fr = shared_ylim(*psth_for_limit)

        rng = np.random.default_rng(int(cell_id) * 7919)

        for row, r in enumerate(rows):
            n_events = r['n_events']

            # ── Raster, one colour block per occupancy group ───────────
            if n_events:
                raster = build_raster(spike_fr[c_idx], r['align_frames'],
                                      fr_halfwidth_raster)
                spk_s = raster_marker_size(ax[row, 0], f, n_events)
                spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)
                # colour each spike by the group of the row it belongs to, so
                # the raster and the tuning curves can never disagree
                spk_groups = r['groups_sorted'][spk_row.astype(int)]
                for g_id in np.unique(r['groups_sorted']):
                    sel = spk_groups == g_id
                    ax[row, 0].scatter(spk_t[sel], spk_row[sel],
                                       color=r['colors'][g_id], marker='|',
                                       lw=0.6, s=spk_s)

                # offset reference line, onset ticks, occupancy block divider
                ax[row, 0].vlines(0, -0.5, n_events - 0.5,
                                  colors='k', linestyles='dashed', lw=event_lw)
                ax[row, 0].scatter(r['offset_ticks'], np.arange(n_events),
                                   color='k', marker='|', lw=event_lw,
                                   s=spk_s / 2, zorder=2)
                for edge in r['block_edges']:
                    ax[row, 0].axhline(edge - 0.5, color='xkcd:gray',
                                       lw=divider_lw, zorder=3)

            # ── Tuning curves, one trace per occupancy group ───────────
            drawn = []
            for g_idx, g_id in enumerate(r['group_ids']):
                if not r['enough'][g_idx]:
                    continue
                ax[row, 2].plot(timepoints_on, r['on_psth'][c_idx, g_idx],
                                lw=psth_lw, color=r['colors'][g_id])
                ax[row, 3].plot(timepoints_off, r['off_psth'][c_idx, g_idx],
                                lw=psth_lw, color=r['colors'][g_id])
                drawn.append((g_id, int(r['n_used'][g_idx])))

            for col, t_pts in [(2, timepoints_on), (3, timepoints_off)]:
                ax[row, col].vlines(0, 0, max_fr,
                                    colors='k', linestyles='dashed', lw=event_lw)
                ax[row, col].hlines(avg_fr_session[c_idx], t_pts[0], t_pts[-1],
                                    colors='xkcd:gray', linestyles='dashed',
                                    lw=event_lw)

            # ── Limits & ticks ─────────────────────────────────────────
            ax[row, 0].set_xlim(raster_t_pts[0], raster_t_pts[-1])
            ax[row, 0].set_ylim(-0.5, max(n_events, 1) - 0.5)
            ax[row, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[row, 0].set_xticks(np.arange(-event_window / 2,
                                            event_window / 2 + time_int, time_int))
            on_start_t = timepoints_on[0]
            on_end_t = timepoints_on[-1]+dt
            off_start_t = timepoints_off[0]
            off_end_t = timepoints_off[-1]+dt
            ax[row, 2].set_xlim(on_start_t, on_end_t)
            ax[row, 2].set_xticks(np.arange(on_start_t, on_end_t+time_int_tc, time_int_tc))
            ax[row, 3].set_xlim(off_start_t, off_end_t)
            ax[row, 3].set_xticks(np.arange(on_start_t, on_end_t+time_int_tc, time_int_tc))

            ax[row, 2].set_ylim(0, max_fr)
            ax[row, 3].set_ylim(0, max_fr)
            ax[row, 2].set_yticks([0, max_fr])
            ax[row, 3].set_yticks([])

            # ── Labels ─────────────────────────────────────────────────
            # Set a plain ylabel for the main label + count
            ax[row, 0].set_ylabel(f'{r["label"]} (n={r["n_events_true"]})',
                                   fontsize=axis_label, labelpad=15)

            # Add "empty" and "occupied" as separate colored text objects
            empty_color = r['colors'][group_ids[0]]
            ax[row, 0].text(-0.3, 0.17, 'empty',
                             color=empty_color, fontsize=axis_label,
                             ha='center', va='center', rotation=90,
                             transform=ax[row, 0].transAxes)
            if len(group_ids) > 1:
                occupied_color = r['colors'][group_ids[1]]
                ax[row, 0].text(-0.3, 0.8, 'occupied',
                                 color=occupied_color, fontsize=axis_label,
                                 ha='center', va='center', rotation=90,
                                 transform=ax[row, 0].transAxes)

            ax[row, 0].set_xlabel('time from event onset (s)', fontsize=axis_label)
            ax[row, 2].set_xlabel('time from onset (s)', fontsize=axis_label)
            ax[row, 3].set_xlabel('time from offset (s)', fontsize=axis_label)
            ax[row, 2].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=0.95)

        f.savefig(f'{save_folder}/{session_id}_check_visit_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
