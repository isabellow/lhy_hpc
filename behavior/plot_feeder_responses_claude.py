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
                        shared_ylim, raster_marker_size)
from format_behavior_data import (load_behavior_data, get_feeder_ints,
                                  get_feeder_periods, classify_feeder_ints,
                                  get_feeder_departure_bounds, get_foot_angle)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

'''
Plot feeder-visit-aligned activity for single cells, split by feeder identity.

Each figure is one cell:
    row 0  OPEN feeder visits    raster | onset PSTH | offset PSTH
    row 1  CLOSED feeder visits  raster | onset PSTH | offset PSTH

All four tuning curves share a y axis, so the response to an open feeder can
be compared directly against the same feeder when closed.

Rasters are aligned to visit offset (departure) and sorted by feeder identity,
then by duration within feeder. Each feeder gets one colour, used for both the
raster block and its tuning curves.

Visits that spanned an open/close transition (feeder_status 0.5) are excluded.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"
posture_file = 'posture_pos_smooth.npy'

''' Data params '''
bird = 'TRQ82'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on/beside stim-responsive channels. It needs
# stim_resp_idx_ch, nucleus_dvs, cell_pos and shank_A_idx, none of which the LHy
# pipeline writes, so leave it False for these recordings. Sessions without the
# stim keys print a note and are left unfiltered rather than raising.
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.05,       # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

# which cells get a figure:
#   'feeder_modulated' — cells whose feeder tuning curve leaves the band set by
#                        firing_up / firing_down around their session baseline
#   'all'              — every cell that survives CELL_FILTERS
CELL_SELECTION = 'feeder_modulated'

''' Feeder modulation thresholds '''
firing_up = 2       # x baseline; >= this counts as elevated
firing_down = 2     # / baseline; <= this counts as reduced
min_spikes_per_visit = 1
# score modulation on 'open', 'closed', or 'both' sets of visits
modulation_from = 'open'

''' Event params '''
# number of usable events needed for a group to get a tuning curve
min_visits_per_feeder = 5

# beak touching feeder vs. feet on feeder perch
use_beak = False

# refine visit offsets using the angle between the feet
align_to_feet = False
angle_thresh = 20   # degrees

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# tuning-curve windows (seconds) relative to visit arrival / departure
fr_on_start, fr_on_end, timepoints_on = window_frames(-1.0, 0.5, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-0.5, 1.0, dt)

# drop visits shorter than the combined window
min_visit_frames = (fr_on_end - fr_on_start)

# raster window centered on visit offset
event_window = 20                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster + 1) * dt

# ~100 ms Gaussian smoothing for the tuning curves
sigma_frames = fps // 10

# style — one color per feeder, shared by its raster block and tuning curves
feeder_colors = ['xkcd:saffron', 'xkcd:green', 'xkcd:scarlet', 'xkcd:blue']
psth_lw = 3
event_lw = 1
divider_lw = 0.8
time_int = 5        # x-tick spacing (s) on the rasters
title_size = 14
axis_label = 12
legend_size = 8

''' Define/create the save folder '''
save_folder = f"{save_figs_dir}/{bird}/feeder_responses_claude/"
os.makedirs(save_folder, exist_ok=True)


''' Plot feeder responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting feeder responsive cells for {bird}_{session_id}')

    ''' Get the file params '''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}/behavior_data/"

    ''' Load and format the neural data '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape

    # session average firing rate (waveform_props is [asymm, width, log10 fr])
    avg_firing_rate = 10 ** data_dict[bird][session_id]['waveform_props'][2]

    # get cluster labels
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

    ''' Load and format behavior data '''
    seed_struct, count_data = load_behavior_data(data_dir)

    # get feeder interactions
    feeder_int_start, feeder_int_end, feeder_ids = get_feeder_ints(count_data, use_beak=use_beak)

    # optionally move the departure time to when the bird turns
    if align_to_feet:
        feet_angle = get_foot_angle(data_dir, posture_file)
        depart_start, depart_end, _ = get_feeder_departure_bounds(count_data)
        assert feeder_int_end.shape[0] == depart_end.shape[0]
        for i, (start_t, end_t) in enumerate(zip(depart_start, depart_end)):
            these_angles = np.degrees(feet_angle[start_t:end_t])
            if np.any(these_angles >= angle_thresh):
                leave_idx = np.argmax(these_angles >= angle_thresh)
                feeder_int_end[i] = np.min([start_t + leave_idx, end_t])

    # get feeder_status: 1 (open throughout), 0 (closed throughout), or 0.5 (status changed mid-visit)
    feeder_open_times, feeder_close_times = get_feeder_periods(session_info_file, bird, session_id)
    feeder_status = classify_feeder_ints(feeder_int_start, feeder_int_end,
                                         feeder_open_times, feeder_close_times)
    n_partial = int(np.sum(feeder_status == 0.5))
    if n_partial:
        print(f'  excluding {n_partial} visits that spanned an open/close transition')

    ''' Pre-compute per-row event ordering and tuning curves '''
    rows = []
    for status, label in [(1, 'open feeder visits'), (0, 'closed feeder visits')]:
        sel = feeder_status == status
        onsets = feeder_int_start[sel]
        offsets = feeder_int_end[sel]
        groups = feeder_ids[sel]
        n_events = onsets.shape[0]

        if n_events == 0:
            rows.append(dict(label=label, n_events=0, group_ids=np.asarray([]),
                             enough=np.asarray([], dtype=bool)))
            print(f'  no {label}')
            continue

        # raster order: feeder block first, then duration within feeder
        order, block_edges = sort_events_by_duration(onsets, offsets, groups)
        durations = (offsets - onsets)[order]
        onset_ticks = np.clip(-durations * dt, raster_t_pts[0], 0)

        on_psth, off_psth, group_ids, n_used = event_psth_on_off(
            spike_fr, onsets, offsets,
            (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
            groups=groups, sigma_frames=sigma_frames,
            min_duration=min_visit_frames)

        enough = n_used >= min_visits_per_feeder
        for g_idx, g_id in enumerate(group_ids):
            if not enough[g_idx]:
                print(f'  excluding feeder {g_id} from the {label} tuning '
                      f'curves ({int(n_used[g_idx])} usable visits, '
                      f'need {min_visits_per_feeder})')

        rows.append(dict(
            label=label, n_events=n_events,
            align_frames=offsets[order],
            groups_sorted=groups[order],
            onset_ticks=onset_ticks,
            block_edges=block_edges,
            on_psth=on_psth, off_psth=off_psth,
            group_ids=group_ids, n_used=n_used, enough=enough,
        ))

    ''' Pick the cells to plot '''
    if CELL_SELECTION == 'feeder_modulated':
        if modulation_from == 'open':
            score_rows = [rows[0]]
        elif modulation_from == 'closed':
            score_rows = [rows[1]]
        else:
            score_rows = rows

        cells_to_plot = []
        for c_idx in range(n_cells):
            baseline = avg_firing_rate[c_idx]
            up_thresh = baseline * firing_up
            down_thresh = baseline / firing_down
            modulated = False
            active = False
            for r in score_rows:
                if r['n_events'] == 0 or not np.any(r['enough']):
                    continue
                traces = np.concatenate(
                    [r['on_psth'][c_idx, r['enough']].ravel(),
                     r['off_psth'][c_idx, r['enough']].ravel()])
                if (traces >= up_thresh).any() or (traces <= down_thresh).any():
                    modulated = True
                # enough spikes in the raster window to trust the curve
                raster = build_raster(spike_fr[c_idx], r['align_frames'],
                                      fr_halfwidth_raster)
                if raster.sum() >= min_spikes_per_visit * r['n_events']:
                    active = True
            if modulated and active:
                cells_to_plot.append(c_idx)
        cells_to_plot = np.asarray(cells_to_plot, dtype=int)
    else:
        cells_to_plot = np.arange(n_cells)

    print(f'  plotting {cells_to_plot.shape[0]}/{n_cells} cells')
    if cells_to_plot.shape[0] == 0:
        continue

    ''' Plot '''
    # 3 visible panels per row (raster | onset PSTH | offset PSTH); column 1 is
    # a spacer so the rasters and tuning curves are not crowded together
    f, ax = plt.subplots(2, 4, figsize=(9, 5.5),
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
            ax[row, 3].spines['left'].set_visible(False)
            ax[row, 3].tick_params(labelleft=False)
            for side in ['top', 'left', 'bottom', 'right']:
                ax[row, 1].spines[side].set_visible(False)
            ax[row, 1].set_xticks([])
            ax[row, 1].set_yticks([])
            ax[row, 1].set_facecolor('none')

        # ── Shared tuning-curve ceiling across all four panels ─────────
        psth_for_limit = []
        for r in rows:
            if r['n_events'] == 0:
                continue
            for g_idx in range(r['group_ids'].shape[0]):
                if r['enough'][g_idx]:
                    psth_for_limit.append(r['on_psth'][c_idx, g_idx])
                    psth_for_limit.append(r['off_psth'][c_idx, g_idx])
        max_fr = shared_ylim(*psth_for_limit)

        rng = np.random.default_rng(int(cell_id) * 7919)

        for row, r in enumerate(rows):
            n_events = r['n_events']

            if n_events:
                # ── Raster, one colour block per feeder ────────────────
                raster = build_raster(spike_fr[c_idx], r['align_frames'],
                                      fr_halfwidth_raster)
                spk_s = raster_marker_size(ax[row, 0], f, n_events)
                spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)
                spk_groups = r['groups_sorted'][spk_row.astype(int)]
                for g_id in np.unique(r['groups_sorted']):
                    sel = spk_groups == g_id
                    ax[row, 0].scatter(spk_t[sel], spk_row[sel],
                                       color=feeder_colors[g_id - 1],
                                       marker='|', lw=0.6, s=spk_s)

                ax[row, 0].vlines(0, -0.5, n_events - 0.5,
                                  colors='k', linestyles='dashed', lw=event_lw)
                ax[row, 0].scatter(r['onset_ticks'], np.arange(n_events),
                                   color='k', marker='|', lw=event_lw,
                                   s=spk_s / 2, zorder=2)
                for edge in r['block_edges']:
                    ax[row, 0].axhline(edge - 0.5, color='xkcd:gray',
                                       lw=divider_lw, zorder=3)

                # ── Tuning curves, one trace per feeder ────────────────
                drawn = []
                for g_idx, g_id in enumerate(r['group_ids']):
                    if not r['enough'][g_idx]:
                        continue
                    ax[row, 2].plot(timepoints_on, r['on_psth'][c_idx, g_idx],
                                    lw=psth_lw, color=feeder_colors[g_id - 1])
                    ax[row, 3].plot(timepoints_off, r['off_psth'][c_idx, g_idx],
                                    lw=psth_lw, color=feeder_colors[g_id - 1])
                    drawn.append((g_id, int(r['n_used'][g_idx])))

                # if drawn:
                #     handles = [Line2D([0], [0], color=feeder_colors[g_id - 1],
                #                       lw=psth_lw, label=f'feeder {g_id} (n={n})')
                #                for g_id, n in drawn]
                #     ax[row, 3].legend(handles=handles, fontsize=legend_size,
                #                       frameon=False, loc='best')
            else:
                ax[row, 0].text(0.5, 0.5, f'no {r["label"]}',
                                ha='center', va='center',
                                transform=ax[row, 0].transAxes,
                                fontsize=axis_label, color='xkcd:grey')

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
                                            event_window / 2 + 0.5, time_int))
            ax[row, 2].set_xlim(timepoints_on[0], timepoints_on[-1])
            ax[row, 3].set_xlim(timepoints_off[0], timepoints_off[-1])
            ax[row, 2].set_ylim(0, max_fr)
            ax[row, 3].set_ylim(0, max_fr)
            ax[row, 2].set_yticks([0, max_fr])
            ax[row, 3].set_yticks([])

            # ── Labels ─────────────────────────────────────────────────
            ax[row, 0].set_ylabel(f'{r["label"]}', fontsize=axis_label)
            ax[row, 0].set_xlabel('time from departure (s)', fontsize=axis_label)
            ax[row, 2].set_xlabel('time from arrival (s)', fontsize=axis_label)
            ax[row, 3].set_xlabel('time from departure (s)', fontsize=axis_label)
            ax[row, 2].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=0.95)

        f.savefig(f'{save_folder}/{session_id}_feeder_tuning_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
