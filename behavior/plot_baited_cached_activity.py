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
                        sort_events_by_duration, sort_events_by_time_group, 
                        subsample_for_raster,
                        event_psth_on_off, shared_ylim, raster_marker_size,
                        plot_psth_trace)
from format_behavior_data import (load_behavior_data, get_checks_raw,
                                  get_retrieve_ints,
                                  _seed_provenance_timeline,
                                  get_expectation_status,
                                  SITE_EMPTY, SITE_BAITED, SITE_CACHED,
                                  SITE_UNKNOWN, SITE_STATUS_NAMES)
from spike_amplitudes import (load_spike_amplitudes, trial_amplitudes,
                              select_trials_by_drift, plot_amp_panel)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

'''
Plot check- and retrieval-aligned activity for single cells.

Events are grouped by what the bird should expect to find in the site,
which is what the baited-vs-cached comparison is actually about:

    empty    no seed in the site
    baited   a seed we placed--on the bird's first interaction with the site,
             it has no way to know it will find a seed there
    cached   any seed the bird should expect: its own caches, plus baited seeds
             it has already found

Grouping comes from format_behavior_data.get_expectation_status

The one thing this pooling hides is that a bait the bird has already found
(and not retrieved) is not the same as a seed it cached itself. 
Set discovered_bait='drop' to leave those events out rather than pool them.

Each figure is one cell:
    row 0      checks raster (onset-aligned) | spike amplitude | onset PSTH | offset PSTH
    row 1  retrievals raster (onset-aligned) | spike amplitude | onset PSTH | offset PSTH
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'ROS107'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
# session_list = ['260831']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on or bounded by stim-responsive channels
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.1,       # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
# site interactions longer than this are probably not checks (SC, EM 2024)
max_check_dur = 1.5     # seconds

# a site the bird has already opened once, still holding a bait:
# 'cached' pools it with the bird's own caches, 'drop' leaves it out
discovered_bait = 'cached'

# which pool a retrieval draws from when a site holds both
# 'cached_first' = the bird takes its own seed back before the bait
removal_rule = 'cached_first'

# count the seeds we placed at the start of the session (initSeedCounts).
# False scores within-session caching only and nothing can come back baited.
use_init_counts = True

# a group needs at least this many usable events to get a tuning curve
min_events_per_group = 2

# max events plotted in the raster
max_events_per_group = 100

# sort by duration or chronological?
sort_by_duration = True

# skip plotting a session if any group has fewer than min_events_per_group.
skip_too_few_events = False

''' Per-cell trial selection (unit drift) '''
# Drop events where the unit has stopped firing or drifted from the probe
# before the tuning curves are computed and trials are selected for the raster.
# Set thresh_t_window to wider than the event window to assess general unit 
# properties, not response to the event itself.
subsample_drift = True
subsample_metric = 'firing rate'   # 'firing rate' | 'amplitude'
subsample_thresh = 0.7             # keep events >= this x the session average
thresh_t_window = 5*60.0           # seconds centered on the event

''' Amplitude panel params '''
# mean KS spike amplitude within the event's raster window, 
# normalized to the unit's session median.
show_amp_panel = True
amp_min_spikes = 1          # events with fewer spikes in window are left blank
amp_panel_lw = 0.8

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# tuning-curve windows (seconds) relative to event onset / offset
fr_on_start, fr_on_end, timepoints_on = window_frames(-0.3, 0.3, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-0.3, 0.3, dt)

# raster window centered on event onset
event_window = 1                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster+1) * dt

# Gaussian smoothing for the tuning curves, in frames
# 1 frame = 20 ms at 50 fps; set 0 for no smoothing at all
sigma_frames = 0 # fps // 50

# style — checks green, retrievals purple; grey = baited
# keyed by status code so a missing group can never shift the colours
check_colors = {SITE_EMPTY: 'xkcd:apple green', SITE_BAITED: 'xkcd:dark grey',
                SITE_CACHED: 'xkcd:deep green'}
retrieve_colors = {SITE_EMPTY: 'xkcd:light grey', SITE_BAITED: 'xkcd:dark grey',
                   SITE_CACHED: 'xkcd:purple'}

# names drawn on the figure
plot_names = {SITE_EMPTY: 'empty', SITE_BAITED: 'bait', SITE_CACHED: 'cache'}
# plot_names = {SITE_EMPTY: 'empty', SITE_BAITED: 'unkmnown', SITE_CACHED: 'known'}

psth_lw = 1.5
event_lw = 0.8
divider_lw = 0.8
time_int = 1        # x-tick spacing (s) on the rasters
time_int_tc = 0.1        # x-tick spacing (s) on the tuning curves
title_size = 14
axis_label = 12
legend_size = 8

# colored group labels on the raster: gap in points between the y tick labels
# and the group label, and between the group label and the plain y label
grp_label_size = axis_label
grp_label_gap = 0
ylabel_pad = grp_label_size + 8

# legend of the drawn tuning curves, with the n behind each trace.
# 'right' puts it clear of the axes, 'inside' is the old overlapping version
tc_legend = 'right'      # 'right' | 'inside' | None

''' Which statuses to plot, in raster order (low code first) '''
keep_groups = np.asarray([SITE_EMPTY, SITE_BAITED, SITE_CACHED])


''' Helpers for the colored group labels '''
def _y_label_offset_pts(ax, fig, renderer, gap=0):
    '''
    Points from the y spine out to just past the widest y tick label, so the
    group labels sit as close to the axis as the tick labels allow instead of
    at a guessed fraction of the axes width.
    '''
    widths = [t.get_window_extent(renderer).width for t in ax.get_yticklabels()
              if t.get_visible() and t.get_text()]
    label_w_pts = (max(widths) * 72 / fig.dpi) if widths else 0.0
    tick_pts = (plt.rcParams['ytick.major.size']
                + plt.rcParams['ytick.major.pad'])
    return label_w_pts + tick_pts + gap


def draw_group_labels(fig, ax, renderer, groups_sorted, block_edges, colors,
                      names, fontsize=12, gap=3.0, min_fontsize=7):
    '''
    One coloured label per raster block, always drawn, however thin the block.

    The bottom block's label starts at the bottom of the axes and the top
    block's ends at the top, so the two can never collide; anything in between
    is centred on its own block and then nudged, if it has to be, into whatever
    room the pinned labels leave.  If three names cannot fit stacked at the
    requested size they are all shrunk together, down to min_fontsize.

    Rotated text anchors on the un-rotated alignment, so with
    rotation_mode='anchor' and rotation=90: ha sets where the text sits along y
    ('left' = starts at the anchor and runs up), and va='bottom' keeps the
    glyphs entirely to the left of the anchor.
    '''
    groups_sorted = np.asarray(groups_sorted)
    n_events = groups_sorted.shape[0]
    if n_events == 0:
        return

    edges = np.unique(np.concatenate(
        ([0], np.asarray(list(block_edges), dtype=int), [n_events])))
    blocks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    n_blocks = len(blocks)

    pad = _y_label_offset_pts(ax, fig, renderer, gap=gap)
    y_bottom, y_top = ax.get_ylim()

    anns = []
    for i, (a, b) in enumerate(blocks):
        g_id = int(groups_sorted[a])
        if n_blocks > 1 and i == 0:
            y, ha = y_bottom, 'left'            # runs up from the axes bottom
        elif n_blocks > 1 and i == n_blocks - 1:
            y, ha = y_top, 'right'              # ends at the axes top
        else:
            y, ha = (a + b - 1) / 2, 'center'   # centred on the block
        anns.append(ax.annotate(
            names[g_id], xy=(0, y), xycoords=ax.get_yaxis_transform(),
            xytext=(-pad, 0), textcoords='offset points',
            rotation=90, rotation_mode='anchor', ha=ha, va='bottom',
            color=colors[g_id], fontsize=fontsize, annotation_clip=False))

    if n_blocks == 1:
        return

    # rotated text: the window extent's height is the length of the name.
    # convert to raster rows so the stacking can be done in data coordinates
    inv = ax.transData.inverted()
    def to_rows(px):
        (_, lo), (_, hi) = inv.transform([(0, 0), (0, px)])
        return abs(hi - lo)

    def heights_rows():
        return [to_rows(t.get_window_extent(renderer).height) for t in anns]

    span = abs(y_top - y_bottom)
    gap_rows = to_rows(0.4 * fontsize * fig.dpi / 72)
    heights = heights_rows()

    # names stacked end to end are taller than the raster: shrink them all by
    # the same factor rather than letting the labels sit on top of each other
    needed = sum(heights) + gap_rows * (n_blocks - 1)
    if needed > span:
        shrunk = max(min_fontsize, fontsize * span / needed)
        if shrunk < fontsize:
            for t in anns:
                t.set_fontsize(shrunk)
            gap_rows = to_rows(0.4 * shrunk * fig.dpi / 72)
            heights = heights_rows()

    # keep the middle labels clear of the two pinned ones (a middle block can
    # be a single row tall, so its own centre is not a safe position)
    low = y_bottom + heights[0] + gap_rows           # top of the bottom label
    high = y_top - heights[-1] - gap_rows            # bottom of the top label
    for i in range(1, n_blocks - 1):
        half = heights[i] / 2
        if low + half > high - half:                 # no room: split what is left
            centre = (low + high) / 2
        else:
            centre = min(max(anns[i].xy[1], low + half), high - half)
        anns[i].xy = (0, centre)
        low = centre + half + gap_rows


''' Define/create the save folder '''
if sort_by_duration:
    save_folder = f"{save_figs_dir}/{bird}/baited_cached_activity/"
else:
    save_folder = f"{save_figs_dir}/{bird}/baited_cached_activity/chronological/"
os.makedirs(save_folder, exist_ok=True)

''' Plot check/retrieval responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting check/retrieval responses for {bird}_{session_id}')

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

    # one seed ledger for the whole session, shared by both event types.
    # built with the same settings passed to get_expectation_status, which
    # otherwise ignores them when a timeline is handed in
    timeline = _seed_provenance_timeline(count_data, seed_struct,
                                         use_init_counts=use_init_counts,
                                         removal_rule=removal_rule)

    # get check times + status
    check_onsets, check_offsets, check_site_idx = get_checks_raw(
        count_data, seed_struct, max_check_dur=max_check_dur, dt=dt)
    check_status, check_first = get_expectation_status(
        count_data, seed_struct, check_onsets, check_site_idx,
        timeline=timeline, use_init_counts=use_init_counts,
        removal_rule=removal_rule, discovered_bait=discovered_bait)

    # get retrieval times + status
    retrieve_onsets, retrieve_offsets, retrieve_site_idx = get_retrieve_ints(
        count_data, seed_struct, return_site_idx=True)
    retrieve_status, retrieve_first = get_expectation_status(
        count_data, seed_struct, retrieve_onsets, retrieve_site_idx,
        timeline=timeline, use_init_counts=use_init_counts,
        removal_rule=removal_rule, discovered_bait=discovered_bait)

    events = {
        'check': {'onsets': check_onsets, 'offsets': check_offsets,
                  'status': check_status, 'first': check_first},
        'retrieve': {'onsets': retrieve_onsets, 'offsets': retrieve_offsets,
                     'status': retrieve_status, 'first': retrieve_first},
    }
    for key, label in [('check', 'checks'), ('retrieve', 'retrievals')]:
        st = events[key]['status']
        tally = ', '.join(f'{int(np.sum(st == g))} {SITE_STATUS_NAMES[g]}'
                          for g in [SITE_EMPTY, SITE_BAITED, SITE_CACHED,
                                    SITE_UNKNOWN]
                          if np.any(st == g))
        n_first = int(np.sum(events[key]['first']))
        print(f'  {st.shape[0]} {label} ({tally}); {n_first} at a site the '
              f'bird had not opened before')

    ''' KS spike amplitudes for the side panel '''
    amp_data = None
    if show_amp_panel:
        session_data = data_dict[bird][session_id]
        ks_dir = f"{bird}_{session_data['ephys_id']}/{session_data['ks_folder']}/"
        try:
            amp_data = load_spike_amplitudes(session_dir, data_dir, ks_dir,
                                             cell_ids, n_frames, fps=fps)
        except FileNotFoundError as err:
            print(f'  no KS amplitudes ({err.filename}), skipping the amp panel')

    ''' Pre-compute per-row event ordering and tuning curves '''
    # one entry per subplot row, always, so an event type with nothing usable
    # leaves its row blank instead of shifting the other row up into it
    rows = []
    for key, label, colors in [('check', 'checks', check_colors),
                               ('retrieve', 'retrievals', retrieve_colors)]:
        onsets = events[key]['onsets'].astype(int)
        offsets = events[key]['offsets'].astype(int)
        groups = events[key]['status'].astype(int)   # 0 empty, 1 baited, 2 cached

        blank_row = dict(
            key=key, label=label, colors=colors, n_events_true=0,
            onsets=np.empty(0, dtype=int), offsets=np.empty(0, dtype=int),
            groups=np.empty(0, dtype=int), keep=None, raster_seed=0,
            on_psth=None, off_psth=None,
            group_ids=np.empty(0, dtype=int),
            n_used=np.empty((n_cells, 0), dtype=int),
            enough=np.empty((n_cells, 0), dtype=bool), n_true_by_group={})

        # drop the events we cannot attribute to one group
        keep = np.isin(groups, keep_groups)
        if not np.all(keep):
            dropped = ', '.join(
                f'{int(np.sum(groups == g))} {SITE_STATUS_NAMES[g]}'
                for g in np.unique(groups[~keep]))
            print(f'  dropping {int(np.sum(~keep))} {label} ({dropped})')
        onsets, offsets, groups = onsets[keep], offsets[keep], groups[keep]

        n_events_true = onsets.shape[0]
        if n_events_true == 0:
            print(f'  no usable {label} in this session')
            rows.append(blank_row)
            continue

        # true tallies per group, before any raster subsampling
        n_true_by_group = {int(g): int(np.sum(groups == g))
                           for g in np.unique(groups)}

        # per-cell event selection, before anything is averaged
        trial_keep = None
        if subsample_drift:
            trial_keep = select_trials_by_drift(spike_fr, onsets, dt,
                                          subsample_metric, subsample_thresh,
                                          thresh_t_window, amp_data=amp_data,
                                          min_spikes=amp_min_spikes)
            print(f'  {subsample_metric} cut keeps '
                  f'{trial_keep.sum(axis=1).min()}-'
                  f'{trial_keep.sum(axis=1).max()} of '
                  f'{n_events_true} {label} per cell')

        # tuning curves, from each cell's selected events
        on_psth, off_psth, group_ids, n_used = event_psth_on_off(
            spike_fr, onsets, offsets,
            (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
            groups=groups, sigma_frames=sigma_frames, keep=trial_keep)

        # no tuning curve if too few events.  n_used is per cell now, so a
        # group can be drawn for one cell and dropped for another
        enough = n_used >= min_events_per_group
        for g_idx, g_id in enumerate(group_ids):
            if not np.any(enough[:, g_idx]):
                print(f'  excluding {SITE_STATUS_NAMES[g_id]} {label} from the '
                      f'tuning curves ({int(n_used[:, g_idx].max())} usable '
                      f'events at best, need {min_events_per_group})')

        rows.append(dict(
            key=key, label=label, colors=colors,
            n_events_true=n_events_true, n_true_by_group=n_true_by_group,
            onsets=onsets, offsets=offsets, groups=groups, keep=trial_keep,
            raster_seed=zlib.crc32(f'{bird}_{session_id}_{key}'.encode()) & 0xffffffff,
            on_psth=on_psth, off_psth=off_psth,
            group_ids=group_ids, n_used=n_used, enough=enough,
        ))

    if all(r['n_events_true'] == 0 for r in rows):
        print(f'  skipping {session_id}: no usable events in either row')
        continue

    ''' Optionally skip the session if any group is underpowered '''
    if skip_too_few_events and any(r['n_events_true'] == 0
                                   or not np.all(r['enough'].any(axis=0))
                                   for r in rows):
        print(f'  skipping {session_id}: one or more groups has < {min_events_per_group} events')
        continue

    ''' Plot '''
    # column 0 raster | 1 amplitude | 2 spacer | 3 onset TC | 4 offset TC
    f, ax = plt.subplots(2, 5, figsize=(10, 4),
                         gridspec_kw=dict(width_ratios=[1, 0.28, 0.22, 1, 1],
                                          wspace=0.2, hspace=0.45))

    avg_fr_session = np.round(avg_firing_rate, 2)

    for c_idx in cells_to_plot:
        cell_id = cell_ids[c_idx]

        for row in range(2):
            for col in range(5):
                ax[row, col].cla()
                ax[row, col].set_visible(True)   # cla does not undo a hidden row

        # ── Cosmetics ──────────────────────────────────────────────────
        for row in range(2):
            for col in [3, 4]:
                ax[row, col].spines['top'].set_visible(False)
                ax[row, col].spines['right'].set_visible(False)
            ax[row, 4].spines['left'].set_visible(False)
            ax[row, 4].tick_params(labelleft=False)
            for side in ['top', 'left', 'bottom', 'right']:
                ax[row, 2].spines[side].set_visible(False)
            ax[row, 2].set_xticks([])
            ax[row, 2].set_yticks([])
            ax[row, 2].set_facecolor('none')

        # ── Shared tuning-curve ceiling across all four panels ─────────
        # only groups that actually get drawn contribute to the limit
        psth_for_limit = []
        for r in rows:
            for g_idx in range(r['group_ids'].shape[0]):
                if r['enough'][c_idx, g_idx]:
                    psth_for_limit.append(r['on_psth'][c_idx, g_idx])
                    psth_for_limit.append(r['off_psth'][c_idx, g_idx])
        # every group underpowered: nothing to scale to, fall back to baseline
        max_fr = (shared_ylim(*psth_for_limit) if psth_for_limit
                  else max(float(avg_fr_session[c_idx]), 1.0))

        # ── Raster rows: this cell's selected events, then subsample ────
        # per cell because the drift cut is; the seed is not, so with
        # subsample_drift off every cell shows the same events as before
        draw = []
        for r in rows:
            sel = slice(None) if r['keep'] is None else r['keep'][c_idx]
            r_onsets, r_offsets, r_groups, _, _ = subsample_for_raster(
                r['onsets'][sel], r['offsets'][sel], r['groups'][sel],
                max_per_group=max_events_per_group,
                rng=np.random.default_rng(r['raster_seed']))

            # raster order: seed-source block first, then duration within block
            if sort_by_duration:
                order, block_edges = sort_events_by_duration(
                    r_onsets, r_offsets, r_groups)
            else:
                order, block_edges = sort_events_by_time_group(
                    r_onsets, r_offsets, r_groups)

            # offset tick per row. events running past the raster window get no
            # tick rather than one pinned to the edge, which reads as a real
            # offset there
            durations_s = (r_offsets - r_onsets)[order] * dt
            offset_ticks = np.where(durations_s <= raster_t_pts[-1],
                                    durations_s, np.nan)
            align_frames = r_onsets[order]
            n_events = align_frames.shape[0]

            # per-event amplitudes for THIS cell, in raster row order
            raster_amp = None
            if amp_data is not None and n_events:
                raster_amp = trial_amplitudes(
                    amp_data[c_idx][0], amp_data[c_idx][1], align_frames,
                    fr_halfwidth_raster, min_spikes=amp_min_spikes)[0]

            draw.append(dict(n_events=n_events, align_frames=align_frames,
                             groups_sorted=r_groups[order],
                             offset_ticks=offset_ticks, block_edges=block_edges,
                             raster_amp=raster_amp,
                             n_selected=int(np.asarray(r['onsets'][sel]).shape[0])))

        # ── Shared amplitude x limit, so the two rows are comparable ────
        amp_all = [d['raster_amp'] for d in draw if d['raster_amp'] is not None]
        amp_xmax = None
        if amp_all:
            finite = np.concatenate(amp_all)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                amp_xmax = max(np.ceil(float(finite.max()) * 10) / 10, 0.5)

        rng = np.random.default_rng(int(cell_id) * 7919)

        for row, (r, d) in enumerate(zip(rows, draw)):
            n_events = d['n_events']

            # nothing usable for this event type: hide the whole row rather
            # than draw an empty one, and never shift the other row into it
            if n_events == 0:
                for col in range(5):
                    ax[row, col].set_visible(False)
                continue

            # ── Raster limits first, so the marker size is computed against
            #    the axes the spikes are actually drawn in ────────────────
            ax[row, 0].set_xlim(raster_t_pts[0], raster_t_pts[-1])
            ax[row, 0].set_ylim(-0.5, n_events - 0.5)
            ax[row, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[row, 0].set_xticks(np.arange(-event_window / 2,
                                            event_window / 2 + time_int, time_int))

            # ── Raster, one colour block per group ────────────────────
            raster = build_raster(spike_fr[c_idx], d['align_frames'],
                                  fr_halfwidth_raster)
            spk_s = raster_marker_size(ax[row, 0], f, n_events)
            spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)
            
            # colour each spike by the group of the row it belongs to, so
            # the raster and the tuning curves can never disagree
            spk_groups = d['groups_sorted'][spk_row.astype(int)]
            for g_id in np.unique(d['groups_sorted']):
                sel = spk_groups == g_id
                ax[row, 0].scatter(spk_t[sel], spk_row[sel],
                                   color=r['colors'][int(g_id)], marker='|',
                                   lw=0.6, s=spk_s)

            # onset reference line, offset ticks, group dividers
            ax[row, 0].vlines(0, -0.5, n_events - 0.5,
                              colors='k', linestyles='dashed', lw=event_lw)
            ax[row, 0].scatter(d['offset_ticks'], np.arange(n_events),
                               color='k', marker='|', lw=event_lw,
                               s=spk_s / 2, zorder=2)
            for edge in d['block_edges']:
                ax[row, 0].axhline(edge - 0.5, color='xkcd:gray',
                                   lw=divider_lw, zorder=3)

            # ── Amplitude panel ───────────────────────────────────────
            # same rows, same colours and dividers as the raster beside it
            if d['raster_amp'] is not None:
                plot_amp_panel(ax[row, 1], d['raster_amp'],
                               groups_sorted=d['groups_sorted'],
                               block_edges=d['block_edges'],
                               colors=r['colors'], divider_lw=divider_lw,
                               lw=amp_panel_lw, axis_label=axis_label,
                               xmax=amp_xmax)
                ax[row, 1].set_ylim(ax[row, 0].get_ylim())
            else:
                # no KS amplitudes for this session: drop the panel rather than
                # leave an empty pair of axes beside the raster
                ax[row, 1].set_visible(False)

            # ── Tuning curves, one trace per group ────────────────────
            drawn = []
            for g_idx, g_id in enumerate(r['group_ids']):
                if not r['enough'][c_idx, g_idx]:
                    continue
                g_id = int(g_id)
                plot_psth_trace(ax[row, 3], timepoints_on,
                                r['on_psth'][c_idx, g_idx], dt,
                                r['colors'][g_id], sigma_frames=sigma_frames,
                                lw=psth_lw)
                plot_psth_trace(ax[row, 4], timepoints_off,
                                r['off_psth'][c_idx, g_idx], dt,
                                r['colors'][g_id], sigma_frames=sigma_frames,
                                lw=psth_lw)
                drawn.append((g_id, int(r['n_used'][c_idx, g_idx])))

            for col, t_pts in [(3, timepoints_on), (4, timepoints_off)]:
                ax[row, col].vlines(0, 0, max_fr,
                                    colors='k', linestyles='dashed', lw=event_lw)
                ax[row, col].hlines(avg_fr_session[c_idx], t_pts[0], t_pts[-1],
                                    colors='xkcd:gray', linestyles='dashed',
                                    lw=event_lw)

            # ── Limits & ticks ─────────────────────────────────────────
            on_start_t = timepoints_on[0]
            on_end_t = timepoints_on[-1]+dt
            off_start_t = timepoints_off[0]
            off_end_t = timepoints_off[-1]+dt
            ax[row, 3].set_xlim(on_start_t, on_end_t)
            ax[row, 3].set_xticks(np.arange(on_start_t, on_end_t+time_int_tc, time_int_tc))
            ax[row, 4].set_xlim(off_start_t, off_end_t)
            ax[row, 4].set_xticks(np.arange(off_start_t, off_end_t+time_int_tc, time_int_tc))

            ax[row, 3].set_ylim(0, max_fr)
            ax[row, 4].set_ylim(0, max_fr)
            ax[row, 3].set_yticks([0, max_fr])
            ax[row, 4].set_yticks([])

            # ── Labels ─────────────────────────────────────────────────
            # plain ylabel for the event type + count, parked outside the
            # coloured group labels
            ax[row, 0].set_ylabel(f'{r["label"]} (n={d["n_selected"]})',
                                  fontsize=axis_label, labelpad=ylabel_pad)
            ax[row, 0].set_xlabel('time from onset (s)', fontsize=axis_label)

            # legend of the traces that were actually drawn, with their n
            if tc_legend and drawn:
                handles = [Line2D([0], [0], color=r['colors'][g_id], lw=psth_lw,
                                  label=f'{plot_names[g_id]} (n={n})')
                           for g_id, n in drawn]
                if tc_legend == 'right':
                    # outside the axes, so it cannot sit on the traces or grid
                    ax[row, 4].legend(handles=handles, fontsize=legend_size,
                                      loc='center left', bbox_to_anchor=(1.03, 0.5),
                                      frameon=False, handlelength=1.2,
                                      borderpad=0.1, labelspacing=0.3)
                else:
                    ax[row, 4].legend(handles=handles, fontsize=legend_size,
                                      loc='upper right', frameon=False,
                                      handlelength=1, borderpad=0.1,
                                      labelspacing=0.2)

            ax[row, 3].set_xlabel('time from onset (s)', fontsize=axis_label)
            ax[row, 4].set_xlabel('time from offset (s)', fontsize=axis_label)
            ax[row, 3].set_ylabel('firing rate (Hz)', fontsize=axis_label)

            ax[row, 3].grid(True)
            ax[row, 4].grid(True)

        # ── Colored group labels ──────────────────────────────────────
        # after the row loop: the tick labels have to be final, and one draw
        # then serves both rasters
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        for row, (r, d) in enumerate(zip(rows, draw)):
            if d['n_events']:
                draw_group_labels(f, ax[row, 0], renderer, d['groups_sorted'],
                                  d['block_edges'], r['colors'], plot_names,
                                  fontsize=grp_label_size, gap=grp_label_gap)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=0.95)

        f.savefig(f'{save_folder}/{session_id}_check_retrieve_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
