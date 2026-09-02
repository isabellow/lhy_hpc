import numpy as np

import os
import sys
import zlib
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from scipy.io import loadmat
from format_waveform_data import cluster_ids_for_session
from cell_filters import filter_cells, apply_cell_filter
from event_psth import (window_frames, build_raster, raster_scatter,
                        sort_events_by_duration, sort_events_by_time_group,
                        subsample_for_raster,
                        event_psth_on_off, shared_ylim, raster_marker_size)
from format_behavior_data import (load_behavior_data, get_beak_site_touches,
                                  get_cache_ints, get_retrieve_ints,
                                  get_checks_raw,
                                  _seed_provenance_timeline,
                                  get_expectation_status,
                                  SITE_EMPTY, SITE_BAITED, SITE_CACHED)
from spike_amplitudes import (load_spike_amplitudes, trial_amplitudes,
                              plot_amp_panel)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

'''
Plot activity aligned to individual beak/cache-flap touches for single cells.

Every touch is its own event here, not the merged "site interaction" that
count_data stores: format_behavior_data.get_beak_site_touches recomputes the
beak-on-cache state from the pose data and returns each contiguous run of it,
so a site interaction the bird broke into three pecks contributes three rows.

Each touch is then labeled by the merged site interaction it falls inside:

    cache               the interaction added a seed
    retrieve            the interaction removed a seed the bird expected
                         (its own cache, or a bait it already found)
    retrieve (bait)     the interaction removed a seed we placed, on the
                        bird's first encounter with that site
    check (occupied)    contents unchanged, site held a seed the bird knows
                        about (its own cache, or a bait it has already found)
    check (empty)       contents unchanged, no seed in the site

Retrievals and checks are both split by what the bird should expect to find
(get_expectation_status), so a bait the bird has already discovered and left
counts the same way whether it's later checked or retrieved.  A touch that
falls inside no interaction, or inside one that is none of the five (a long
no-change interaction, a check of a bait the bird has not found yet), is
dropped and reported.

One figure per cell:
    row 0   onset tuning curve | offset tuning curve | -
    row 1   onset raster       | offset raster       | spike amplitude
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# arena objects, for the beak-on-cache-site hulls
arena_dir = "C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/"
arena_items_file = "arena_items_2.mat"

''' Data params '''
bird = 'LMN86'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
# session_list = ['260825']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
# use_stim_filter keeps only cells on or bounded by stim-responsive channels
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.1,        # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Touch detection params '''
# thresholds for the beak-on-cache state.  None reuses the params saved in
# count_data, so the touches are detected exactly as this session's merged
# interactions were; pass a dict to override (keys as BEAK_TOUCH_PARAMS).
touch_params = None

# drop touches shorter than this (frames).  The median filter in
# get_beak_site_touches has already removed anything briefer than ~3 frames.
min_touch_frames = 1

# slack (frames) when matching a touch to the interaction containing it
assign_tol = 2

''' Event params '''
# site interactions longer than this are probably not checks (SC, EM 2024)
max_check_dur = 1.5     # seconds

# first check of a baited site: not known occupied or empty
# 'drop' leaves it out, 'occupied' pools it with  known seed checks
first_bait_check = 'drop'

# which pool a retrieval draws from when a site holds both baited + cached
# 'cached_first' = the bird takes its own seed back before the bait
removal_rule = 'cached_first'

# count the seeds we placed at the start of the session (initSeedCounts).
# False scores within-session caching only and nothing can come back baited.
use_init_counts = True

# a group needs at least this many usable touches to get a tuning curve
min_events_per_group = 1

# max touches plotted in the raster, per group
max_events_per_group = 100

# sort by duration or chronological?
sort_by_duration = False

# skip plotting a session if any group has fewer than min_events_per_group
skip_too_few_events = False

''' Amplitude panel params '''
# mean KS spike amplitude within the touch's raster window,
# normalized to the unit's session median.
show_amp_panel = True
amp_min_spikes = 1          # touches with fewer spikes in window are left blank
amp_panel_lw = 0.8

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# raster window centered on the alignment event
event_window = 0.6                                    # seconds, total
fr_halfwidth_raster = int((event_window / 2) / dt)   # frames each side
raster_t_pts = np.arange(-fr_halfwidth_raster, fr_halfwidth_raster+1) * dt

# tuning-curve window.  Equal to event_window so each curve sits on the same
# time axis as the raster directly below it; narrow it to zoom the curves in,
# and the two rows then stop lining up (the x tick labels come back to say so).
tc_window = event_window                            # seconds, total
fr_on_start, fr_on_end, timepoints_on = window_frames(-tc_window/2, tc_window/2, dt)
fr_off_start, fr_off_end, timepoints_off = window_frames(-tc_window/2, tc_window/2, dt)
axes_aligned = np.isclose(tc_window, event_window)

# Gaussian smoothing for the tuning curves, in frames
# 1 frame = 20 ms at 50 fps; set 0 for no smoothing at all
sigma_frames = fps // 50

''' Touch groups, in raster order (low code first) '''
(TOUCH_CACHE, TOUCH_RETRIEVE_CACHED, TOUCH_RETRIEVE_BAITED,
 TOUCH_CHECK_OCCUPIED, TOUCH_CHECK_EMPTY) = 0, 1, 2, 3, 4
TOUCH_OTHER = -1

keep_groups = np.asarray([TOUCH_CACHE, TOUCH_RETRIEVE_CACHED,
                          TOUCH_RETRIEVE_BAITED, TOUCH_CHECK_OCCUPIED,
                          TOUCH_CHECK_EMPTY])

# keyed by group code so a missing group can never shift the colours
touch_colors = {TOUCH_CACHE: 'xkcd:orange',
                TOUCH_RETRIEVE_CACHED: 'xkcd:purple',
                TOUCH_RETRIEVE_BAITED: 'xkcd:lavender',
                TOUCH_CHECK_OCCUPIED: 'xkcd:deep green',
                TOUCH_CHECK_EMPTY: 'xkcd:apple green'}

# short names for the rotated labels down the side of the raster
plot_names = {TOUCH_CACHE: 'cache',
              TOUCH_RETRIEVE_CACHED: 'retrieve',
              TOUCH_RETRIEVE_BAITED: 'bait',
              TOUCH_CHECK_OCCUPIED: 'occupied',
              TOUCH_CHECK_EMPTY: 'empty'}

# full names for the tuning-curve legend and the printouts
legend_names = {TOUCH_CACHE: 'cache',
                TOUCH_RETRIEVE_CACHED: 'retrieve (cache)',
                TOUCH_RETRIEVE_BAITED: 'retrieve (bait)',
                TOUCH_CHECK_OCCUPIED: 'check (occupied)',
                TOUCH_CHECK_EMPTY: 'check (empty)',
                TOUCH_OTHER: 'other'}

psth_lw = 2
event_lw = 1
divider_lw = 0.8
time_int = 0.1      # x-tick spacing (s), rasters and tuning curves
title_size = 14
axis_label = 12
legend_size = 8

# coloured group labels on the raster: gap in points between the y tick labels
# and the group label, and between the group label and the plain y label
grp_label_size = axis_label
grp_label_gap = 0
ylabel_pad = grp_label_size + 8

# legend of the drawn tuning curves, with the n behind each trace.
# 'right' puts it clear of the axes, 'inside' is the old overlapping version
tc_legend = 'right'      # 'right' | 'inside' | None


''' Helpers for the colored group labels '''
# TODO these two are copied from plot_baited_cached_activity.py — worth moving
# to utils/event_psth.py rather than keeping two copies in sync
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
    room the pinned labels leave.  If the names cannot fit stacked at the
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


''' Helpers for labeling the touches '''
def classify_site_interactions(count_data, seed_struct, timeline):
    '''
    One group code per merged site interaction.

    Caches come straight from the seed changes; retrievals and checks are both
    split by what the bird should expect to find (get_expectation_status), so
    a bait the bird has already discovered and left counts the same way
    whether it's later checked or retrieved. Anything that is none of the five
    groups — a long no-change interaction, a retrieval the ledger cannot
    attribute, a check of a bait the bird has not found yet — is left as
    TOUCH_OTHER, and the touches inside it are dropped.

    Both statuses are scored over every interaction in one call rather than
    per category, so "the first time the bird opened this site" counts its
    caches too, not just its checks and retrievals.

    Returns
    -------
    starts, ends : int arrays, shape (n_interactions,)
    sites : int array, 0-indexed
    groups : int array of group codes
    '''
    all_starts = np.asarray(count_data['newSite']).astype(int)
    all_ends = np.asarray(count_data['endSite']).astype(int)
    all_sites = np.asarray(count_data['siteNum']).astype(int) - 1  # 1-indexed
    n_int = all_starts.shape[0]
    groups = np.full(n_int, TOUCH_OTHER, dtype=int)
    if n_int == 0:
        e = np.asarray([], dtype=int)
        return e, e, e, e

    # what the bird should expect to find, shared by retrievals and checks
    exp_status, _ = get_expectation_status(
        count_data, seed_struct, all_starts, all_sites, timeline=timeline,
        use_init_counts=use_init_counts, removal_rule=removal_rule,
        discovered_bait='cached')

    # onsets are unique across interactions, so each category's onsets index
    # straight back into the full list
    idx_of = {int(t): i for i, t in enumerate(all_starts)}
    def rows(onsets):
        return np.asarray([idx_of[int(t)] for t in np.asarray(onsets)],
                          dtype=int)

    # caches: the interaction added a seed
    c_on, _ = get_cache_ints(count_data, seed_struct)
    if np.asarray(c_on).shape[0]:
        groups[rows(c_on)] = TOUCH_CACHE

    # retrievals: split by whether the bird expected the seed
    r_on, _ = get_retrieve_ints(count_data, seed_struct)
    if np.asarray(r_on).shape[0]:
        r_idx = rows(r_on)
        st = exp_status[r_idx]
        r_groups = np.full(r_idx.shape[0], TOUCH_OTHER, dtype=int)
        r_groups[st == SITE_CACHED] = TOUCH_RETRIEVE_CACHED
        r_groups[st == SITE_BAITED] = TOUCH_RETRIEVE_BAITED
        # SITE_EMPTY should not happen for a retrieval; stays TOUCH_OTHER
        groups[r_idx] = r_groups
        n_amb = int(np.sum(r_groups == TOUCH_OTHER))
        if n_amb:
            print(f'  {n_amb} retrieval(s) could not be attributed to a seed '
                  'source')

    # checks: split by what the bird should expect to find
    k_on, _, _ = get_checks_raw(count_data, seed_struct,
                               max_check_dur=max_check_dur, dt=dt)
    if k_on.shape[0]:
        k_idx = rows(k_on)
        st = exp_status[k_idx]
        k_groups = np.full(k_idx.shape[0], TOUCH_OTHER, dtype=int)
        k_groups[st == SITE_EMPTY] = TOUCH_CHECK_EMPTY
        k_groups[st == SITE_CACHED] = TOUCH_CHECK_OCCUPIED
        # a bait the bird has not opened before is not a seed it knows about
        if first_bait_check == 'occupied':
            k_groups[st == SITE_BAITED] = TOUCH_CHECK_OCCUPIED
        elif first_bait_check != 'drop':
            raise ValueError("first_bait_check must be 'drop' or 'occupied'")
        groups[k_idx] = k_groups
        n_novel = int(np.sum(st == SITE_BAITED))
        if n_novel and first_bait_check == 'drop':
            print(f'  dropping {n_novel} check(s) of a bait the bird had not '
                  'opened before')

    return all_starts, all_ends, all_sites, groups


def label_touches(touch_onsets, touch_site_idx, int_starts, int_ends,
                  int_sites, int_groups, tol=2):
    '''
    Give each unmerged touch the group of the merged site interaction it falls
    inside: same site and the touch starts within the interaction (with tol
    frames of slack at each end).

    Merged interactions never overlap in time, so the containing one is either
    the last interaction starting at or before the touch or, when the slack is
    what brings them together, the next one.

    Returns groups, int array shape (n_touches,), TOUCH_OTHER where no
    interaction matched.
    '''
    groups = np.full(np.asarray(touch_onsets).shape[0], TOUCH_OTHER, dtype=int)
    if groups.shape[0] == 0 or np.asarray(int_starts).shape[0] == 0:
        return groups

    order = np.argsort(int_starts, kind='stable')
    s_start, s_end = int_starts[order], int_ends[order]
    s_site, s_group = int_sites[order], int_groups[order]
    n_int = s_start.shape[0]

    k = np.searchsorted(s_start, touch_onsets, side='right') - 1
    for i, (t_on, t_site) in enumerate(zip(touch_onsets, touch_site_idx)):
        for cand in (k[i], k[i] + 1):
            if not (0 <= cand < n_int):
                continue
            if s_site[cand] != t_site:
                continue
            if (t_on >= s_start[cand] - tol) and (t_on <= s_end[cand] + tol):
                groups[i] = s_group[cand]
                break
    return groups


''' Define/create the save folder '''
if sort_by_duration:
    save_folder = f"{save_figs_dir}/{bird}/beak_flap_activity/"
else:
    save_folder = f"{save_figs_dir}/{bird}/beak_flap_activity/chronological/"
os.makedirs(save_folder, exist_ok=True)

''' Arena objects, shared by every session '''
arena_data = loadmat(f'{arena_dir}{arena_items_file}', squeeze_me=True)

''' Plot beak/flap touch responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting beak/flap touch responses for {bird}_{session_id}')

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

    # one seed ledger for the whole session, shared by every call to
    # get_expectation_status below (which otherwise ignores use_init_counts
    # and removal_rule when a timeline is handed in)
    timeline = _seed_provenance_timeline(count_data, seed_struct,
                                         use_init_counts=use_init_counts,
                                         removal_rule=removal_rule)

    ''' Individual beak/flap touches, unmerged '''
    pred_date = data_dict[bird][session_id].get('pred_date')
    pred_file = f'{pred_date}_posture_2stage_face.npy' if pred_date else None
    touch_onsets, touch_offsets, touch_site_idx = get_beak_site_touches(
        data_dir, arena_data, pred_file=pred_file, params=touch_params,
        count_data=count_data, min_dur_frames=min_touch_frames)

    if touch_onsets.shape[0] == 0:
        print('  no beak/flap touches detected, skipping')
        continue

    # touches can run past the end of the neural recording
    in_range = touch_offsets < n_frames
    if not np.all(in_range):
        print(f'  dropping {int(np.sum(~in_range))} touch(es) past the last '
              'aligned frame')
        touch_onsets = touch_onsets[in_range]
        touch_offsets = touch_offsets[in_range]
        touch_site_idx = touch_site_idx[in_range]

    ''' Label each touch by the interaction it belongs to '''
    int_starts, int_ends, int_sites, int_groups = classify_site_interactions(
        count_data, seed_struct, timeline)
    groups = label_touches(touch_onsets, touch_site_idx, int_starts, int_ends,
                           int_sites, int_groups, tol=assign_tol)

    n_touches_raw = touch_onsets.shape[0]
    keep = np.isin(groups, keep_groups)
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        print(f'  dropping {n_dropped}/{n_touches_raw} touch(es) that fall '
              'outside the five groups')
    onsets = touch_onsets[keep]
    offsets = touch_offsets[keep]
    groups = groups[keep]

    n_events_true = onsets.shape[0]
    if n_events_true == 0:
        print('  no usable touches in this session, skipping')
        continue

    tally = ', '.join(f'{int(np.sum(groups == g))} {legend_names[g]}'
                      for g in keep_groups if np.any(groups == g))
    print(f'  {n_events_true} touches in {int_starts.shape[0]} site '
          f'interactions ({tally})')

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

    ''' Tuning curves from every touch '''
    on_psth, off_psth, group_ids, n_used = event_psth_on_off(
        spike_fr, onsets, offsets,
        (fr_on_start, fr_on_end), (fr_off_start, fr_off_end), dt,
        groups=groups, sigma_frames=sigma_frames)

    # no tuning curve if too few touches
    enough = n_used >= min_events_per_group
    for g_idx, g_id in enumerate(group_ids):
        if not enough[g_idx]:
            print(f'  excluding {legend_names[int(g_id)]} from the tuning '
                  f'curves ({int(n_used[g_idx])} usable touches, need '
                  f'{min_events_per_group})')

    if skip_too_few_events and not np.all(enough):
        print(f'  skipping {session_id}: a group has < {min_events_per_group} '
              'touches')
        continue

    ''' Raster rows: subsample, then order '''
    raster_seed = zlib.crc32(f'{bird}_{session_id}_beak_flap'.encode()) & 0xffffffff
    raster_rng = np.random.default_rng(raster_seed)
    r_onsets, r_offsets, r_groups, _, n_total_shown = subsample_for_raster(
        onsets, offsets, groups,
        max_per_group=max_events_per_group, rng=raster_rng)
    n_events = r_onsets.shape[0]
    for g_id, n_full in n_total_shown.items():
        if n_full > max_events_per_group:
            print(f'  {legend_names[int(g_id)]}: showing '
                  f'{max_events_per_group}/{n_full} touches in the raster')

    # raster order: group block first, then duration within block
    if sort_by_duration:
        order, block_edges = sort_events_by_duration(r_onsets, r_offsets, r_groups)
    else:
        order, block_edges = sort_events_by_time_group(r_onsets, r_offsets, r_groups)

    align_on = r_onsets[order]
    align_off = r_offsets[order]
    groups_sorted = r_groups[order]

    # the other end of each touch, marked in the raster.  Touches running past
    # the window get no tick rather than one pinned to the edge, which would
    # read as a real boundary there
    durations_s = (align_off - align_on) * dt
    offset_ticks = np.where(durations_s <= raster_t_pts[-1], durations_s, np.nan)
    onset_ticks = np.where(durations_s <= raster_t_pts[-1], -durations_s, np.nan)

    ''' Per-touch amplitudes, ordered so row i here is row i of the raster '''
    raster_amp = None
    if amp_data is not None and n_events:
        raster_amp = np.array([
            trial_amplitudes(amp_data[c][0], amp_data[c][1],
                             align_on, fr_halfwidth_raster,
                             min_spikes=amp_min_spikes)[0]
            for c in range(n_cells)])

    ''' Plot '''
    # row 0 tuning curves | row 1 rasters
    # col 0 onset-aligned | col 1 offset-aligned | col 2 amplitude
    # the height buys room for five group names stacked down the side of the
    # raster: they need about 100 raster rows out of the 123 the axes gives
    # them at this size, and shrink below the requested font size if the axes
    # is made much shorter (the ratio does not depend on the touch count)
    f, ax = plt.subplots(2, 3, figsize=(8.5, 8),
                         gridspec_kw=dict(width_ratios=[1, 1, 0.28],
                                          height_ratios=[0.5, 1],
                                          wspace=0.2, hspace=0.08))

    avg_fr_session = np.round(avg_firing_rate, 2)

    for c_idx in cells_to_plot:
        cell_id = cell_ids[c_idx]

        for row in range(2):
            for col in range(3):
                ax[row, col].cla()
                ax[row, col].set_visible(True)   # cla does not undo a hidden axes

        # ── Cosmetics ──────────────────────────────────────────────────
        for col in [0, 1]:
            ax[0, col].spines['top'].set_visible(False)
            ax[0, col].spines['right'].set_visible(False)
        # offset column shares the y axis of the onset column beside it
        ax[0, 1].spines['left'].set_visible(False)
        ax[0, 1].tick_params(labelleft=False)
        ax[1, 1].tick_params(labelleft=False)
        # no amplitude panel above the rasters
        ax[0, 2].set_visible(False)

        # ── Shared tuning-curve ceiling across both panels ─────────────
        # only groups that actually get drawn contribute to the limit
        psth_for_limit = []
        for g_idx in range(group_ids.shape[0]):
            if enough[g_idx]:
                psth_for_limit.append(on_psth[c_idx, g_idx])
                psth_for_limit.append(off_psth[c_idx, g_idx])
        # every group underpowered: nothing to scale to, fall back to baseline
        max_fr = (shared_ylim(*psth_for_limit) if psth_for_limit
                  else max(float(avg_fr_session[c_idx]), 1.0))

        rng = np.random.default_rng(int(cell_id) * 7919)

        # ── Rasters, one colour block per group ───────────────────────
        for col, (align_frames, ticks) in enumerate(
                [(align_on, offset_ticks), (align_off, onset_ticks)]):
            # limits first, so the marker size is computed against the axes
            # the spikes are actually drawn in
            ax[1, col].set_xlim(raster_t_pts[0], raster_t_pts[-1])
            ax[1, col].set_ylim(-0.5, n_events - 0.5)
            ax[1, col].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1, col].set_xticks(np.arange(-event_window / 2,
                                            event_window / 2 + time_int, time_int))

            raster = build_raster(spike_fr[c_idx], align_frames,
                                  fr_halfwidth_raster)
            spk_s = raster_marker_size(ax[1, col], f, n_events)
            spk_t, spk_row = raster_scatter(raster, raster_t_pts, dt, rng=rng)

            # colour each spike by the group of the row it belongs to, so the
            # raster and the tuning curves can never disagree
            spk_groups = groups_sorted[spk_row.astype(int)]
            for g_id in np.unique(groups_sorted):
                sel = spk_groups == g_id
                ax[1, col].scatter(spk_t[sel], spk_row[sel],
                                   color=touch_colors[int(g_id)], marker='|',
                                   lw=0.6, s=spk_s)

            # alignment line, the other end of each touch, group dividers
            ax[1, col].vlines(0, -0.5, n_events - 0.5,
                              colors='k', linestyles='dashed', lw=event_lw)
            ax[1, col].scatter(ticks, np.arange(n_events),
                               color='k', marker='|', lw=event_lw,
                               s=spk_s / 2, zorder=2)
            for edge in block_edges:
                ax[1, col].axhline(edge - 0.5, color='xkcd:gray',
                                   lw=divider_lw, zorder=3)

        ax[1, 0].set_xlabel('time from touch onset (s)', fontsize=axis_label)
        ax[1, 1].set_xlabel('time from touch offset (s)', fontsize=axis_label)

        # plain ylabel for the event type + count, parked outside the coloured
        # group labels
        ax[1, 0].set_ylabel(f'beak/flap touches (n={n_events_true})',
                            fontsize=axis_label, labelpad=ylabel_pad)

        # ── Amplitude panel ───────────────────────────────────────────
        # same rows, same colours and dividers as the rasters beside it
        if raster_amp is not None:
            finite = raster_amp[c_idx][np.isfinite(raster_amp[c_idx])]
            amp_xmax = (max(np.ceil(float(finite.max()) * 10) / 10, 0.5)
                        if finite.size else None)
            plot_amp_panel(ax[1, 2], raster_amp[c_idx],
                           groups_sorted=groups_sorted,
                           block_edges=block_edges,
                           colors=touch_colors, divider_lw=divider_lw,
                           lw=amp_panel_lw, axis_label=axis_label,
                           xmax=amp_xmax)
            ax[1, 2].set_ylim(ax[1, 0].get_ylim())
        else:
            # no KS amplitudes for this session: drop the panel rather than
            # leave an empty pair of axes beside the raster
            ax[1, 2].set_visible(False)

        # ── Tuning curves, one trace per group ────────────────────────
        drawn = []
        for g_idx, g_id in enumerate(group_ids):
            if not enough[g_idx]:
                continue
            g_id = int(g_id)
            ax[0, 0].plot(timepoints_on, on_psth[c_idx, g_idx],
                          lw=psth_lw, color=touch_colors[g_id])
            ax[0, 1].plot(timepoints_off, off_psth[c_idx, g_idx],
                          lw=psth_lw, color=touch_colors[g_id])
            drawn.append((g_id, int(n_used[g_idx])))

        for col, t_pts in [(0, timepoints_on), (1, timepoints_off)]:
            ax[0, col].vlines(0, 0, max_fr,
                              colors='k', linestyles='dashed', lw=event_lw)
            ax[0, col].hlines(avg_fr_session[c_idx], t_pts[0], t_pts[-1],
                              colors='xkcd:gray', linestyles='dashed',
                              lw=event_lw)

        # ── Limits & ticks ────────────────────────────────────────────
        on_start_t, on_end_t = timepoints_on[0], timepoints_on[-1] + dt
        off_start_t, off_end_t = timepoints_off[0], timepoints_off[-1] + dt
        ax[0, 0].set_xlim(on_start_t, on_end_t)
        ax[0, 1].set_xlim(off_start_t, off_end_t)
        ax[0, 0].set_xticks(np.arange(on_start_t, on_end_t + time_int, time_int))
        ax[0, 1].set_xticks(np.arange(off_start_t, off_end_t + time_int, time_int))

        ax[0, 0].set_ylim(0, max_fr)
        ax[0, 1].set_ylim(0, max_fr)
        ax[0, 0].set_yticks([0, max_fr])
        ax[0, 1].set_yticks([])
        ax[0, 0].set_ylabel('firing rate (Hz)', fontsize=axis_label)

        # curves sit on the same time axis as the raster below, so the x tick
        # labels only go on the bottom row.  If tc_window has been narrowed the
        # columns no longer line up and the labels come back to say so
        if axes_aligned:
            ax[0, 0].tick_params(labelbottom=False)
            ax[0, 1].tick_params(labelbottom=False)
        else:
            ax[0, 0].set_xlabel('time from touch onset (s)', fontsize=axis_label)
            ax[0, 1].set_xlabel('time from touch offset (s)', fontsize=axis_label)

        ax[0, 0].grid(True)
        ax[0, 1].grid(True)

        # legend of the traces that were actually drawn, with their n.
        # 'right' parks it in the empty amplitude column of the top row
        if tc_legend and drawn:
            handles = [Line2D([0], [0], color=touch_colors[g_id], lw=psth_lw,
                              label=f'{legend_names[g_id]} (n={n})')
                       for g_id, n in drawn]
            if tc_legend == 'right':
                ax[0, 1].legend(handles=handles, fontsize=legend_size,
                                loc='center left', bbox_to_anchor=(1.03, 0.5),
                                frameon=False, handlelength=1.2,
                                borderpad=0.1, labelspacing=0.3)
            else:
                ax[0, 1].legend(handles=handles, fontsize=legend_size,
                                loc='upper right', frameon=False,
                                handlelength=1, borderpad=0.1,
                                labelspacing=0.2)

        # ── Colored group labels ──────────────────────────────────────
        # after the panels: the tick labels have to be final first
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        draw_group_labels(f, ax[1, 0], renderer, groups_sorted, block_edges,
                          touch_colors, plot_names, fontsize=grp_label_size,
                          gap=grp_label_gap)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr_session[c_idx]} Hz)',
                   fontsize=title_size, y=0.95)

        f.savefig(f'{save_folder}/{session_id}_beak_flap_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
