import numpy as np
import pandas as pd

import os
import sys
import warnings
import zlib
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import cluster_ids_for_session, pop_normalize
from cell_filters import filter_cells, apply_cell_filter
from event_psth import window_frames, event_psth_on_off
from format_behavior_data import (load_behavior_data, get_checks_raw,
                                  get_retrieve_ints,
                                  _seed_provenance_timeline,
                                  get_expectation_status,
                                  SITE_EMPTY, SITE_BAITED, SITE_CACHED,
                                  SITE_UNKNOWN, SITE_STATUS_NAMES)
from spike_amplitudes import load_spike_amplitudes, select_trials_by_drift

import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy.ndimage import gaussian_filter1d

'''
Population summary of check- and retrieval-aligned activity.

One heatmap per condition, every good cell a row, time on x:

                      left                     right
    row 0  checks     occupied (seed expected) empty
    row 1  retrievals cached   (seed expected) baited (seed not expected)

Occupied/cached use the same expectation-based grouping as
plot_baited_cached_activity.py (format_behavior_data.get_expectation_status):
"cached" is any seed the bird should expect, i.e. its own caches plus baits it
has already found in any earlier interaction with the site.  Checks of a
not-yet-discovered bait are rare and are dropped by default (check_baited).

Figures are made per session, per bird, and for all birds (plot_levels).
Each figure is kept minimal; a matching *_legend.txt is written beside it with
everything needed for a figure legend (definitions, normalization, sort, block
order, and cell / event / session counts per panel).

Pipeline
--------
1. per session: filter cells, normalize each cell's whole-session trace with
   format_waveform_data.pop_normalize (as in plot_summary_feeder_responses),
   score events, then average the normalized trace around each condition's
   events at onset and offset alignment, plus split-half averages for the
   peak sort.  Cached to disk.
2. pool every session into one cell table, attaching probe positions and the
   saved rastermap order (both read fresh each run, not cached).
3. per figure: pick cells (missing-event rules), sort, plot.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
cache_file = f"{root_dir}check_retrieve_summary_cache.npy"

''' Which data to include '''
birds_to_plot = None           # None = every bird in data_dict, or a list
exclude_birds = ['LMN86']             # e.g. ['LIM63']
exclude_sessions = {}          # e.g. {'ROS107': ['260831']}
plot_levels = ('bird', 'all')

''' Data params '''
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in (same as the single-cell code) '''
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.1,        # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
max_check_dur = 1.5            # s; longer unchanged-content visits are not checks
discovered_bait = 'cached'     # 'cached' | 'drop'  (see get_expectation_status)
removal_rule = 'cached_first'  # 'cached_first' | 'baited_first'
use_init_counts = True

# checks of a bait the bird has not found yet (rare):
# 'drop' leaves them out, 'occupied' pools them into the occupied-check column
check_baited = 'drop'

# a cell needs at least this many usable events for a condition to be drawn
min_events_per_group = 3

# equalize event counts between conditions, per cell, by randomly dropping
# events from the larger condition(s) after the drift cut:
#   None    no matching
#   'rows'  occupied vs empty checks, and cached vs baited retrievals
#   a list of tuples, e.g. [('check_occ', 'check_empty')]
match_events = None
match_seed = 0

''' Per-cell trial selection (unit drift) — same as the single-cell code '''
subsample_drift = True
subsample_metric = 'amplitude'   # 'firing rate' | 'amplitude'
subsample_thresh = 0.7             # keep events >= this x the session average
thresh_t_window = 1 * 60.0         # s centered on the event
amp_min_spikes = 1

''' PSTH windows '''
align_to = ('onset', 'offset')     # one figure per entry
psth_windows = {'onset': (-1.0, 1.0),   # s relative to event onset
                'offset': (-1.0, 1.0)}  # s relative to event offset
sigma_frames = 1                   # Gaussian smoothing (frames), 0 = none

''' Normalization '''
# 'pop_normalize'  format_waveform_data.pop_normalize on each cell's whole-
#                  session trace before any event averaging, as in the feeder
#                  summary: instantaneous rate / (session SD + std_reg), minus
#                  a running-average baseline.  Units are SDs, so 0 = baseline
#                  and both increases and decreases show up.  Nothing about it
#                  depends on the events, so conditions stay comparable.
# None             raw firing rate (Hz)
norm_method = 'pop_normalize'
pop_norm_kw = dict(std_reg=1e-2, baseline_window=30)   # pop_normalize defaults,
                                                       # same as the feeder summary

''' Sorting the y axis '''
# 'depth'      blocks of (bird, shank), each sorted by DV.  Positions are the
#              'cell_pos' / 'shank_idx' saved by get_probe_coords_lhy
# 'rastermap'  blocks of (bird, session), each in the order saved by the
#              rastermap GUI (<data_dir>/<rastermap_file>, key 'isort')
# 'peak'       time of each cell's peak in one condition, cross-validated:
#              peaks come from one half of that condition's events and the
#              other half is what gets plotted for it (see peak_sort)
# 'labels'     user labels (cell_labels), then label_secondary_sort within label
# 'response'   mean rate in response_sort['window'] (optionally a difference)
# 'none'       pooling order
sort_by = 'peak'

# depth: order of the (bird, shank) blocks, and of cells within each block
shank_order = 'ap'                 # 'ap' = by mean shank AP across all birds,
                                   #        as in plot_fr_vs_depth | 'bird'
ap_order = 'posterior_first'       # 'posterior_first' | 'anterior_first'
depth_order = 'dorsal_first'      # 'ventral_first' | 'dorsal_first'
                                   # (DV = depth below surface, larger = ventral)

# rastermap: file written by the rastermap GUI into each session's data_dir.
# isort indexes the rows of aligned_spikes.npy, before the cell filter
rastermap_file = 'aligned_spikes_embedding.npy'

# peak sort
peak_sort = dict(
    cond='check_empty',       # condition whose peak orders the cells
    align=None,               # None = each figure's own alignment; 'onset' or
                              # 'offset' fixes it so both figures share one order
    window=None,              # (t0, t1) s to search for the peak; None = whole PSTH
    metric='max',             # 'max' peak | 'min' trough |
                              # 'absdev' largest deviation from the cell's mean
    split='alternate',        # 'alternate' = odd/even events in time order
                              # (balanced against slow drift) | 'random' |
                              # 'all' = sort on all events and plot all of them
                              # (NOT cross-validated; for visualization only)
    seed=0,                   # 'random' only
    swap_halves=False,        # False: half 0 sorts, half 1 is plotted
    extra_sigma_frames=0,     # extra smoothing of the sorting half before the argmax
)

# user-defined cluster labels, either
#   - a path to a csv with columns bird, session, cluster_id, label, or
#   - a flat array, one label per cell in cell_table.csv order (every cell
#     that passes CELL_FILTERS, before any missing-event exclusion)
cell_labels = None
label_order = None                 # None = sorted, or a list giving block order
label_secondary_sort = 'depth'     # 'depth' | 'rastermap' | 'peak' | 'response' | None

# 'response' sort: cond minus (optional) other cond, window relative to alignment
response_sort = dict(cond='ret_cached', minus='ret_baited', window=(0.0, 0.5))

# peak / response / none only: keep each bird's (or session's) cells together.
# depth and rastermap define their own blocks and ignore this
block_by = None                    # None | 'bird' | 'session'

''' Missing events '''
# single-session figures: a condition with < min_events_per_group events
# 'skip'   don't make that session's figure
# 'blank'  make it, leaving that subplot blank
session_missing = 'blank'

# pooled (bird / all) figures, when some sessions are short of a condition
# 'exclude_sessions'  drop those sessions (and any cell missing a condition
#                     after the drift cut), so all four subplots show the same
#                     cells in the same rows
# 'blank_rows'        keep every cell in the same row of every subplot; rows a
#                     cell has no data for are drawn in nan_color
# 'unaligned'         each subplot shows only the cells that have data for it,
#                     so rows no longer line up across subplots
pooled_missing = 'exclude_sessions'
resort_unaligned = False           # 'unaligned' only: sort each subplot on its own

''' Plotting params '''
clim = (-1, 1)                     # as in the feeder summary; None = auto
                                   # (symmetric 99th pct of |SD|, or 0-99th pct Hz)
cmap_norm = 'bwr'                  # diverging: blue below baseline, red above
cmap_raw = 'viridis'               # norm_method=None
nan_color = 'xkcd:light grey'
show_block_dividers = False         # solid between outer blocks, dashed within
show_block_labels = False          # block names beside the left column
show_median_duration = False       # dotted line at the median event offset/onset
show_suptitle = True
write_legend = True                # *_legend.txt beside every figure
fig_size = (6, 7)
xtick_step = 0.5                   # s

title_size = 13
axis_label = 11
tick_label = 9
block_label_size = 7

# titles in the same colours as the single-cell figures
cond_colors = {'check_occ': 'xkcd:deep green', 'check_empty': 'xkcd:apple green',
               'ret_cached': 'xkcd:purple', 'ret_baited': 'xkcd:dark grey'}

CACHE_VERSION = 5                  # bump when compute_session's output changes


''' Conditions and layout '''
CONDS = {
    'check_occ':   dict(event='check',    statuses=(SITE_CACHED,), name='occupied'),
    'check_empty': dict(event='check',    statuses=(SITE_EMPTY,),  name='empty'),
    'ret_cached':  dict(event='retrieve', statuses=(SITE_CACHED,), name='cached'),
    'ret_baited':  dict(event='retrieve', statuses=(SITE_BAITED,), name='baited'),
}
if peak_sort['cond'] not in CONDS:
    raise ValueError(f"unknown peak_sort cond {peak_sort['cond']}")
if check_baited == 'occupied':
    CONDS['check_occ']['statuses'] = (SITE_CACHED, SITE_BAITED)
LAYOUT = (('check_occ', 'check_empty'), ('ret_cached', 'ret_baited'))
ROW_NAMES = ('checks', 'retrievals')

if peak_sort['split'] not in ('alternate', 'random', 'all'):
    raise ValueError("peak_sort['split'] must be 'alternate', 'random' or 'all'")
PEAK_CV = peak_sort['split'] != 'all'       # cross-validated peak sort


def _match_groups():
    '''match_events as a list of tuples of condition names.'''
    if match_events is None:
        return []
    groups = list(LAYOUT) if match_events == 'rows' else [tuple(g) for g in match_events]
    flat = [c for g in groups for c in g]
    if any(c not in CONDS for c in flat) or len(flat) != len(set(flat)):
        raise ValueError(f'match_events: unknown or repeated condition in {groups}')
    return [g for g in groups if len(g) > 1]


MATCH_GROUPS = _match_groups()
ALIGNS = ('onset', 'offset')

WIN_FR, T_PTS = {}, {}
for _a in ALIGNS:
    _s, _e, T_PTS[_a] = window_frames(*psth_windows[_a], dt)
    WIN_FR[_a] = (_s, _e)


def _cond_label(cond):
    r = [i for i, row in enumerate(LAYOUT) if cond in row][0]
    return f"{CONDS[cond]['name']} {ROW_NAMES[r]}"


''' Step 1: per-session computation (cached) '''
def score_session_events(count_data, seed_struct):
    '''
    Onsets, offsets and expectation status of every check and retrieval.
    get_expectation_status judges "first encounter" against every site
    interaction, so scoring the two event types separately is safe.
    '''
    timeline = _seed_provenance_timeline(count_data, seed_struct,
                                         use_init_counts=use_init_counts,
                                         removal_rule=removal_rule)
    kw = dict(timeline=timeline, use_init_counts=use_init_counts,
              removal_rule=removal_rule, discovered_bait=discovered_bait)

    check_on, check_off, check_site = get_checks_raw(
        count_data, seed_struct, max_check_dur=max_check_dur, dt=dt)
    ret_on, ret_off, ret_site = get_retrieve_ints(count_data, seed_struct,
                                                  return_site_idx=True)
    ret_on, ret_off, ret_site = (np.asarray(x).astype(int)
                                 for x in (ret_on, ret_off, ret_site))

    events = {}
    for key, on, off, site in [('check', check_on, check_off, check_site),
                               ('retrieve', ret_on, ret_off, ret_site)]:
        status, _ = get_expectation_status(count_data, seed_struct, on, site, **kw)
        events[key] = dict(onsets=on, offsets=off, status=status)
        tally = ', '.join(f'{int(np.sum(status == g))} {SITE_STATUS_NAMES[g]}'
                          for g in [SITE_EMPTY, SITE_BAITED, SITE_CACHED, SITE_UNKNOWN]
                          if np.any(status == g))
        print(f'  {status.shape[0]} {key} events ({tally})')

    n_bad = int(np.sum(events['retrieve']['status'] == SITE_EMPTY))
    if n_bad:
        print(f'  warning: {n_bad} retrieval(s) from a site the ledger scores '
              f'as empty; left out (check the annotation)')
    return events


def usable_events(onsets, offsets, n_frames):
    '''Events whose onset AND offset windows both fit inside the session.'''
    return ((onsets + WIN_FR['onset'][0] >= 0) &
            (onsets + WIN_FR['onset'][1] <= n_frames) &
            (offsets + WIN_FR['offset'][0] >= 0) &
            (offsets + WIN_FR['offset'][1] <= n_frames))


def split_events(onsets, offsets, n_frames, seed_key):
    '''
    Two disjoint halves of a condition's usable events, as index arrays.

    'alternate' interleaves events in time order (1st, 3rd, 5th... vs 2nd,
    4th...), so slow changes over the session hit both halves equally.
    'random' is seeded per session so the split is the same every run.
    '''
    idx = np.flatnonzero(usable_events(onsets, offsets, n_frames))
    idx = idx[np.argsort(onsets[idx], kind='stable')]
    # ('all' never reaches here: there is nothing to split)
    if peak_sort['split'] == 'alternate':
        return [idx[0::2], idx[1::2]]
    if peak_sort['split'] == 'random':
        seed = (zlib.crc32(seed_key.encode()) + int(peak_sort['seed'])) & 0xffffffff
        perm = np.random.default_rng(seed).permutation(idx)
        half = perm.shape[0] // 2
        return [np.sort(perm[:half]), np.sort(perm[half:])]
    raise ValueError("peak_sort['split'] must be 'alternate' or 'random'")


def prepare_cond(spike_fr, onsets, offsets, amp_data=None):
    '''
    One condition's usable events and its per-cell drift mask.

    An event is used only if both of its windows fit in the session, so the
    two alignments always average the same events.

    Returns dict(on, off, keep, durations); keep is (n_cells, n_events), all
    True when there is no drift cut.
    '''
    n_cells, n_frames = spike_fr.shape
    onsets = np.asarray(onsets).astype(int)
    offsets = np.asarray(offsets).astype(int)
    usable = usable_events(onsets, offsets, n_frames)
    onsets, offsets = onsets[usable], offsets[usable]
    keep = np.ones((n_cells, onsets.shape[0]), dtype=bool)
    if (onsets.shape[0] and subsample_drift and
            (subsample_metric != 'amplitude' or amp_data is not None)):
        keep = np.asarray(select_trials_by_drift(
            spike_fr, onsets, dt, subsample_metric, subsample_thresh,
            thresh_t_window, amp_data=amp_data, min_spikes=amp_min_spikes), dtype=bool)
    return dict(on=onsets, off=offsets, keep=keep, durations=(offsets - onsets) * dt)


def average_events(trace, on, off, keep):
    '''
    Onset- and offset-aligned averages of `trace` over the events each cell
    keeps.  `trace` is pop_normalize output (SD units), or raw counts when
    norm_method is None (then converted to Hz).  Cells with fewer than
    min_events_per_group kept events are NaN.
    '''
    n_cells = trace.shape[0]
    out = {a: np.full((n_cells, T_PTS[a].shape[0]), np.nan) for a in ALIGNS}
    if on.shape[0] == 0:
        return out, np.zeros(n_cells, dtype=int)
    # event_psth divides by its dt to turn counts into Hz; a normalized trace
    # must not be rescaled, so pass 1
    psth_dt = dt if norm_method is None else 1.0
    p_on, p_off, _, nu = event_psth_on_off(trace, on, off,
                                           WIN_FR['onset'], WIN_FR['offset'], psth_dt,
                                           sigma_frames=sigma_frames, keep=keep)
    n_used = nu[:, 0]
    enough = n_used >= min_events_per_group
    out['onset'][enough] = p_on[enough, 0]
    out['offset'][enough] = p_off[enough, 0]
    return out, n_used


def match_keeps(keeps, rng):
    '''
    Equalize, cell by cell, how many events each entry of `keeps` (name ->
    (n_cells, n_events) bool) contributes, by dropping events at random from
    the larger ones down to the smallest count for that cell.

    Each entry gets ONE random priority order for the session and every cell
    keeps its highest-priority events, so without a drift cut all cells use
    the same events.
    '''
    n_target = np.min([k.sum(axis=1) for k in keeps.values()], axis=0)
    out = {}
    for name, k in keeps.items():
        order = rng.permutation(k.shape[1])
        k_ord = k[:, order]
        sel = k_ord & (np.cumsum(k_ord, axis=1) <= n_target[:, None])
        m = np.zeros_like(k)
        m[:, order] = sel
        out[name] = m
    return out


def _session_rng(seed_key, seed):
    return np.random.default_rng((zlib.crc32(seed_key.encode()) + int(seed)) & 0xffffffff)


def compute_session(data_dict, bird, session_id):
    '''Everything the figures need from one session, or None if no cells.'''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}behavior_data/"
    session_data = data_dict[bird][session_id]

    ''' Neural data + cell filters (same mask as the single-cell code) '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape
    avg_firing_rate = 10 ** session_data['waveform_props'][2]
    all_ids = np.load(f"{data_dir}aligned_spikes_ids.npy")

    keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw, **CELL_FILTERS)
    filtered = apply_cell_filter(keep_cells, spike_fr=spike_fr,
                                 avg_firing_rate=avg_firing_rate, cluster_id=all_ids)
    spike_fr = filtered['spike_fr']
    cell_ids = filtered['cluster_id']
    n_cells = spike_fr.shape[0]
    if n_cells == 0:
        print('  no cells survived the filters, skipping')
        return None
    ks_dir = f"{bird}_{session_data['ephys_id']}/{session_data['ks_folder']}/"

    # normalize each cell's whole-session trace, exactly as the feeder summary
    if norm_method == 'pop_normalize':
        trace = pop_normalize(spike_fr, dt=dt, **pop_norm_kw)
    elif norm_method is None:
        trace = spike_fr
    else:
        raise ValueError("norm_method must be 'pop_normalize' or None")

    ''' Behavior '''
    seed_struct, count_data = load_behavior_data(data_dir)
    events = score_session_events(count_data, seed_struct)

    amp_data = None
    if subsample_drift and subsample_metric == 'amplitude':
        try:
            amp_data = load_spike_amplitudes(session_dir, data_dir, ks_dir,
                                             cell_ids, n_frames, fps=fps)
        except FileNotFoundError as err:
            print(f'  no KS amplitudes ({err.filename}); no drift cut this session')

    ''' Usable events + drift masks per condition '''
    prep = {}
    for cond, spec in CONDS.items():
        ev = events[spec['event']]
        sel = np.isin(ev['status'], spec['statuses'])
        prep[cond] = prepare_cond(spike_fr, ev['onsets'][sel], ev['offsets'][sel],
                                  amp_data=amp_data)

    ''' Optional event-count matching, per cell '''
    rng = _session_rng(f'{bird}_{session_id}_match', match_seed)
    keeps = {c: prep[c]['keep'] for c in CONDS}
    for g in MATCH_GROUPS:
        keeps.update(match_keeps({c: keeps[c] for c in g}, rng))

    ''' Event averages per condition '''
    psth = {a: {} for a in ALIGNS}
    n_used, n_events, durations = {}, {}, {}
    for cond in CONDS:
        p = prep[cond]
        out, nu = average_events(trace, p['on'], p['off'], keeps[cond])
        for a in ALIGNS:
            psth[a][cond] = out[a]
        n_used[cond], n_events[cond], durations[cond] = nu, p['on'].shape[0], p['durations']
        print(f'  {cond}: {n_events[cond]} usable events, '
              f'{int(np.sum(nu >= min_events_per_group))}/{n_cells} cells drawn')

    ''' Split halves for the cross-validated peak sort '''
    split = None
    if PEAK_CV:
        pc = peak_sort['cond']
        p = prep[pc]
        halves = split_events(p['on'], p['off'], n_frames, seed_key=f'{bird}_{session_id}')
        units = {f'h{i}': p['keep'][:, h] for i, h in enumerate(halves)}
        # if the peak condition is matched, match its partners to the HALF
        # count instead, so the held-out half and its partners stay equal
        partners = [c for g in MATCH_GROUPS if pc in g for c in g if c != pc]
        if partners:
            units.update({c: prep[c]['keep'] for c in partners})
            units = match_keeps(units, rng)
        split = dict(psth=[], n_used=[], n_events=[], durations=[], partner={})
        for i, h in enumerate(halves):
            out, nu = average_events(trace, p['on'][h], p['off'][h], units[f'h{i}'])
            split['psth'].append(out)
            split['n_used'].append(nu)
            split['n_events'].append(int(h.shape[0]))
            split['durations'].append(p['durations'][h])
        for c in partners:
            out, nu = average_events(trace, prep[c]['on'], prep[c]['off'], units[c])
            split['partner'][c] = dict(psth=out, n_used=nu)

    return dict(bird=bird, session=str(session_id), cluster_id=cell_ids.astype(int),
                row_idx=np.flatnonzero(keep_cells), n_cells_raw=n_cells_raw,
                avg_fr=np.asarray(filtered['avg_firing_rate'], dtype=float),
                psth=psth, n_used=n_used, n_events=n_events, durations=durations,
                split=split)


def compute_params():
    '''Settings that change step 1; the cache is rebuilt if any differ.'''
    return dict(version=CACHE_VERSION, CELL_FILTERS=dict(CELL_FILTERS),
                norm_method=norm_method, pop_norm_kw=dict(pop_norm_kw),
                max_check_dur=max_check_dur, discovered_bait=discovered_bait,
                removal_rule=removal_rule, use_init_counts=use_init_counts,
                check_baited=check_baited, min_events_per_group=min_events_per_group,
                subsample_drift=subsample_drift, subsample_metric=subsample_metric,
                subsample_thresh=subsample_thresh, thresh_t_window=thresh_t_window,
                psth_windows=dict(psth_windows), sigma_frames=sigma_frames, fps=fps,
                match_events=MATCH_GROUPS, match_seed=match_seed,
                split=dict(cond=peak_sort['cond'], split=peak_sort['split'],
                           seed=peak_sort['seed']),
                birds_to_plot=birds_to_plot, exclude_birds=list(exclude_birds),
                exclude_sessions=dict(exclude_sessions))


def load_or_compute_sessions(data_dict, recompute=False):
    params = compute_params()
    if not recompute and os.path.isfile(cache_file):
        cached = np.load(cache_file, allow_pickle=True).item()
        if cached.get('params') == params:
            print(f'loaded {len(cached["sessions"])} sessions from {cache_file}')
            return cached['sessions']
        print('cache was built with different settings, recomputing')

    birds = birds_to_plot if birds_to_plot is not None else list(data_dict.keys())
    sessions = []
    for bird in birds:
        if bird in exclude_birds:
            continue
        for session_id in data_dict[bird]['all_sessions']:
            if str(session_id) in [str(s) for s in exclude_sessions.get(bird, [])]:
                continue
            pre = data_dict[bird][session_id]['preprocessed_data']
            if not (('behavior' in pre) & ('ephys' in pre)):
                continue
            print(f'{bird}_{session_id}')
            res = compute_session(data_dict, bird, session_id)
            if res is not None:
                sessions.append(res)

    np.save(cache_file, dict(params=params, sessions=sessions), allow_pickle=True)
    return sessions


''' Step 2: pooling, positions, rastermap order, labels '''
def shank_name(idx):
    return '?' if idx < 0 else chr(ord('A') + int(idx))


def get_cell_positions(session_data, n_cells_raw, row_idx):
    '''
    [ML, AP, DV] and shank of each kept cell, from the 'cell_pos' and
    'shank_idx' that get_probe_coords_lhy.save_cell_positions stores per
    session (indexed by the rows of aligned_spikes.npy, before filtering).
    NaN / -1 if missing or mismatched.
    '''
    n = row_idx.shape[0]
    pos, shank = session_data.get('cell_pos'), session_data.get('shank_idx')
    if pos is None or shank is None:
        print('  no cell_pos / shank_idx in data_dict (run get_probe_coords_lhy)')
        return np.full((n, 3), np.nan), np.full(n, -1, dtype=int)
    pos, shank = np.asarray(pos, dtype=float), np.asarray(shank).astype(int)
    if pos.shape[0] != n_cells_raw or shank.shape[0] != n_cells_raw:
        print(f'  WARNING: cell_pos has {pos.shape[0]} cells but aligned_spikes has '
              f'{n_cells_raw}; ignoring positions for this session')
        return np.full((n, 3), np.nan), np.full(n, -1, dtype=int)
    return pos[row_idx], shank[row_idx]


def get_rastermap_rank(bird, session_id, n_cells_raw, row_idx):
    '''
    Position of each kept cell in the saved rastermap order (lower = higher
    on the plot).  NaN if the file is missing or the cell is not in isort.
    '''
    path = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/{rastermap_file}"
    if not os.path.isfile(path):
        print(f'  no rastermap sort for {bird}_{session_id} ({path})')
        return np.full(row_idx.shape[0], np.nan)
    isort = np.asarray(np.load(path, allow_pickle=True).item()['isort']).ravel().astype(int)
    if isort.max() >= n_cells_raw:
        print(f'  WARNING: rastermap isort for {bird}_{session_id} indexes past the '
              f'{n_cells_raw} rows of aligned_spikes.npy; ignoring it')
        return np.full(row_idx.shape[0], np.nan)
    rank = np.full(n_cells_raw, np.nan)
    rank[isort] = np.arange(isort.shape[0])
    return rank[row_idx]


def uses_method(method):
    return sort_by == method or (sort_by == 'labels' and label_secondary_sort == method)


def apply_split(sessions):
    '''
    Copies of the sessions in which peak_sort['cond'] holds only the held-out
    half (PSTHs, event counts, durations), with the sorting half kept aside
    under 'sort_psth'.  Everything downstream then plots, normalizes and
    counts the held-out half without knowing about the split.
    '''
    c = peak_sort['cond']
    h_sort = 1 if peak_sort['swap_halves'] else 0
    h_plot = 1 - h_sort
    out = []
    for s in sessions:
        sp = s['split']
        s2 = dict(s)
        s2['psth'] = {a: dict(s['psth'][a]) for a in ALIGNS}
        s2['n_used'], s2['n_events'] = dict(s['n_used']), dict(s['n_events'])
        s2['durations'] = dict(s['durations'])
        for a in ALIGNS:
            s2['psth'][a][c] = sp['psth'][h_plot][a]
        s2['n_used'][c] = sp['n_used'][h_plot]
        s2['n_events'][c] = sp['n_events'][h_plot]
        s2['durations'][c] = sp['durations'][h_plot]
        # partners of a matched peak condition, re-matched to the half count
        for p_cond, part in sp['partner'].items():
            for a in ALIGNS:
                s2['psth'][a][p_cond] = part['psth'][a]
            s2['n_used'][p_cond] = part['n_used']
        s2['sort_psth'] = sp['psth'][h_sort]
        s2['sort_n_events'] = sp['n_events'][h_sort]
        out.append(s2)
    return out


def pool_sessions(sessions, data_dict):
    '''One row per cell across every session, and matching PSTH stacks.'''
    need_rm = uses_method('rastermap')
    tables = []
    for s_idx, s in enumerate(sessions):
        pos, shank = get_cell_positions(data_dict[s['bird']][s['session']],
                                        s['n_cells_raw'], s['row_idx'])
        t = pd.DataFrame(dict(bird=s['bird'], session=s['session'],
                              cluster_id=s['cluster_id'], shank=shank,
                              ml=pos[:, 0], ap=pos[:, 1], dv=pos[:, 2],
                              avg_fr=s['avg_fr'], sess_idx=s_idx))
        t['rm_rank'] = (get_rastermap_rank(s['bird'], s['session'], s['n_cells_raw'],
                                           s['row_idx']) if need_rm else np.nan)
        for c in CONDS:
            t[f'n_ev_{c}'] = s['n_events'][c]
            t[f'n_used_{c}'] = s['n_used'][c]
        tables.append(t)
    cells = pd.concat(tables, ignore_index=True)
    cells['shank_name'] = [shank_name(i) for i in cells['shank']]
    cells['bird_shank'] = cells['bird'] + ' ' + cells['shank_name']
    cells['bird_session'] = cells['bird'] + ' ' + cells['session']
    # mean AP of each (bird, shank) over all its cells and sessions, which is
    # what orders the depth blocks posterior -> anterior (plot_fr_vs_depth)
    cells['shank_ap'] = cells.groupby('bird_shank')['ap'].transform('mean')

    psth = {a: {c: np.concatenate([s['psth'][a][c] for s in sessions], axis=0)
                for c in CONDS} for a in ALIGNS}
    psth_sort = None
    if all('sort_psth' in s for s in sessions):
        psth_sort = {a: np.concatenate([s['sort_psth'][a] for s in sessions], axis=0)
                     for a in ALIGNS}
    return cells, psth, psth_sort


def assign_labels(cells):
    cells = cells.copy()
    if cell_labels is None:
        cells['label'] = None
        return cells
    if isinstance(cell_labels, str):
        lab = pd.read_csv(cell_labels)
        lab['session'] = lab['session'].astype(str)
        lab['cluster_id'] = lab['cluster_id'].astype(int)
        cells = cells.merge(lab[['bird', 'session', 'cluster_id', 'label']],
                            on=['bird', 'session', 'cluster_id'], how='left')
    else:
        lab = np.asarray(cell_labels, dtype=object)
        if lab.shape[0] != cells.shape[0]:
            raise ValueError(f'cell_labels has {lab.shape[0]} entries but '
                             f'{cells.shape[0]} cells passed the filters; see '
                             f'cell_table.csv for the expected order')
        cells['label'] = lab
    n_missing = int(cells['label'].isna().sum())
    if n_missing:
        print(f'{n_missing} cells have no label; they go in an "unlabeled" block')
    cells['label'] = cells['label'].astype(object).where(cells['label'].notna(), 'unlabeled')
    return cells


''' Step 3a: choosing cells '''
def select_cells(cells, member, mode):
    '''
    Which pooled cells each subplot shows, before sorting.

    member : bool mask of the cells belonging to this figure
    mode   : 'exclude_sessions' | 'blank_rows' | 'unaligned'
    '''
    valid = {c: cells[f'n_used_{c}'].to_numpy() >= min_events_per_group for c in CONDS}
    sess_ok = np.all([cells[f'n_ev_{c}'].to_numpy() >= min_events_per_group
                      for c in CONDS], axis=0)
    any_valid = np.any(list(valid.values()), axis=0)
    all_valid = np.all(list(valid.values()), axis=0)

    if mode == 'exclude_sessions':
        m = member & sess_ok & all_valid
        return {c: m for c in CONDS}
    if mode == 'blank_rows':
        m = member & any_valid
        return {c: m for c in CONDS}
    if mode == 'unaligned':
        return {c: member & valid[c] for c in CONDS}
    raise ValueError(f'unknown missing-event mode {mode}')


''' Step 3b: sorting '''
def peak_times(sub, ctx, align):
    '''
    Peak time (s, start of the peak bin) of each cell in the SORTING half of
    peak_sort['cond'].  NaN where the cell has too few events in that half or
    its PSTH is flat over the search window.  'absdev' also catches cells
    whose largest change is a decrease.
    '''
    if ctx['psth_sort'] is None:
        raise ValueError("peak sort needs the split halves (apply_split)")
    a = peak_sort['align'] or align
    t = T_PTS[a]
    w = (np.ones(t.shape[0], dtype=bool) if peak_sort['window'] is None else
         (t >= peak_sort['window'][0]) & (t < peak_sort['window'][1]))
    X = ctx['psth_sort'][a][np.asarray(sub, dtype=int)][:, w]
    out = np.full(X.shape[0], np.nan)
    valid = np.all(np.isfinite(X), axis=1)
    if not np.any(valid):
        return out
    Xv = X[valid]
    if peak_sort['extra_sigma_frames'] > 0:
        Xv = gaussian_filter1d(Xv, peak_sort['extra_sigma_frames'], axis=1, mode='nearest')
    if peak_sort['metric'] == 'max':
        k = np.argmax(Xv, axis=1)
    elif peak_sort['metric'] == 'min':
        k = np.argmin(Xv, axis=1)
    elif peak_sort['metric'] == 'absdev':
        k = np.argmax(np.abs(Xv - Xv.mean(axis=1, keepdims=True)), axis=1)
    else:
        raise ValueError("peak_sort['metric'] must be 'max', 'min' or 'absdev'")
    pk = t[w][k]
    pk[np.ptp(Xv, axis=1) == 0] = np.nan
    out[valid] = pk
    return out


def _response_value(sub, ctx, align):
    t = T_PTS[align]
    w = (t >= response_sort['window'][0]) & (t < response_sort['window'][1])
    p = ctx['psth'][align]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = np.nanmean(p[response_sort['cond']][sub][:, w], axis=1)
        if response_sort.get('minus'):
            v = v - np.nanmean(p[response_sort['minus']][sub][:, w], axis=1)
    return v


def order_within(sub, method, ctx, align):
    '''Order cells inside one block.  Unknown values always go last.'''
    cells = ctx['cells']
    sub = np.asarray(sub)
    if method in (None, 'none') or sub.shape[0] < 2:
        return sub
    if method == 'depth':
        dv = cells['dv'].to_numpy(float)[sub]
        key = -dv if depth_order == 'ventral_first' else dv
    elif method == 'rastermap':
        key = cells['rm_rank'].to_numpy(float)[sub]
    elif method == 'peak':
        key = peak_times(sub, ctx, align)
    elif method == 'response':
        key = -_response_value(sub, ctx, align)             # largest on top
    else:
        raise ValueError(f'unknown sort {method}')
    key = np.where(np.isnan(key), np.inf, key)
    return sub[np.argsort(key, kind='stable')]


def _label_key(x):
    '''Numeric labels in numeric order, then strings, "unlabeled" last.'''
    if x == 'unlabeled':
        return (2, 0, '')
    try:
        return (0, float(x), '')
    except (TypeError, ValueError):
        return (1, 0, str(x))


# blocks each within-block sort needs: depth ranks are only comparable within
# a shank, rastermap ranks only within the session they were fit on
INNER_BLOCKS = {'depth': 'bird_shank', 'rastermap': 'bird_session'}
OUTER_COLS = {None: None, 'bird': 'bird', 'session': 'bird_session'}


def _block_names_in_order(col, names, cells):
    if col == 'label':
        if label_order is not None:
            return ([n for n in label_order if n in names] +
                    sorted([n for n in names if n not in label_order], key=_label_key))
        return sorted(names, key=_label_key)
    if col == 'bird_shank':
        info = cells.drop_duplicates('bird_shank').set_index('bird_shank')
        birds = list(pd.unique(cells['bird']))
        if shank_order == 'ap':
            ap = info.loc[names, 'shank_ap'].to_numpy(float)
            ap = -ap if ap_order == 'anterior_first' else ap
            key = np.where(np.isnan(ap), np.inf, ap)       # no histology: last
            return [names[i] for i in np.argsort(key, kind='stable')]
        return sorted(names, key=lambda n: (birds.index(info.loc[n, 'bird']),
                                            info.loc[n, 'shank_name'] == '?',
                                            info.loc[n, 'shank_name']))
    return names                                           # pooling order


def order_cells(idx, ctx, align):
    '''
    Ordered pooled indices, plus up to two block levels per row:
        outer : label (sort_by='labels') or block_by (peak / response / none)
        inner : (bird, shank) for depth, (bird, session) for rastermap
    A level is None when it does not apply or is the same for every row.
    '''
    cells = ctx['cells']
    idx = np.asarray(idx, dtype=int)
    if idx.shape[0] == 0:
        return idx, dict(outer=None, inner=None)

    if sort_by == 'labels':
        if cells['label'].isna().all():
            raise ValueError("sort_by='labels' needs cell_labels")
        within = label_secondary_sort
        outer_col = 'label'
    else:
        within = sort_by
        outer_col = None if within in INNER_BLOCKS else OUTER_COLS[block_by]
    inner_col = INNER_BLOCKS.get(within)

    def groups(sub, col):
        if col is None:
            return [(None, sub)]
        vals = cells[col].to_numpy(object)[sub]
        names = _block_names_in_order(col, list(pd.unique(vals)), cells)
        return [(n, sub[vals == n]) for n in names]

    ordered, outer, inner = [], [], []
    for o_name, o_sub in groups(idx, outer_col):
        for i_name, i_sub in groups(o_sub, inner_col):
            srt = order_within(i_sub, within, ctx, align)
            ordered.append(srt)
            outer += [o_name] * srt.shape[0]
            inner += [i_name] * srt.shape[0]
    outer, inner = np.asarray(outer, dtype=object), np.asarray(inner, dtype=object)
    if outer_col is None or (outer_col != 'label' and len(set(outer)) == 1):
        outer = None
    if inner_col is None or len(set(inner)) == 1:
        inner = None
    return np.concatenate(ordered), dict(outer=outer, inner=inner)


def arrange_subplots(shown, ctx, align, mode):
    '''Ordered cell indices (and block names) for every subplot.'''
    rows, blocks = {}, {}
    if mode == 'unaligned' and resort_unaligned:
        for c in CONDS:
            rows[c], blocks[c] = order_cells(np.flatnonzero(shown[c]), ctx, align)
        return rows, blocks

    # one shared order over every cell shown anywhere; each subplot keeps its
    # own cells in that order (identical sets unless mode == 'unaligned')
    union = np.any([shown[c] for c in CONDS], axis=0)
    order, names = order_cells(np.flatnonzero(union), ctx, align)
    for c in CONDS:
        m = shown[c][order]
        rows[c] = order[m]
        blocks[c] = {k: (None if v is None else v[m]) for k, v in names.items()}
    return rows, blocks


''' Step 3c: plotting '''
def _runs(keys):
    '''(start, end) of each run of equal consecutive keys.'''
    change = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    return list(zip([0] + change, change + [len(keys)]))


def _block_runs(blocks):
    '''Runs for the outer level and for the (outer, inner) level.'''
    outer, inner = blocks['outer'], blocks['inner']
    if outer is None and inner is None:
        return [], []
    n = len(outer) if outer is not None else len(inner)
    o_key = list(outer) if outer is not None else [None] * n
    o_runs = _runs(o_key) if outer is not None else []
    i_runs = _runs(list(zip(o_key, inner))) if inner is not None else []
    return o_runs, i_runs


def _draw_blocks(ax, blocks, draw_labels):
    o_runs, i_runs = _block_runs(blocks)
    if show_block_dividers:
        o_edges = {a for a, _ in o_runs[1:]}
        # dashed within an outer block only when there is an outer level
        inner_ls = (0, (3, 2)) if o_runs else '-'
        for a, _ in i_runs[1:]:
            if a not in o_edges:
                ax.axhline(a - 0.5, color='k', lw=0.5, ls=inner_ls)
        for e in o_edges:
            ax.axhline(e - 0.5, color='k', lw=0.8)
    if not (draw_labels and show_block_labels):
        return
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for runs, names, pad in [(i_runs, blocks['inner'], 2),
                             (o_runs, blocks['outer'], 2 if not i_runs else 40)]:
        for a, b in runs:
            ax.annotate(str(names[a]), xy=(0, (a + b - 1) / 2), xycoords=trans,
                        xytext=(-pad, 0), textcoords='offset points', ha='right',
                        va='center', fontsize=block_label_size, annotation_clip=False)


def _clim(psth, rows):
    if clim is not None:
        return tuple(clim)
    vals = [psth[a][c][rows[c]].ravel() for a in ALIGNS for c in CONDS if rows[c].size]
    vals = np.concatenate(vals) if vals else np.zeros(1)
    vals = vals[np.isfinite(vals)]
    if norm_method is None:
        return (0.0, float(np.percentile(vals, 99)) if vals.size else 1.0)
    v = float(np.percentile(np.abs(vals), 99)) if vals.size else 1.0
    return (-v, v)


def plot_summary(psth, rows, blocks, align, mode, title, save_path):
    t = T_PTS[align]
    x0, x1 = t[0], t[-1] + dt
    cm = plt.get_cmap(cmap_raw if norm_method is None else cmap_norm).copy()
    cm.set_bad(nan_color)
    line_color = 'w' if norm_method is None else 'k'
    vmin, vmax = _clim(psth, rows)

    f, ax = plt.subplots(2, 2, figsize=fig_size, sharex=True,
                         gridspec_kw=dict(wspace=0.12, hspace=0.25))
    ticks = np.arange(np.ceil(x0 / xtick_step), np.floor(x1 / xtick_step) + 1) * xtick_step
    im = None
    for r, row in enumerate(LAYOUT):
        for col, cond in enumerate(row):
            a = ax[r, col]
            idx = rows[cond]
            n = idx.shape[0]
            a.set_title(_cond_label(cond), fontsize=axis_label, color=cond_colors[cond])
            a.set_xlim(x0, x1)
            a.set_xticks(ticks)
            a.set_xticklabels([f'{v:g}' for v in ticks])
            a.tick_params(labelsize=tick_label)
            if n == 0:
                a.set_facecolor(nan_color)
                continue

            im = a.imshow(psth[align][cond][idx], aspect='auto', cmap=cm,
                          vmin=vmin, vmax=vmax, interpolation='none',
                          extent=(x0, x1, n - 0.5, -0.5))
            a.axvline(0, color=line_color, ls='--', lw=0.8)

            if show_median_duration:
                med = rows['_median_dur'][cond]
                med = med if align == 'onset' else -med
                if np.isfinite(med) and x0 < med < x1:
                    a.axvline(med, color=line_color, ls=':', lw=0.8)

            _draw_blocks(a, blocks[cond], draw_labels=(col == 0 or mode == 'unaligned'))

        ax[r, 0].set_ylabel(f'cells (sorted by {sort_by})', fontsize=axis_label)
        ax[r, 1].set_yticks([])
    for col in range(2):
        ax[1, col].set_xlabel(f'time from {align} (s)', fontsize=axis_label)

    if im is not None:
        top, bot = ax[0, 1].get_position(), ax[1, 1].get_position()
        cax = f.add_axes([top.x1 + 0.02, bot.y0 + (top.y1 - bot.y0) * 0.35,
                          0.015, (top.y1 - bot.y0) * 0.3])
        cb = f.colorbar(im, cax=cax)
        cticks = [vmin, vmax] if norm_method is None else [vmin, 0, vmax]
        cb.set_ticks(cticks)
        cb.set_ticklabels([f'{v:.3g}' for v in cticks])
        cb.ax.tick_params(labelsize=tick_label)
        cb.set_label('rate (Hz)' if norm_method is None else 'activity (SD)',
                     fontsize=tick_label)

    if show_suptitle:
        f.suptitle(title, fontsize=title_size, y=0.97)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(f)


''' Figure legend text '''
def _method_text(method):
    if method == 'depth':
        within = 'ventral to dorsal' if depth_order == 'ventral_first' else 'dorsal to ventral'
        across = (f"blocks ordered by the shank's mean AP position "
                  f"({ap_order.replace('_', ' ')})" if shank_order == 'ap'
                  else 'blocks ordered by bird, then shank letter')
        return (f'grouped by bird and shank ({across}); within each shank, cells '
                f'ordered by estimated DV of their best channel, {within}')
    if method == 'rastermap':
        return ('grouped by bird and session; within each session, cells in the '
                f'rastermap order saved in {rastermap_file}')
    if method == 'peak':
        win = ('the whole window' if peak_sort['window'] is None
               else f"{peak_sort['window'][0]} to {peak_sort['window'][1]} s")
        what = {'max': 'peak', 'min': 'trough',
                'absdev': 'largest deviation from its mean'}[peak_sort['metric']]
        return (f"ordered by the time of each cell's {what} "
                f"({win}) in {_cond_label(peak_sort['cond'])}, earliest on top")
    if method == 'response':
        w = response_sort['window']
        what = response_sort['cond'] + (f" minus {response_sort['minus']}"
                                        if response_sort.get('minus') else '')
        return f'ordered by mean activity {w[0]} to {w[1]} s, {what}, largest on top'
    return 'in pooling order'


def write_legend_file(path, title, align, mode, cells, sessions, rows, blocks):
    L = [f'{title}, {align}-aligned', '']
    L.append('Each row is one cell and each column a time bin '
             f'({dt * 1000:.0f} ms) relative to event {align} (dashed line). '
             'Top row: checks (site opened, contents unchanged, '
             f'<= {max_check_dur} s). Bottom row: retrievals (seed removed). '
             'Left: a seed the bird should expect (its own caches, or baits it has '
             'already found in any earlier interaction with the site'
             + (', which are dropped instead' if discovered_bait == 'drop' else '')
             + '). Right: empty checks / retrievals of baits at a site the bird '
             'had never opened.'
             + (' Checks of undiscovered baits are left out.' if check_baited == 'drop'
                else ' Checks of undiscovered baits are pooled with occupied checks.'))
    L.append('')
    smooth = (f'Gaussian-smoothed (sigma {sigma_frames * dt * 1000:.0f} ms)'
              if sigma_frames > 0 else 'unsmoothed')
    if norm_method == 'pop_normalize':
        L.append("Activity: each cell's instantaneous firing rate over the whole session "
                 f"was divided by its session SD (+{pop_norm_kw['std_reg']:g}) and a "
                 f"running-average baseline (window {pop_norm_kw['baseline_window']}) "
                 'was subtracted (pop_normalize), so units are SDs from baseline; this '
                 f'trace was then averaged across events and {smooth}. '
                 f'Colour limits {_clim_text()} SD (red above, blue below baseline).')
    else:
        L.append(f'Activity: trial-averaged firing rate in Hz, {smooth}. '
                 f'Colour limits {_clim_text()}.')
    if subsample_drift:
        L.append(f"Per cell, events were excluded where the unit's {subsample_metric} "
                 f'in a {thresh_t_window / 60:g}-min window around the event was below '
                 f'{subsample_thresh}x its session average.')
    L.append(f'A cell/condition is drawn only with >= {min_events_per_group} usable events. '
             f'Cells: {_filter_text()}.')
    L.append('')
    sort_txt = (_method_text(sort_by) if sort_by != 'labels' else
                'grouped by user label' + (f', then {_method_text(label_secondary_sort)}'
                                           if label_secondary_sort else ''))
    if sort_by not in ('labels', 'depth', 'rastermap') and block_by:
        sort_txt = f'grouped by {block_by}, then ' + sort_txt
    L.append(f'Sorting: {sort_txt}.')
    if uses_method('peak') and PEAK_CV:
        h = 1 if peak_sort['swap_halves'] else 0
        split = ('alternating events in time' if peak_sort['split'] == 'alternate'
                 else 'a random split')
        L.append(f"Cross-validated: {_cond_label(peak_sort['cond'])} events were split "
                 f'in half ({split}); half {h} set the order and the '
                 f'{_cond_label(peak_sort["cond"])} panel shows only half {1 - h}. '
                 + _partner_text())
    elif uses_method('peak'):
        L.append(f"Not cross-validated: the order was set from all "
                 f"{_cond_label(peak_sort['cond'])} events, which are also the ones "
                 'plotted, so the sequence in that panel is partly built in by the sort.')
    if MATCH_GROUPS:
        grp = '; '.join(' vs. '.join(_cond_label(c) for c in g) for g in MATCH_GROUPS)
        L.append(f'Event counts were matched within each cell ({grp}) by randomly '
                 'dropping events from the larger condition after the drift cut.')

    first = next(iter(blocks.values()))
    o_runs, i_runs = _block_runs(first)
    if o_runs or i_runs:
        L.append('Blocks, top to bottom (rows):')
        names = first['inner'] if i_runs else first['outer']
        for a, b in (i_runs or o_runs):
            outer = f"{first['outer'][a]} / " if (i_runs and first['outer'] is not None) else ''
            L.append(f'  {outer}{names[a]}: rows {a + 1}-{b}')
    L.append('')
    L.append({'exclude_sessions': 'Sessions short of any condition were excluded, so all '
                                  'panels show the same cells in the same rows.',
              'blank_rows': 'All panels show the same cells in the same rows; grey rows '
                            'had too few events in that condition.',
              'unaligned': 'Each panel shows only cells with enough events in that '
                           'condition, so rows do not line up across panels.'}[mode])
    L.append('')
    L.append('Per panel:')
    for cond in CONDS:
        idx = rows[cond]
        has = cells[f'n_used_{cond}'].to_numpy()[idx] >= min_events_per_group
        s_used = np.unique(cells['sess_idx'].to_numpy()[idx][has])
        n_ev = int(sum(sessions[s]['n_events'][cond] for s in s_used))
        durs = [sessions[s]['durations'][cond] for s in s_used]
        durs = np.concatenate(durs) if durs else np.zeros(0)
        med = f'{np.median(durs):.2f} s' if durs.size else 'n/a'
        per_cell = cells[f'n_used_{cond}'].to_numpy()[idx][has]
        matched = any(cond in g for g in MATCH_GROUPS)
        step = 'matching' if matched else ('the drift cut' if subsample_drift else '')
        per_cell_txt = (f' (median {np.median(per_cell):g} used per cell'
                        + (f' after {step}' if step else '') + ')'
                        if per_cell.size else '')
        L.append(f'  {_cond_label(cond)}: {int(has.sum())} cells'
                 + (f' (of {idx.shape[0]} rows)' if has.sum() < idx.shape[0] else '')
                 + f', {_n(n_ev, "usable event")}{per_cell_txt}, {_n(len(s_used), "session")}, '
                 f'{_n(cells.loc[idx[has], "bird"].nunique(), "bird")}; '
                 f'median event duration {med}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def _partner_text():
    partners = [c for g in MATCH_GROUPS if peak_sort['cond'] in g
                for c in g if c != peak_sort['cond']]
    if not partners:
        return 'The other panels use all their events.'
    names = ' and '.join(_cond_label(c) for c in partners)
    names = names[0].upper() + names[1:]
    return (f'{names} (matched to it) were re-matched to the held-out half count; '
            'the other panels use all their events.')


def _n(k, word):
    return f'{k} {word}' + ('' if k == 1 else 's')


def _filter_text():
    parts = ['all good units' if CELL_FILTERS.get('cell_type', 'all') == 'all'
             else f"{CELL_FILTERS['cell_type']} units only"]
    if CELL_FILTERS.get('fr_thresh') is not None:
        parts.append(f"session firing rate > {CELL_FILTERS['fr_thresh']} Hz")
    if CELL_FILTERS.get('use_stim_filter'):
        parts.append('on or beside stim-responsive channels')
    return ', '.join(parts)


def _clim_text():
    if clim is not None:
        return f'{clim[0]:g} to {clim[1]:g}'
    return ('0 to the 99th percentile (Hz)' if norm_method is None
            else '+/- the 99th percentile of |activity|')


def _median_durations(cells, sessions, rows):
    out = {}
    for cond in CONDS:
        idx = rows[cond]
        has = cells[f'n_used_{cond}'].to_numpy()[idx] >= min_events_per_group
        durs = [sessions[s]['durations'][cond]
                for s in np.unique(cells['sess_idx'].to_numpy()[idx][has])]
        durs = np.concatenate(durs) if durs else np.zeros(0)
        out[cond] = float(np.median(durs)) if durs.size else np.nan
    return out


''' Figure groups '''
def iter_figures(cells, sessions):
    '''(level, member mask, title, save folder, prefix, mode) for every figure.'''
    birds = cells['bird'].to_numpy(object)
    sess_idx = cells['sess_idx'].to_numpy()
    for level in plot_levels:
        if level == 'session':
            for s_idx, s in enumerate(sessions):
                short = [c for c in CONDS if s['n_events'][c] < min_events_per_group]
                if short and session_missing == 'skip':
                    print(f'skipping {s["bird"]}_{s["session"]} figure: too few '
                          f'{", ".join(short)}')
                    continue
                yield (level, sess_idx == s_idx, f'{s["bird"]} {s["session"]}',
                       f'{save_figs_dir}{s["bird"]}/check_retrieve_summary/',
                       f'{s["session"]}_', 'blank_rows')
        elif level == 'bird':
            for bird in pd.unique(birds):
                yield (level, birds == bird, bird,
                       f'{save_figs_dir}{bird}/check_retrieve_summary/',
                       f'{bird}_', pooled_missing)
        elif level == 'all':
            yield (level, np.ones(cells.shape[0], dtype=bool), 'all birds',
                   f'{save_figs_dir}check_retrieve_summary/', 'all_', pooled_missing)
        else:
            raise ValueError(f'unknown plot level {level}')


def _sort_tag():
    tag = sort_by
    if sort_by == 'labels' and label_secondary_sort:
        tag += f'-{label_secondary_sort}'
    if uses_method('peak'):
        tag += f"-{peak_sort['cond']}" + (f"-{peak_sort['align']}" if peak_sort['align'] else '')
    if sort_by in ('peak', 'response', 'none') and block_by:
        tag += f'-by{block_by}'
    return tag


def main(recompute=False):
    data_dict = np.load(data_file, allow_pickle=True).item()
    sessions = load_or_compute_sessions(data_dict, recompute=recompute)
    if not sessions:
        print('no sessions with cells')
        return

    # cross-validated peak sort: the sort condition is plotted from its
    # held-out half only.  split='all' sorts on and plots all of its events
    if uses_method('peak') and PEAK_CV:
        sessions = apply_split(sessions)
    cells, psth, psth_sort = pool_sessions(sessions, data_dict)
    if uses_method('peak') and not PEAK_CV:
        psth_sort = {a: psth[a][peak_sort['cond']] for a in ALIGNS}
    cells = assign_labels(cells)
    table_dir = f'{save_figs_dir}check_retrieve_summary/'
    os.makedirs(table_dir, exist_ok=True)
    cells.to_csv(f'{table_dir}cell_table.csv', index=False)
    print(f'{cells.shape[0]} cells from {len(sessions)} sessions; '
          f'cell table in {table_dir}cell_table.csv')

    for level, member, title, folder, prefix, mode in iter_figures(cells, sessions):
        shown = select_cells(cells, member, mode)
        union = np.flatnonzero(np.any([shown[c] for c in CONDS], axis=0))
        if union.size == 0:
            print(f'{title}: nothing to plot under {mode}')
            continue
        ctx = dict(cells=cells, psth=psth, psth_sort=psth_sort)

        for align in align_to:
            rows, blocks = arrange_subplots(shown, ctx, align, mode)
            if show_median_duration:
                rows['_median_dur'] = _median_durations(cells, sessions, rows)
            fname = (f'{prefix}check_retrieve_{align}_sort-{_sort_tag()}_'
                     f'norm-{norm_method or "raw"}_{mode}')
            plot_summary(psth, rows, blocks, align, mode,
                         f'{title}, {align}-aligned', folder + fname + '.png')
            if write_legend:
                write_legend_file(folder + fname + '_legend.txt', title, align, mode,
                                  cells, sessions, {c: rows[c] for c in CONDS},
                                  blocks)


if __name__ == '__main__':
    main(recompute=False)
