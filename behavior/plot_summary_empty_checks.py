import numpy as np
import pandas as pd

import os
import sys
import warnings
import zlib
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import pop_normalize, cluster_ids_for_session
from cell_filters import filter_cells, apply_cell_filter
from event_psth import window_frames, event_psth_on_off
from format_behavior_data import (load_behavior_data, get_checks_raw,
                                  get_eating_bouts,
                                  _seed_provenance_timeline,
                                  get_expectation_status, SITE_EMPTY)
from spike_amplitudes import load_spike_amplitudes, select_trials_by_drift

import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, detrend

'''
Population heatmap of activity around EMPTY checks only: every good cell a
row, time on x.  A pared-down sibling of plot_summary_check_retrieve_responses.py
for the condition with the most data.

Empty = no seed in the site at the check, per the seed ledger
(get_expectation_status, SITE_EMPTY).  Checks are site interactions with no
seed change lasting <= max_check_dur (get_checks_raw).

Isolated trials only (exclude_overlaps): a check is dropped if its analysis
window overlaps any OTHER site interaction (check, cache or retrieval, from
count_data newSite/endSite) or any eating bout (newBeakPerch/endBeakPerch, as
get_eating_bouts).  Perch arrivals/departures are allowed in the window.

Onset- and offset-aligned panels (panels) sit side by side in one figure and
share one row order.  A *_legend.txt beside each figure has the definitions,
trial counts (including how many checks the overlap rule removed), and sort.

Pipeline
--------
1. per session: filter cells, pop_normalize each cell's whole-session trace,
   find isolated empty checks, average the normalized trace around them
   (plus split halves for a cross-validated peak sort).  Cached to disk.
2. pool sessions, attaching probe positions and saved rastermap order.
3. per figure: pick cells, sort, plot.
'''

''' Dataset '''
# switch between datasets here; everything that differs between them lives in
# DATASETS below, everything else in this file is shared so the two sets of
# figures are directly comparable
dataset = 'lh'                 # 'lh' | 'hpc'

DATASETS = {
    'lh': dict(
        label='LH',
        root_dir="Z:/Isabel/data/lhy_implants/",
        save_figs_dir="../figures/basic_neural_analysis/",
        data_file='good_session_data.npy',
        # cluster IDs: 'ids_file' = <data_dir>/aligned_spikes_ids.npy,
        # 'cluster_group' = good units in KS cluster_group.tsv
        cluster_ids='ids_file',
        use_stim_filter=False,
        exclude_birds=['LMN86'],                 # e.g. ['LIM63']
        exclude_sessions={},              # e.g. {'ROS107': ['260831']}
    ),
    'hpc': dict(
        label='HPC',
        root_dir="Z:/Isabel/data/hpc_implants/",
        save_figs_dir="../figures/basic_neural_analysis/hpc/",
        data_file='stim_session_data.npy',
        cluster_ids='cluster_group',
        # projection-nucleus cells only: cells on or beside stim-responsive
        # channels (format_chronic_stim.idx_cells_by_stim, as in
        # plot_summary_feeder_responses), applied through cell_filters
        use_stim_filter=True,
        exclude_birds=[],
        exclude_sessions={},
    ),
}
# when the stim filter is on, skip sessions it cannot be applied to (no stim
# data, or a cell-count mismatch) instead of keeping all their cells
require_stim_filter = True

''' File paths (from the dataset) '''
DS = DATASETS[dataset]
root_dir = DS['root_dir']
save_figs_dir = DS['save_figs_dir']
data_file = f"{root_dir}{DS['data_file']}"
cache_file = f"{root_dir}empty_check_summary_cache.npy"

''' Which data to include '''
birds_to_plot = None           # None = every bird in data_dict, or a list
exclude_birds = DS['exclude_birds']
exclude_sessions = DS['exclude_sessions']
plot_levels = ('bird', 'all')

''' Data params '''
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in (same as the single-cell code) '''
# shared across datasets; use_stim_filter comes from the dataset preset
CELL_FILTERS = dict(
    use_stim_filter = DS['use_stim_filter'],   # True = projection-nucleus cells only
    fr_thresh       = 0.1,        # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
max_check_dur = 1.5            # s; longer unchanged-content visits are not checks
removal_rule = 'cached_first'  # seed ledger settings: these decide whether a
use_init_counts = True         # site counts as empty (see get_expectation_status)

# a cell needs at least this many usable checks to be drawn
min_events = 3

''' Isolated trials '''
# drop checks whose analysis window overlaps another site interaction and/or
# an eating bout.  The window is the span of the PSTH windows in `panels`
# (onset window start to offset window end when both are shown), widened by
# overlap_pad on each side.  Perch events never count as overlaps.
exclude_overlaps = False
overlap_with = ('site', 'eating')  # any of 'site', 'eating'
overlap_pad = 0.0                  # s

''' Per-cell trial selection (unit drift) — same as the single-cell code '''
subsample_drift = True
subsample_metric = 'amplitude'     # 'firing rate' | 'amplitude'
subsample_thresh = 0.7             # keep events >= this x the session average
thresh_t_window = 1 * 60.0         # s centered on the event
amp_min_spikes = 1

''' PSTH windows '''
# heatmap panels, left to right, sharing one row order: ('onset',),
# ('offset',) or ('onset', 'offset')
panels = ('onset', 'offset')
psth_windows = {'onset': (-1.0, 1.0),   # s relative to check onset
                'offset': (-1.0, 1.0)}  # s relative to check offset
sigma_frames = 1                   # Gaussian smoothing (frames), 0 = none

''' Normalization '''
# 'pop_normalize'  format_waveform_data.pop_normalize on each cell's whole-
#                  session trace before event averaging, as in the feeder
#                  summary (SD units, 0 = running baseline)
# None             raw firing rate (Hz)
norm_method = 'pop_normalize'
pop_norm_kw = dict(std_reg=1e-2, baseline_window=30)

''' Sorting the y axis '''
# 'depth'      blocks of (bird, shank), each sorted by DV (get_probe_coords_lhy)
# 'rastermap'  blocks of (bird, session), each in the order saved by the
#              rastermap GUI (<data_dir>/<rastermap_file>, key 'isort')
# 'peak'       time of each cell's peak response, or the phase of its Hilbert
#              transform (see peak_sort['method'])
# 'labels'     user labels (cell_labels), then label_secondary_sort within label
# 'response'   mean activity in response_sort['window'], largest on top
# 'none'       pooling order
sort_by = 'peak'

# depth
shank_order = 'ap'                 # 'ap' = by mean shank AP across birds | 'bird'
ap_order = 'posterior_first'       # 'posterior_first' | 'anterior_first'
depth_order = 'dorsal_first'       # 'ventral_first' | 'dorsal_first'

# rastermap: isort indexes the rows of aligned_spikes.npy, before the cell filter
rastermap_file = 'aligned_spikes_embedding.npy'

# peak sort
peak_sort = dict(
    # 'peak'           time of each cell's peak (metric below)
    # 'hilbert_phase'  instantaneous phase of the Hilbert transform of each
    #                  cell's average response, read at phase_time -- the sort
    #                  in Payne & Aronov 2025 (Nature 643:1037), Fig. 4b
    method='peak',
    align=None,               # None = the first panel's alignment
    window=None,              # (t0, t1) s used for the peak search / Hilbert
                              # transform; None = whole PSTH
    metric='max',             # 'peak' only: 'max' peak | 'min' trough |
                              # 'absdev' largest deviation from the cell's mean
    # 'hilbert_phase' only:
    phase_time='pop_peak',    # s relative to the sort alignment at which the
                              # phase is read, or 'pop_peak' = the time of the
                              # peak of the population-average response (the
                              # paper used 187 ms, the peak of E-cell firing)
    phase_detrend='mean',     # 'mean' | 'linear' | None: removed before the
                              # transform (not stated in the paper; without it a
                              # cell's baseline rate dominates the phase)
    phase_cut_deg=-180,       # where the circular order is cut: the cell with
                              # phase just above this goes on top
    phase_reverse=True,       # True: descending phase, so cells that peak
                              # earlier sit on top, as in the peak sort
    split='alternate',        # cross-validated: 'alternate' (odd/even checks in
                              # time) | 'random'; or 'all' = sort on and plot all
                              # checks (NOT cross-validated; visualization only)
    seed=0,                   # 'random' only
    swap_halves=False,        # False: half 0 sorts, half 1 is plotted
    extra_sigma_frames=0,     # extra smoothing of the sorting half before the argmax
)

# user-defined cluster labels: csv path (bird, session, cluster_id, label) or a
# flat array in cell_table.csv order
cell_labels = None
label_order = None
label_secondary_sort = 'peak'      # 'depth' | 'rastermap' | 'peak' | 'response' | None

# 'response' sort: mean activity in window (s), relative to align
response_sort = dict(align=None, window=(0.0, 0.3))   # align None = first panel

# peak / response / none only: keep each bird's (or session's) cells together
block_by = None                    # None | 'bird' | 'session'

''' Cells without enough checks '''
# 'exclude'     leave them out
# 'blank_rows'  keep them as grey rows
# Session figures are skipped when the session has fewer than min_events
# usable checks.
missing = 'exclude'

''' Plotting params '''
clim = (-1, 1)                     # None = auto (symmetric 99th pct of |SD|, or 0-99th pct Hz)
cmap_norm = 'bwr'
cmap_raw = 'viridis'
nan_color = 'xkcd:light grey'
show_block_dividers = False        # solid between outer blocks, dashed within
show_block_labels = False
show_median_duration = False       # dotted line at the median check offset/onset
show_suptitle = True
write_legend = True
panel_size = (3.0, 6.0)            # width, height of each heatmap panel (in)
xtick_step = 0.5                   # s

title_size = 13
axis_label = 11
tick_label = 9
block_label_size = 7

CACHE_VERSION = 1                  # bump when compute_session's output changes


''' Derived settings '''
ALIGNS = tuple(panels)
if not ALIGNS or any(a not in ('onset', 'offset') for a in ALIGNS):
    raise ValueError("panels must be a non-empty tuple of 'onset' / 'offset'")
if peak_sort['split'] not in ('alternate', 'random', 'all'):
    raise ValueError("peak_sort['split'] must be 'alternate', 'random' or 'all'")
PEAK_CV = peak_sort['split'] != 'all'
if peak_sort['method'] not in ('peak', 'hilbert_phase'):
    raise ValueError("peak_sort['method'] must be 'peak' or 'hilbert_phase'")
_RUNTIME = {}                      # values computed at plot time, for the legend
SORT_ALIGN = peak_sort['align'] or ALIGNS[0]
RESPONSE_ALIGN = response_sort['align'] or ALIGNS[0]
for _a in (SORT_ALIGN, RESPONSE_ALIGN):
    if _a not in ALIGNS:
        raise ValueError(f"sort alignment {_a!r} is not one of panels {ALIGNS}")

WIN_FR, T_PTS = {}, {}
for _a in ALIGNS:
    _s, _e, T_PTS[_a] = window_frames(*psth_windows[_a], dt)
    WIN_FR[_a] = (_s, _e)


''' Step 1: per-session computation (cached) '''
def get_empty_checks(count_data, seed_struct):
    '''Onsets / offsets of every check of an empty site.'''
    on, off, site = get_checks_raw(count_data, seed_struct,
                                   max_check_dur=max_check_dur, dt=dt)
    status, _ = get_expectation_status(count_data, seed_struct, on, site,
                                       use_init_counts=use_init_counts,
                                       removal_rule=removal_rule)
    empty = status == SITE_EMPTY
    return on[empty], off[empty], int(on.shape[0])


def trial_windows(on, off, n_pad=0):
    '''
    [start, end) frames of each trial's analysis window: from the earliest
    to the latest edge of the PSTH windows in `panels`, padded by n_pad.
    '''
    starts = [(on if a == 'onset' else off) + WIN_FR[a][0] for a in ALIGNS]
    ends = [(on if a == 'onset' else off) + WIN_FR[a][1] for a in ALIGNS]
    return np.min(starts, axis=0) - n_pad, np.max(ends, axis=0) + n_pad


def overlap_flags(count_data, on, off):
    '''
    For each check, whether its analysis window overlaps another site
    interaction and/or an eating bout.  Intervals are inclusive of their end
    frame; the check's own interaction (same newSite and endSite) is ignored.

    Returns dict 'site' / 'eating' -> bool array, shape (n_checks,)
    '''
    w0, w1 = trial_windows(on, off, n_pad=int(round(overlap_pad / dt)))
    flags = {}
    if 'site' in overlap_with:
        s_on = np.asarray(count_data['newSite']).astype(int)
        s_off = np.asarray(count_data['endSite']).astype(int)
        hit = (s_on[None, :] < w1[:, None]) & (s_off[None, :] >= w0[:, None])
        own = (s_on[None, :] == on[:, None]) & (s_off[None, :] == off[:, None])
        flags['site'] = np.any(hit & ~own, axis=1)
    if 'eating' in overlap_with:
        e_on, e_off = get_eating_bouts(count_data)
        hit = (e_on[None, :] < w1[:, None]) & (e_off[None, :] >= w0[:, None])
        flags['eating'] = np.any(hit, axis=1)
    return flags


def split_events(on, seed_key):
    '''Two disjoint halves of the (already usable) checks, as index arrays.'''
    idx = np.argsort(on, kind='stable')
    if peak_sort['split'] == 'alternate':
        return [idx[0::2], idx[1::2]]
    seed = (zlib.crc32(seed_key.encode()) + int(peak_sort['seed'])) & 0xffffffff
    perm = np.random.default_rng(seed).permutation(idx)
    half = perm.shape[0] // 2
    return [np.sort(perm[:half]), np.sort(perm[half:])]


def average_events(trace, on, off, keep):
    '''
    Event averages of `trace` for each panel alignment, over the checks each
    cell keeps.  Cells with fewer than min_events kept checks are NaN.
    '''
    n_cells = trace.shape[0]
    out = {a: np.full((n_cells, T_PTS[a].shape[0]), np.nan) for a in ALIGNS}
    if on.shape[0] == 0:
        return out, np.zeros(n_cells, dtype=int)
    # event_psth divides by its dt to turn counts into Hz; a normalized trace
    # must not be rescaled, so pass 1
    psth_dt = dt if norm_method is None else 1.0
    # event_psth_on_off always computes both alignments and drops any event
    # whose onset OR offset window leaves the session.  With a single panel,
    # anchor both windows on that panel's event edge, so only the plotted
    # window decides which checks are usable.
    a_on = 'onset' if 'onset' in ALIGNS else 'offset'
    a_off = 'offset' if 'offset' in ALIGNS else 'onset'
    anchor = {'onset': on, 'offset': off}
    p_on, p_off, _, nu = event_psth_on_off(trace, anchor[a_on], anchor[a_off],
                                           WIN_FR[a_on], WIN_FR[a_off], psth_dt,
                                           sigma_frames=sigma_frames, keep=keep)
    n_used = nu[:, 0]
    enough = n_used >= min_events
    out[a_on][enough] = p_on[enough, 0]
    out[a_off][enough] = p_off[enough, 0]
    return out, n_used


def compute_session(data_dict, bird, session_id):
    '''Everything the figures need from one session, or None if no cells.'''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}behavior_data/"
    session_data = data_dict[bird][session_id]

    ''' Neural data + cell filters '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape
    avg_firing_rate = 10 ** session_data['waveform_props'][2]
    if DS['cluster_ids'] == 'ids_file':
        all_ids = np.load(f"{data_dir}aligned_spikes_ids.npy")
    elif DS['cluster_ids'] == 'cluster_group':
        all_ids = cluster_ids_for_session(data_dict, bird, session_id, root_dir)
    else:
        raise ValueError("cluster_ids must be 'ids_file' or 'cluster_group'")

    keep_cells, report = filter_cells(data_dict, bird, session_id, n_cells_raw,
                                      **CELL_FILTERS)
    if (CELL_FILTERS['use_stim_filter'] and require_stim_filter
            and 'stim' not in report['applied']):
        print('  stim filter could not be applied, skipping session '
              '(require_stim_filter=True)')
        return None
    filtered = apply_cell_filter(keep_cells, spike_fr=spike_fr,
                                 avg_firing_rate=avg_firing_rate, cluster_id=all_ids)
    spike_fr = filtered['spike_fr']
    cell_ids = filtered['cluster_id']
    n_cells = spike_fr.shape[0]
    if n_cells == 0:
        print('  no cells survived the filters, skipping')
        return None
    ks_dir = f"{bird}_{session_data['ephys_id']}/{session_data['ks_folder']}/"

    if norm_method == 'pop_normalize':
        trace = pop_normalize(spike_fr, dt=dt, **pop_norm_kw)
    elif norm_method is None:
        trace = spike_fr
    else:
        raise ValueError("norm_method must be 'pop_normalize' or None")

    ''' Empty checks: in bounds, then isolated '''
    seed_struct, count_data = load_behavior_data(data_dir)
    on, off, n_checks = get_empty_checks(count_data, seed_struct)
    n_empty = int(on.shape[0])
    w0, w1 = trial_windows(on, off)
    in_bounds = (w0 >= 0) & (w1 <= n_frames)
    flags = overlap_flags(count_data, on, off) if exclude_overlaps else {}
    overlapping = np.any(list(flags.values()), axis=0) if flags else np.zeros(n_empty, bool)
    use = in_bounds & ~overlapping
    counts = dict(n_checks=n_checks, n_empty=n_empty,
                  n_out_of_bounds=int(np.sum(~in_bounds)),
                  n_overlap=int(np.sum(in_bounds & overlapping)),
                  **{f'n_overlap_{k}': int(np.sum(in_bounds & v)) for k, v in flags.items()})
    on, off = on[use], off[use]
    print(f'  {n_empty} empty checks of {n_checks}; {counts["n_overlap"]} overlap another '
          f'event, {counts["n_out_of_bounds"]} out of bounds -> {on.shape[0]} used')

    ''' Drift masks '''
    amp_data = None
    if subsample_drift and subsample_metric == 'amplitude':
        try:
            amp_data = load_spike_amplitudes(session_dir, data_dir, ks_dir,
                                             cell_ids, n_frames, fps=fps)
        except FileNotFoundError as err:
            print(f'  no KS amplitudes ({err.filename}); no drift cut this session')
    keep = np.ones((n_cells, on.shape[0]), dtype=bool)
    if (on.shape[0] and subsample_drift and
            (subsample_metric != 'amplitude' or amp_data is not None)):
        keep = np.asarray(select_trials_by_drift(
            spike_fr, on, dt, subsample_metric, subsample_thresh, thresh_t_window,
            amp_data=amp_data, min_spikes=amp_min_spikes), dtype=bool)

    ''' Averages: all checks, and split halves for a cross-validated peak sort '''
    psth, n_used = average_events(trace, on, off, keep)
    split = None
    if PEAK_CV:
        halves = split_events(on, seed_key=f'{bird}_{session_id}')
        split = dict(psth=[], n_used=[], n_events=[])
        for h in halves:
            p, nu = average_events(trace, on[h], off[h], keep[:, h])
            split['psth'].append(p)
            split['n_used'].append(nu)
            split['n_events'].append(int(h.shape[0]))
    print(f'  {int(np.sum(n_used >= min_events))}/{n_cells} cells drawn')

    return dict(bird=bird, session=str(session_id), cluster_id=cell_ids.astype(int),
                row_idx=np.flatnonzero(keep_cells), n_cells_raw=n_cells_raw,
                avg_fr=np.asarray(filtered['avg_firing_rate'], dtype=float),
                psth=psth, n_used=n_used, n_events=int(on.shape[0]),
                durations=(off - on) * dt, counts=counts, split=split)


def compute_params():
    '''Settings that change step 1; the cache is rebuilt if any differ.'''
    return dict(version=CACHE_VERSION, dataset=dataset, CELL_FILTERS=dict(CELL_FILTERS),
                cluster_ids=DS['cluster_ids'], require_stim_filter=require_stim_filter,
                norm_method=norm_method, pop_norm_kw=dict(pop_norm_kw),
                max_check_dur=max_check_dur, removal_rule=removal_rule,
                use_init_counts=use_init_counts, min_events=min_events,
                exclude_overlaps=exclude_overlaps, overlap_with=tuple(overlap_with),
                overlap_pad=overlap_pad,
                subsample_drift=subsample_drift, subsample_metric=subsample_metric,
                subsample_thresh=subsample_thresh, thresh_t_window=thresh_t_window,
                panels=ALIGNS, psth_windows={a: psth_windows[a] for a in ALIGNS},
                sigma_frames=sigma_frames, fps=fps,
                split=dict(split=peak_sort['split'], seed=peak_sort['seed']),
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
    '''Position of each kept cell in the saved rastermap order; NaN if absent.'''
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
    Copies of the sessions holding only the held-out half (averages, counts,
    durations), with the sorting half kept aside under 'sort_psth'.
    '''
    h_sort = 1 if peak_sort['swap_halves'] else 0
    h_plot = 1 - h_sort
    out = []
    for s in sessions:
        sp = s['split']
        s2 = dict(s)
        s2['psth'] = sp['psth'][h_plot]
        s2['n_used'] = sp['n_used'][h_plot]
        s2['n_events'] = sp['n_events'][h_plot]
        s2['sort_psth'] = sp['psth'][h_sort]
        out.append(s2)
    return out


def pool_sessions(sessions, data_dict):
    '''One row per cell across every session, and matching average stacks.'''
    need_rm = uses_method('rastermap')
    tables = []
    for s_idx, s in enumerate(sessions):
        pos, shank = get_cell_positions(data_dict[s['bird']][s['session']],
                                        s['n_cells_raw'], s['row_idx'])
        t = pd.DataFrame(dict(bird=s['bird'], session=s['session'],
                              cluster_id=s['cluster_id'], shank=shank,
                              ml=pos[:, 0], ap=pos[:, 1], dv=pos[:, 2],
                              avg_fr=s['avg_fr'], sess_idx=s_idx,
                              n_ev=s['n_events'], n_used=s['n_used']))
        t['rm_rank'] = (get_rastermap_rank(s['bird'], s['session'], s['n_cells_raw'],
                                           s['row_idx']) if need_rm else np.nan)
        tables.append(t)
    cells = pd.concat(tables, ignore_index=True)
    cells['shank_name'] = [shank_name(i) for i in cells['shank']]
    cells['bird_shank'] = cells['bird'] + ' ' + cells['shank_name']
    cells['bird_session'] = cells['bird'] + ' ' + cells['session']
    cells['shank_ap'] = cells.groupby('bird_shank')['ap'].transform('mean')

    psth = {a: np.concatenate([s['psth'][a] for s in sessions], axis=0) for a in ALIGNS}
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


''' Step 3a: sorting '''
def peak_times(sub, ctx):
    '''
    Peak time (s, start of the peak bin) of each cell in the sorting data
    (the sorting half, or all checks with split='all'), at SORT_ALIGN.
    NaN where the cell has too few checks or a flat average.
    '''
    t = T_PTS[SORT_ALIGN]
    w = (np.ones(t.shape[0], dtype=bool) if peak_sort['window'] is None else
         (t >= peak_sort['window'][0]) & (t < peak_sort['window'][1]))
    X = ctx['psth_sort'][SORT_ALIGN][np.asarray(sub, dtype=int)][:, w]
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


def _sort_window():
    t = T_PTS[SORT_ALIGN]
    w = (np.ones(t.shape[0], dtype=bool) if peak_sort['window'] is None else
         (t >= peak_sort['window'][0]) & (t < peak_sort['window'][1]))
    return t, w


def phase_reference_time(ctx):
    '''
    Time (s, relative to SORT_ALIGN) at which the Hilbert phase is read.
    'pop_peak' = peak of the average sorting-data response over every pooled
    cell with data, so the reference is the same in every figure.
    '''
    if 'phase_time' in _RUNTIME:
        return _RUNTIME['phase_time']
    pt = peak_sort['phase_time']
    if pt == 'pop_peak':
        t, w = _sort_window()
        X = ctx['psth_sort'][SORT_ALIGN][:, w]
        X = X[np.all(np.isfinite(X), axis=1)]
        if X.shape[0] == 0:
            raise ValueError('no cells with data to find the population peak')
        pt = float(t[w][np.argmax(X.mean(axis=0))])
    _RUNTIME['phase_time'] = float(pt)
    return float(pt)


def hilbert_phases(sub, ctx):
    '''
    Instantaneous phase (degrees, -180 to 180) of the Hilbert transform of
    each cell's average response in the sorting data, read at
    phase_reference_time.  Following Payne & Aronov 2025: transform the mean
    firing rate, store the phase at a fixed time after the event.  NaN where
    the cell has too few checks or a flat average.

    The transform is taken over the sort window only, and FFT-based, so
    phases read close to the window edges are distorted; keep phase_time
    well inside the window.
    '''
    t, w = _sort_window()
    tw = t[w]
    ref = phase_reference_time(ctx)
    k = int(np.argmin(np.abs(tw - ref)))
    X = ctx['psth_sort'][SORT_ALIGN][np.asarray(sub, dtype=int)][:, w]
    out = np.full(X.shape[0], np.nan)
    valid = np.all(np.isfinite(X), axis=1) & (np.ptp(np.nan_to_num(X), axis=1) > 0)
    if not np.any(valid):
        return out
    Xv = X[valid]
    if peak_sort['extra_sigma_frames'] > 0:
        Xv = gaussian_filter1d(Xv, peak_sort['extra_sigma_frames'], axis=1, mode='nearest')
    if peak_sort['phase_detrend'] == 'mean':
        Xv = Xv - Xv.mean(axis=1, keepdims=True)
    elif peak_sort['phase_detrend'] == 'linear':
        Xv = detrend(Xv, axis=1, type='linear')
    elif peak_sort['phase_detrend'] is not None:
        raise ValueError("peak_sort['phase_detrend'] must be 'mean', 'linear' or None")
    out[valid] = np.degrees(np.angle(hilbert(Xv, axis=1)[:, k]))
    return out


def phase_sort_key(phase_deg):
    '''Position along the circle, starting at phase_cut_deg; NaN stays NaN.'''
    key = np.mod(phase_deg - peak_sort['phase_cut_deg'], 360.0)
    return -key if peak_sort['phase_reverse'] else key


def order_within(sub, method, ctx):
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
        key = (phase_sort_key(hilbert_phases(sub, ctx))
               if peak_sort['method'] == 'hilbert_phase' else peak_times(sub, ctx))
    elif method == 'response':
        t = T_PTS[RESPONSE_ALIGN]
        w = (t >= response_sort['window'][0]) & (t < response_sort['window'][1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            key = -np.nanmean(ctx['psth'][RESPONSE_ALIGN][sub][:, w], axis=1)
    else:
        raise ValueError(f'unknown sort {method}')
    key = np.where(np.isnan(key), np.inf, key)
    return sub[np.argsort(key, kind='stable')]


def _label_key(x):
    if x == 'unlabeled':
        return (2, 0, '')
    try:
        return (0, float(x), '')
    except (TypeError, ValueError):
        return (1, 0, str(x))


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
            key = np.where(np.isnan(ap), np.inf, ap)
            return [names[i] for i in np.argsort(key, kind='stable')]
        return sorted(names, key=lambda n: (birds.index(info.loc[n, 'bird']),
                                            info.loc[n, 'shank_name'] == '?',
                                            info.loc[n, 'shank_name']))
    return names


def order_cells(idx, ctx):
    '''Ordered pooled indices, plus up to two block levels per row.'''
    cells = ctx['cells']
    idx = np.asarray(idx, dtype=int)
    if idx.shape[0] == 0:
        return idx, dict(outer=None, inner=None)
    if sort_by == 'labels':
        if cells['label'].isna().all():
            raise ValueError("sort_by='labels' needs cell_labels")
        within, outer_col = label_secondary_sort, 'label'
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
            srt = order_within(i_sub, within, ctx)
            ordered.append(srt)
            outer += [o_name] * srt.shape[0]
            inner += [i_name] * srt.shape[0]
    outer, inner = np.asarray(outer, dtype=object), np.asarray(inner, dtype=object)
    if outer_col is None or (outer_col != 'label' and len(set(outer)) == 1):
        outer = None
    if inner_col is None or len(set(inner)) == 1:
        inner = None
    return np.concatenate(ordered), dict(outer=outer, inner=inner)


''' Step 3b: plotting '''
def _runs(keys):
    change = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    return list(zip([0] + change, change + [len(keys)]))


def _block_runs(blocks):
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
    vals = np.concatenate([psth[a][rows].ravel() for a in ALIGNS]) if rows.size else np.zeros(1)
    vals = vals[np.isfinite(vals)]
    if norm_method is None:
        return (0.0, float(np.percentile(vals, 99)) if vals.size else 1.0)
    v = float(np.percentile(np.abs(vals), 99)) if vals.size else 1.0
    return (-v, v)


def plot_summary(psth, rows, blocks, median_dur, title, save_path):
    n_p = len(ALIGNS)
    n = rows.shape[0]
    cm = plt.get_cmap(cmap_raw if norm_method is None else cmap_norm).copy()
    cm.set_bad(nan_color)
    line_color = 'w' if norm_method is None else 'k'
    vmin, vmax = _clim(psth, rows)

    f, ax = plt.subplots(1, n_p, figsize=(panel_size[0] * n_p, panel_size[1]),
                         squeeze=False, gridspec_kw=dict(wspace=0.15))
    ax = ax[0]
    for col, align in enumerate(ALIGNS):
        a = ax[col]
        t = T_PTS[align]
        x0, x1 = t[0], t[-1] + dt
        ticks = np.arange(np.ceil(x0 / xtick_step), np.floor(x1 / xtick_step) + 1) * xtick_step
        im = a.imshow(psth[align][rows], aspect='auto', cmap=cm, vmin=vmin, vmax=vmax,
                      interpolation='none', extent=(x0, x1, n - 0.5, -0.5))
        a.axvline(0, color=line_color, ls='--', lw=0.8)
        if show_median_duration and np.isfinite(median_dur):
            med = median_dur if align == 'onset' else -median_dur
            if x0 < med < x1:
                a.axvline(med, color=line_color, ls=':', lw=0.8)
        _draw_blocks(a, blocks, draw_labels=(col == 0))
        a.set_xlim(x0, x1)
        a.set_xticks(ticks)
        a.set_xticklabels([f'{v:g}' for v in ticks])
        a.tick_params(labelsize=tick_label)
        a.set_xlabel(f'time from check {align} (s)', fontsize=axis_label)
        if col > 0:
            a.set_yticks([])
    sort_word = ('Hilbert phase' if sort_by == 'peak' and peak_sort['method'] == 'hilbert_phase'
                 else sort_by)
    ax[0].set_ylabel(f'cells (sorted by {sort_word})', fontsize=axis_label)

    pos = ax[-1].get_position()
    cax = f.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.35, 0.015, pos.height * 0.3])
    cb = f.colorbar(im, cax=cax)
    cticks = [vmin, vmax] if norm_method is None else [vmin, 0, vmax]
    cb.set_ticks(cticks)
    cb.set_ticklabels([f'{v:.3g}' for v in cticks])
    cb.ax.tick_params(labelsize=tick_label)
    cb.set_label('rate (Hz)' if norm_method is None else 'activity (SD)', fontsize=tick_label)

    if show_suptitle:
        f.suptitle(title, fontsize=title_size, y=0.98)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(f)


''' Figure legend text '''
_OVERLAP_NAMES = {'site': 'site interaction', 'eating': 'eating bout'}


def _n(k, word):
    return f'{k} {word}' + ('' if k == 1 else 's')


def _filter_text():
    parts = ['all good units' if CELL_FILTERS.get('cell_type', 'all') == 'all'
             else f"{CELL_FILTERS['cell_type']} units only"]
    if CELL_FILTERS.get('fr_thresh') is not None:
        parts.append(f"session firing rate > {CELL_FILTERS['fr_thresh']} Hz")
    if CELL_FILTERS.get('use_stim_filter'):
        parts.append('in the projection nucleus (on or beside stim-responsive '
                     'channels)' + ('; sessions without stim data excluded'
                                    if require_stim_filter else ''))
    return ', '.join(parts)


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
    if method == 'peak' and peak_sort['method'] == 'hilbert_phase':
        win = ('the whole window' if peak_sort['window'] is None
               else f"{peak_sort['window'][0]} to {peak_sort['window'][1]} s")
        ref = _RUNTIME.get('phase_time', float('nan'))
        ref_txt = (' (the peak of the population-average response)'
                   if peak_sort['phase_time'] == 'pop_peak' else '')
        det = {'mean': 'mean-subtracted ', 'linear': 'linearly detrended ',
               None: ''}[peak_sort['phase_detrend']]
        order = 'descending' if peak_sort['phase_reverse'] else 'ascending'
        return (f'ordered by the instantaneous phase of the Hilbert transform of each '
                f"cell's {det}average response ({win}, {SORT_ALIGN}-aligned), read "
                f'{ref * 1000:.0f} ms after check {SORT_ALIGN}{ref_txt}, as in Payne & Aronov 2025 '
                f'(Fig. 4b); phases in {order} order starting from '
                f"{peak_sort['phase_cut_deg']:g} deg")
    if method == 'peak':
        win = ('the whole window' if peak_sort['window'] is None
               else f"{peak_sort['window'][0]} to {peak_sort['window'][1]} s")
        what = {'max': 'peak', 'min': 'trough',
                'absdev': 'largest deviation from its mean'}[peak_sort['metric']]
        return (f"ordered by the time of each cell's {what} ({win}, "
                f'{SORT_ALIGN}-aligned), earliest on top')
    if method == 'response':
        w = response_sort['window']
        return (f'ordered by mean activity {w[0]} to {w[1]} s after check '
                f'{RESPONSE_ALIGN}, largest on top')
    return 'in pooling order'


def write_legend_file(path, title, cells, sessions, rows, blocks):
    L = [title, '']
    span = ' and '.join(f'{a} ({psth_windows[a][0]:g} to {psth_windows[a][1]:g} s)'
                        for a in ALIGNS)
    L.append('Each row is one cell and each column a time bin '
             f'({dt * 1000:.0f} ms) relative to check {span}; dashed line at 0. '
             'Empty checks: site interactions with no seed change lasting '
             f'<= {max_check_dur} s, at a site the seed ledger scores as empty.')
    if exclude_overlaps:
        what = ' or '.join({'site': 'another site interaction (check, cache or retrieval)',
                            'eating': 'an eating bout'}[k] for k in overlap_with)
        pad = f', widened by {overlap_pad:g} s each side' if overlap_pad else ''
        L.append(f'Isolated checks only: checks whose analysis window (spanning the '
                 f'plotted windows{pad}) overlapped {what} were excluded; perch '
                 'visits were allowed.')
    smooth = (f'Gaussian-smoothed (sigma {sigma_frames * dt * 1000:.0f} ms)'
              if sigma_frames > 0 else 'unsmoothed')
    clim_txt = (f'{clim[0]:g} to {clim[1]:g}' if clim is not None else
                ('0 to the 99th percentile' if norm_method is None
                 else '+/- the 99th percentile of |activity|'))
    if norm_method == 'pop_normalize':
        L.append("Activity: each cell's instantaneous firing rate over the whole session "
                 f"was divided by its session SD (+{pop_norm_kw['std_reg']:g}) and a "
                 f"running-average baseline (window {pop_norm_kw['baseline_window']}) "
                 'was subtracted (pop_normalize), so units are SDs from baseline; this '
                 f'trace was averaged across checks and {smooth}. Colour limits '
                 f'{clim_txt} SD (red above, blue below baseline).')
    else:
        L.append(f'Activity: trial-averaged firing rate in Hz, {smooth}. '
                 f'Colour limits {clim_txt}.')
    if subsample_drift:
        L.append(f"Per cell, checks were excluded where the unit's {subsample_metric} "
                 f'in a {thresh_t_window / 60:g}-min window around the check was below '
                 f'{subsample_thresh}x its session average.')
    L.append(f'A cell is drawn only with >= {min_events} usable checks. '
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
        split = ('alternating checks in time' if peak_sort['split'] == 'alternate'
                 else 'a random split')
        L.append(f'Cross-validated: each session\'s checks were split in half ({split}); '
                 f'half {h} set the order and the heatmap shows only half {1 - h}.')
    elif uses_method('peak'):
        L.append('Not cross-validated: the order was set from the same checks that are '
                 'plotted, so the sequence is partly built in by the sort.')
    o_runs, i_runs = _block_runs(blocks)
    if o_runs or i_runs:
        L.append('Blocks, top to bottom (rows):')
        names = blocks['inner'] if i_runs else blocks['outer']
        for a, b in (i_runs or o_runs):
            outer = f"{blocks['outer'][a]} / " if (i_runs and blocks['outer'] is not None) else ''
            L.append(f'  {outer}{names[a]}: rows {a + 1}-{b}')
    L.append('')

    has = cells['n_used'].to_numpy()[rows] >= min_events
    s_used = np.unique(cells['sess_idx'].to_numpy()[rows][has])
    tot = lambda k: int(sum(sessions[s]['counts'][k] for s in s_used))
    n_plot = int(sum(sessions[s]['n_events'] for s in s_used))
    plotted = 'in the plotted half' if (uses_method('peak') and PEAK_CV) else 'used'
    per_cell = cells['n_used'].to_numpy()[rows][has]
    L.append(f'{int(has.sum())} cells'
             + (f' (of {rows.shape[0]} rows; grey rows had too few checks)'
                if has.sum() < rows.shape[0] else '')
             + f', {_n(len(s_used), "session")}, '
             f'{_n(cells.loc[rows[has], "bird"].nunique(), "bird")}.')
    L.append(f'Checks: {tot("n_checks")} checks, {tot("n_empty")} of them empty; '
             f'{tot("n_out_of_bounds")} too close to the session edge'
             + (f', {tot("n_overlap")} overlapping another event ('
                + ', '.join(f'{_OVERLAP_NAMES[k]}: {tot("n_overlap_" + k)}' for k in overlap_with)
                + ')' if exclude_overlaps else '')
             + f'; {n_plot} {plotted}'
             + (f', median {np.median(per_cell):g} per cell after the drift cut'
                if per_cell.size else '') + '.')
    durs = [sessions[s]['durations'] for s in s_used]
    durs = np.concatenate(durs) if durs else np.zeros(0)
    if durs.size:
        L.append(f'Median check duration {np.median(durs):.2f} s.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


''' Figure groups '''
def iter_figures(cells, sessions):
    birds = cells['bird'].to_numpy(object)
    sess_idx = cells['sess_idx'].to_numpy()
    for level in plot_levels:
        if level == 'session':
            for s_idx, s in enumerate(sessions):
                if s['n_events'] < min_events:
                    print(f'skipping {s["bird"]}_{s["session"]} figure: '
                          f'{s["n_events"]} usable checks')
                    continue
                yield (sess_idx == s_idx, f'{DS["label"]} {s["bird"]} {s["session"]}, empty checks',
                       f'{save_figs_dir}{s["bird"]}/empty_check_summary/', f'{s["session"]}_')
        elif level == 'bird':
            for bird in pd.unique(birds):
                yield (birds == bird, f'{DS["label"]} {bird}, empty checks',
                       f'{save_figs_dir}{bird}/empty_check_summary/', f'{bird}_')
        elif level == 'all':
            yield (np.ones(cells.shape[0], dtype=bool), f'{DS["label"]} all birds, empty checks',
                   f'{save_figs_dir}empty_check_summary/', 'all_')
        else:
            raise ValueError(f'unknown plot level {level}')


def _sort_tag():
    tag = sort_by
    if sort_by == 'labels' and label_secondary_sort:
        tag += f'-{label_secondary_sort}'
    if uses_method('peak'):
        tag += (('-phase' if peak_sort['method'] == 'hilbert_phase' else '')
                + f"-{SORT_ALIGN}" + ('' if PEAK_CV else '-all'))
    if sort_by in ('peak', 'response', 'none') and block_by:
        tag += f'-by{block_by}'
    return tag


def main(recompute=False):
    data_dict = np.load(data_file, allow_pickle=True).item()
    sessions = load_or_compute_sessions(data_dict, recompute=recompute)
    if not sessions:
        print('no sessions with cells')
        return

    # cross-validated peak sort: plot the held-out half; split='all' sorts on
    # and plots every check
    if uses_method('peak') and PEAK_CV:
        sessions = apply_split(sessions)
    cells, psth, psth_sort = pool_sessions(sessions, data_dict)
    if uses_method('peak') and not PEAK_CV:
        psth_sort = psth
    cells = assign_labels(cells)
    table_dir = f'{save_figs_dir}empty_check_summary/'
    os.makedirs(table_dir, exist_ok=True)
    cells.to_csv(f'{table_dir}cell_table.csv', index=False)
    print(f'{cells.shape[0]} cells from {len(sessions)} sessions; '
          f'cell table in {table_dir}cell_table.csv')

    ctx = dict(cells=cells, psth=psth, psth_sort=psth_sort)
    valid = cells['n_used'].to_numpy() >= min_events
    for member, title, folder, prefix in iter_figures(cells, sessions):
        shown = member & valid if missing == 'exclude' else member
        if not np.any(shown & valid):
            print(f'{title}: no cells with >= {min_events} usable checks')
            continue
        rows, blocks = order_cells(np.flatnonzero(shown), ctx)
        s_used = np.unique(cells['sess_idx'].to_numpy()[rows])
        durs = [sessions[s]['durations'] for s in s_used]
        durs = np.concatenate(durs) if durs else np.zeros(0)
        median_dur = float(np.median(durs)) if durs.size else np.nan

        fname = (f'{prefix}empty_checks_{"-".join(ALIGNS)}_sort-{_sort_tag()}_'
                 f'norm-{norm_method or "raw"}')
        plot_summary(psth, rows, blocks, median_dur, title, folder + fname + '.png')
        if write_legend:
            write_legend_file(folder + fname + '_legend.txt', title, cells, sessions,
                              rows, blocks)


if __name__ == '__main__':
    main(recompute=False)
