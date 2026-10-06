import numpy as np

import os
import sys
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import pop_normalize
from cell_filters import filter_cells, apply_cell_filter
from event_psth import window_frames, event_psth_on_off
from format_behavior_data import (load_behavior_data, get_event_times,
                                  get_saccades, get_preceding_saccades,
                                  SITE_EMPTY, SITE_BAITED, SITE_CACHED,
                                  SITE_STATUS_NAMES)
from spike_amplitudes import load_spike_amplitudes, select_trials_by_drift

import matplotlib.pyplot as plt

'''
Population heatmaps: every cell a row, activity around the saccade preceding
each check / retrieval / cache on the left and around the event itself on the
right, with one shared row order (peak time in sort_panel).

Preceding saccade and event exclusions as in plot_saccade_event_activity.py.  The dotted line in
each panel marks the median saccade-to-event lag.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = "../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Which data to include '''
birds_to_plot = None           # None = every bird in data_dict, or a list
plot_levels = ('bird', 'all')

''' Data params '''
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — same as the single-cell code '''
CELL_FILTERS = dict(
    use_stim_filter = False,
    fr_thresh       = 0.1,
    cell_type       = 'all',
)

''' Event params '''
event_key = 'check'            # 'check' | 'retrieval' | 'cache'
align_event = 'onset'          # right panel aligned to 'onset' | 'offset'
plot_groups = None             # statuses to pool, e.g. [SITE_EMPTY]; None = all
max_check_dur = 1.5
discovered_bait = 'cached'
removal_rule = 'cached_first'
use_init_counts = True

# per-session averages are cached here and reused while the settings that
# feed them are unchanged, so re-plotting (sort, clim, levels...) is instant
cache_file = f"{root_dir}saccade_{event_key}_summary_cache.npy"

''' Saccade params '''
saccade_params = None          # None = SACCADE_PARAMS defaults
max_saccade_lag = np.inf       # s; max saccade-to-onset time (np.inf = no limit)
different_perch = True         # saccade must be made from a different perch

min_events = 3                 # per cell, else the row is dropped

''' Per-cell trial selection (unit drift) '''
subsample_drift = True
subsample_metric = 'amplitude'
subsample_thresh = 0.7
thresh_t_window = 60.0
amp_min_spikes = 1

''' PSTH windows (s) '''
windows = {'saccade': (-0.4, 0.4), 'event': (-0.4, 0.4)}
sigma_frames = 1

''' Normalization '''
norm_method = 'pop_normalize'  # 'pop_normalize' (SD from baseline) | None (Hz)
pop_norm_kw = dict(std_reg=1e-2, baseline_window=30)

''' Sorting '''
sort_panel = 'saccade'         # 'saccade' | 'event': whose peak time sets the order
cv_sort = True                 # sort on odd events, plot even events

''' Plotting params '''
clim = (-1, 1)                 # None = auto
panel_size = (3.0, 6.0)        # in, per panel
title_size = 13
axis_label = 11
tick_label = 9

CACHE_VERSION = 1              # bump when compute_session's output changes

WIN = {k: window_frames(*w, dt) for k, w in windows.items()}   # (start, end, t_pts)
XLABEL = {'saccade': 'time from saccade peak (s)',
          'event': f'time from {event_key} {align_event} (s)'}


def compute_session(data_dict, bird, session_id):
    '''Saccade- and event-aligned averages for every cell, or None.'''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}behavior_data/"

    ''' Neural data + cell filter '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")
    n_cells_raw, n_frames = spike_fr.shape
    all_ids = np.load(f"{data_dir}aligned_spikes_ids.npy")
    keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw, **CELL_FILTERS)
    filtered = apply_cell_filter(keep_cells, spike_fr=spike_fr, cluster_id=all_ids)
    spike_fr, cell_ids = filtered['spike_fr'], filtered['cluster_id']
    n_cells = spike_fr.shape[0]
    if n_cells == 0:
        return None
    if norm_method == 'pop_normalize':
        trace, psth_dt = pop_normalize(spike_fr, dt=dt, **pop_norm_kw), 1.0
    else:
        trace, psth_dt = spike_fr, dt

    ''' Events and their preceding saccades '''
    seed_struct, count_data = load_behavior_data(data_dir)
    onsets, offsets, _, status = get_event_times(
        count_data, seed_struct, event_key, max_check_dur=max_check_dur, dt=dt,
        use_init_counts=use_init_counts, removal_rule=removal_rule,
        discovered_bait=discovered_bait)
    saccade_time, saccade_perch_idx = get_saccades(data_dir, count_data, fps=fps,
                                                   params=saccade_params)
    prev, blocked = get_preceding_saccades(onsets, saccade_time, saccade_perch_idx,
                                           count_data, max_lag=max_saccade_lag,
                                           fps=fps, different_perch=different_perch)
    use = prev >= 0
    if plot_groups is not None:
        use &= np.isin(status, plot_groups)
    sacc = saccade_time[prev[use]]
    align = (onsets if align_event == 'onset' else offsets)[use]
    print(f'  {int(use.sum())} of {onsets.shape[0]} {event_key}s used '
          f'({int(np.sum((prev < 0) & ~blocked))} without a preceding saccade, '
          f'{int(blocked.sum())} with another event in between)')
    if align.shape[0] < min_events:
        return None

    ''' Drift masks '''
    keep = np.ones((n_cells, align.shape[0]), dtype=bool)
    if subsample_drift:
        try:
            amp_data = None
            if subsample_metric == 'amplitude':
                sd = data_dict[bird][session_id]
                amp_data = load_spike_amplitudes(
                    session_dir, data_dir, f"{bird}_{sd['ephys_id']}/{sd['ks_folder']}/",
                    cell_ids, n_frames, fps=fps)
            keep = select_trials_by_drift(spike_fr, align, dt, subsample_metric,
                                          subsample_thresh, thresh_t_window,
                                          amp_data=amp_data, min_spikes=amp_min_spikes)
        except FileNotFoundError as err:
            print(f'  no KS amplitudes ({err.filename}), no drift cut')

    def average(idx):
        '''Averages over events idx; nan for cells with < min_events of them.'''
        out = {k: np.full((n_cells, WIN[k][2].shape[0]), np.nan) for k in WIN}
        if idx.shape[0] == 0:
            return out
        p_sacc, p_event, _, n_used = event_psth_on_off(
            trace, sacc[idx], align[idx], WIN['saccade'][:2], WIN['event'][:2],
            psth_dt, sigma_frames=sigma_frames, keep=keep[:, idx])
        ok = n_used[:, 0] >= min_events
        out['saccade'][ok], out['event'][ok] = p_sacc[ok, 0], p_event[ok, 0]
        return out

    chrono = np.argsort(align, kind='stable')
    if cv_sort:
        sort, plot = average(chrono[0::2]), average(chrono[1::2])
    else:
        sort = plot = average(chrono)
    return dict(bird=bird, plot=plot, sort=sort, lags=(align - sacc) * dt)


def plot_heatmap(psth, rows, median_lag, title, save_path):
    cmap = plt.get_cmap('bwr' if norm_method else 'viridis')
    vmin, vmax = clim if clim is not None else (None, None)
    f, ax = plt.subplots(1, 2, figsize=(panel_size[0] * 2, panel_size[1]),
                         gridspec_kw=dict(wspace=0.15))
    for a, k, lag_line in zip(ax, ('saccade', 'event'), (median_lag, -median_lag)):
        t = WIN[k][2]
        im = a.imshow(psth[k][rows], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='none', extent=(t[0], t[-1] + dt, rows.shape[0] - 0.5, -0.5))
        a.axvline(0, color='k', ls='--', lw=0.8)
        if t[0] < lag_line < t[-1] + dt:
            a.axvline(lag_line, color='k', ls=':', lw=0.8)
        a.set_xlabel(XLABEL[k], fontsize=axis_label)
        a.tick_params(labelsize=tick_label)
    ax[1].set_yticks([])
    ax[0].set_ylabel(f'cells (sorted by {sort_panel} peak'
                     f'{", cross-validated" if cv_sort else ""})', fontsize=axis_label)

    pos = ax[-1].get_position()
    cax = f.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.35, 0.015, pos.height * 0.3])
    cb = f.colorbar(im, cax=cax)
    cb.set_label('activity (SD)' if norm_method else 'rate (Hz)', fontsize=tick_label)
    cb.ax.tick_params(labelsize=tick_label)

    f.suptitle(title, fontsize=title_size, y=0.98)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(f)


def compute_params():
    '''Every setting that changes compute_session; the cache is rebuilt if any differ.'''
    return dict(version=CACHE_VERSION, fps=fps, CELL_FILTERS=dict(CELL_FILTERS),
                birds_to_plot=birds_to_plot, event_key=event_key,
                align_event=align_event, plot_groups=plot_groups,
                max_check_dur=max_check_dur, discovered_bait=discovered_bait,
                removal_rule=removal_rule, use_init_counts=use_init_counts,
                saccade_params=saccade_params, max_saccade_lag=max_saccade_lag,
                different_perch=different_perch, min_events=min_events,
                subsample_drift=subsample_drift, subsample_metric=subsample_metric,
                subsample_thresh=subsample_thresh, thresh_t_window=thresh_t_window,
                amp_min_spikes=amp_min_spikes, windows=dict(windows),
                sigma_frames=sigma_frames, norm_method=norm_method,
                pop_norm_kw=dict(pop_norm_kw), cv_sort=cv_sort)


def load_or_compute_sessions(recompute=False):
    params = compute_params()
    if not recompute and os.path.isfile(cache_file):
        cached = np.load(cache_file, allow_pickle=True).item()
        if cached.get('params') == params:
            print(f'loaded {len(cached["sessions"])} sessions from {cache_file}')
            return cached['sessions']
        print('cache was built with different settings, recomputing')

    data_dict = np.load(data_file, allow_pickle=True).item()
    sessions = []
    for bird in (birds_to_plot or list(data_dict.keys())):
        for session_id in data_dict[bird]['all_sessions']:
            if not {'behavior', 'ephys'} <= set(data_dict[bird][session_id]['preprocessed_data']):
                continue
            print(f'{bird}_{session_id}')
            res = compute_session(data_dict, bird, session_id)
            if res is not None:
                sessions.append(res)
    np.save(cache_file, dict(params=params, sessions=sessions), allow_pickle=True)
    return sessions


def main(recompute=False):
    sessions = load_or_compute_sessions(recompute=recompute)
    if not sessions:
        print('no sessions with usable events')
        return

    figures = []
    if 'bird' in plot_levels:
        for bird in dict.fromkeys(s['bird'] for s in sessions):
            figures.append((bird, f'{save_figs_dir}{bird}/',
                            [s for s in sessions if s['bird'] == bird]))
    if 'all' in plot_levels:
        figures.append(('all birds', save_figs_dir, sessions))

    group_txt = ('' if plot_groups is None else
                 ' (' + ', '.join(SITE_STATUS_NAMES[g] for g in plot_groups) + ')')
    for name, folder, group in figures:
        plot = {k: np.concatenate([s['plot'][k] for s in group]) for k in WIN}
        sort = np.concatenate([s['sort'][sort_panel] for s in group])
        valid = np.flatnonzero(np.isfinite(sort).all(axis=1) &
                               np.isfinite(plot['saccade']).all(axis=1) &
                               np.isfinite(plot['event']).all(axis=1))
        if valid.shape[0] == 0:
            print(f'{name}: no cells with >= {min_events} usable events')
            continue
        rows = valid[np.argsort(np.argmax(sort[valid], axis=1), kind='stable')]
        median_lag = float(np.median(np.concatenate([s['lags'] for s in group])))

        title = f'{name}, {event_key}s{group_txt}: {rows.shape[0]} cells'
        fname = (f"{name.replace(' ', '_')}_saccade_{event_key}_{align_event}_"
                 f"sort-{sort_panel}{'-cv' if cv_sort else ''}.png")
        plot_heatmap(plot, rows, median_lag, title,
                     f'{folder}saccade_{event_key}_summary/{fname}')


if __name__ == '__main__':
    main(recompute=False)   # True forces a rebuild, e.g. after re-sorting spikes
