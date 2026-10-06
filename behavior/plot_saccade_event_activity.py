import numpy as np

import os
import sys
import zlib
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from cell_filters import filter_cells, apply_cell_filter
from event_psth import (window_frames, build_raster, raster_scatter,
                        sort_events_by_duration, sort_events_by_time_group,
                        subsample_for_raster, event_psth_on_off, shared_ylim,
                        raster_marker_size, plot_psth_trace, draw_group_labels)
from format_behavior_data import (load_behavior_data, get_event_times,
                                  get_saccades, get_preceding_saccades,
                                  SITE_EMPTY, SITE_BAITED, SITE_CACHED)
from spike_amplitudes import (load_spike_amplitudes, trial_amplitudes,
                              select_trials_by_drift, plot_amp_panel)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

'''
Single-cell activity around the saccade preceding each check / retrieval /
cache, and around the event itself.

The preceding saccade is the most recent head saccade before event onset made
from a different perch than the one the bird is on at onset
(format_behavior_data.get_preceding_saccades).  Events are dropped if there is
no such saccade within max_saccade_lag, or if a site interaction, feeder
interaction or eating bout happened between the saccade and the onset.

Checks and retrievals are split by what the bird should expect to find
(get_expectation_status), as in plot_baited_cached_activity.py.  Caches are
one group.  Saccade-aligned activity uses the same colors as its event.

Each figure is one cell, two blocks:
    top     saccade-aligned
    bottom  event-aligned
each with beak height (z) and angular head speed above the raster (single
events in light gray, mean in black, same events as the raster), then
raster | spike amplitude | tuning curve.
Black ticks in each raster mark the other alignment point, so the
saccade-to-event lag is visible in both blocks.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = "../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Data params '''
bird = 'TRQ82'  # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50  # Hz
dt = 1 / fps

''' Cell filtering — every criterion is opt-in '''
CELL_FILTERS = dict(
    use_stim_filter = False,      # True = projection-nucleus cells only
    fr_thresh       = 0.1,        # Hz; None disables the firing-rate cut
    cell_type       = 'all',      # 'all' | 'excitatory' | 'inhibitory'
)

''' Event params '''
event_key = 'check'         # 'check' | 'retrieval' | 'cache'
align_event = 'onset'       # bottom row aligned to event 'onset' | 'offset'
max_check_dur = 1.5         # s; longer no-change interactions are not checks
discovered_bait = 'cached'  # see get_expectation_status
removal_rule = 'cached_first'
use_init_counts = True

''' Saccade params '''
saccade_params = None       # None = SACCADE_PARAMS defaults; dict to override
max_saccade_lag = np.inf    # s; max saccade-to-onset time (np.inf = no limit)
different_perch = True      # saccade must be made from a different perch

min_events_per_group = 2    # fewer usable events: no tuning curve
max_events_per_group = 200  # max raster rows per group
sort_by_lag = True          # raster rows by saccade-to-event lag, else chronological

''' Per-cell trial selection (unit drift) '''
subsample_drift = True
subsample_metric = 'amplitude'   # 'firing rate' | 'amplitude'
subsample_thresh = 0.7           # keep events >= this x the session average
thresh_t_window = 60.0           # s centered on the event

''' Amplitude panel params '''
show_amp_panel = True
amp_min_spikes = 1          # events with fewer spikes in window are left blank
amp_panel_lw = 0.8

''' Plotting params '''
sacc_window = 1.0           # s total, centered on the saccade peak
event_window = 1.0          # s total, centered on the event onset/offset
sigma_frames = 1            # tuning-curve smoothing (frames), 0 = none

# keyed by status code; colors as in plot_beak_flap_activity.py
# (no baited check there; dark grey as in plot_baited_cached_activity.py)
group_colors = {
    'check': {SITE_EMPTY: 'xkcd:apple green', SITE_BAITED: 'xkcd:dark grey',
              SITE_CACHED: 'xkcd:deep green'},
    'retrieval': {SITE_BAITED: 'xkcd:lavender', SITE_CACHED: 'xkcd:purple'},
    'cache': {0: 'xkcd:orange'},
}[event_key]
plot_names = ({0: 'cache'} if event_key == 'cache' else
              {SITE_EMPTY: 'empty', SITE_BAITED: 'bait', SITE_CACHED: 'cache'})

psth_lw = 1.5
event_lw = 0.8
divider_lw = 0.8
title_size = 14
axis_label = 12

# behavior traces above each raster
trace_color = 'xkcd:light gray'   # single events
trace_lw = 0.4
mean_lw = 1.5                     # event average, black
trace_label_size = 9

# colored group labels beside the raster, and the plain y label outside them
grp_label_size = axis_label
grp_label_gap = 0
ylabel_pad = grp_label_size + 8

def nice_max(x):
    '''Round x up to one significant figure, for a clean top tick.'''
    if not np.isfinite(x) or x <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(x))
    return float(np.round(np.ceil(x / mag) * mag, 12))


# one entry per figure row: tuning-curve window (frames), its time axis, and
# the raster half-width / time axis covering the same span
ROWS = []
for win, xlabel in [(sacc_window, 'time from saccade peak (s)'),
                    (event_window, f'time from {event_key} {align_event} (s)')]:
    fr_start, fr_end, tc_t = window_frames(-win / 2, win / 2, dt)
    ROWS.append(dict(win=(fr_start, fr_end), tc_t=tc_t, hw=fr_end,
                     raster_t=np.arange(fr_start, fr_end + 1) * dt, xlabel=xlabel))

# sessions with pose tracking & ephys
behavior_sessions = [s for s in session_list
                     if {'behavior', 'ephys'} <= set(data_dict[bird][s]['preprocessed_data'])]

save_folder = f"{save_figs_dir}{bird}/saccade_{event_key}_activity/"
os.makedirs(save_folder, exist_ok=True)

for session_id in behavior_sessions:
    print(f'plotting saccade/{event_key} responses for {bird}_{session_id}')
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}behavior_data/"

    ''' Neural data + cell filter '''
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy")  # cells x video frames
    n_cells_raw, n_frames = spike_fr.shape
    avg_firing_rate = 10 ** data_dict[bird][session_id]['waveform_props'][2]
    all_ids = np.load(f"{data_dir}aligned_spikes_ids.npy")

    keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw,
                                 **CELL_FILTERS)
    filtered = apply_cell_filter(keep_cells, spike_fr=spike_fr,
                                 avg_firing_rate=avg_firing_rate)
    spike_fr = filtered['spike_fr']
    avg_fr = np.round(filtered['avg_firing_rate'], 2)
    cell_ids = all_ids[keep_cells]
    if spike_fr.shape[0] == 0:
        print('  no cells survived the filters, skipping')
        continue

    ''' Events and their preceding saccades '''
    seed_struct, count_data = load_behavior_data(data_dir)
    onsets, offsets, _, status = get_event_times(
        count_data, seed_struct, event_key, max_check_dur=max_check_dur, dt=dt,
        use_init_counts=use_init_counts, removal_rule=removal_rule,
        discovered_bait=discovered_bait)
    saccade_time, saccade_perch_idx, kin = get_saccades(
        data_dir, count_data, fps=fps, params=saccade_params, return_kinematics=True)
    # per-frame traces drawn above the rasters: (label, trace)
    behavior_traces = [('beak z', kin['beak_pos'][:, 2]),
                       ('head °/s', kin['ang_speed'])]
    prev, blocked = get_preceding_saccades(onsets, saccade_time, saccade_perch_idx,
                                           count_data, max_lag=max_saccade_lag,
                                           fps=fps, different_perch=different_perch)

    groups = status if event_key != 'cache' else np.zeros_like(status)
    keep = (prev >= 0) & np.isin(groups, list(group_colors))
    sacc = saccade_time[prev[keep]]
    align = (onsets if align_event == 'onset' else offsets)[keep]
    groups = groups[keep]
    tally = ', '.join(f'{int(np.sum(groups == g))} {plot_names[g]}'
                      for g in group_colors)
    print(f'  {saccade_time.shape[0]} saccades; of {onsets.shape[0]} {event_key}s, '
          f'{int(np.sum((prev < 0) & ~blocked))} have no preceding saccade and '
          f'{int(blocked.sum())} have another event in between; using {tally}')
    if align.shape[0] == 0:
        continue

    ''' KS spike amplitudes, for the side panel and the drift cut '''
    amp_data = None
    if show_amp_panel or (subsample_drift and subsample_metric == 'amplitude'):
        sd = data_dict[bird][session_id]
        try:
            amp_data = load_spike_amplitudes(
                session_dir, data_dir, f"{bird}_{sd['ephys_id']}/{sd['ks_folder']}/",
                cell_ids, n_frames, fps=fps)
        except FileNotFoundError as err:
            print(f'  no KS amplitudes ({err.filename}): no amp panel / amplitude drift cut')

    ''' Per-cell event selection (unit drift) '''
    trial_keep = None
    if subsample_drift and (subsample_metric != 'amplitude' or amp_data is not None):
        trial_keep = select_trials_by_drift(spike_fr, align, dt, subsample_metric,
                                            subsample_thresh, thresh_t_window,
                                            amp_data=amp_data, min_spikes=amp_min_spikes)

    ''' Tuning curves: saccade- and event-aligned, over the same events '''
    sacc_psth, event_psth, group_ids, n_used = event_psth_on_off(
        spike_fr, sacc, align, ROWS[0]['win'], ROWS[1]['win'], dt,
        groups=groups, sigma_frames=sigma_frames, keep=trial_keep)
    psths = [sacc_psth, event_psth]
    enough = n_used >= min_events_per_group

    ''' Plot '''
    # per block: beak / head traces above the raster (height 1:1:4), then
    # raster | amplitude | spacer | tuning curve
    raster_seed = zlib.crc32(f'{bird}_{session_id}_{event_key}'.encode()) & 0xffffffff
    f = plt.figure(figsize=(9, 9))
    outer = f.add_gridspec(2, 1, hspace=0.32)
    ax = []
    for row in range(2):
        gs = outer[row].subgridspec(3, 4, height_ratios=[1, 1, 4], hspace=0.15,
                                    width_ratios=[1, 0.28, 0.22, 1], wspace=0.2)
        raster_ax = f.add_subplot(gs[2, 0])
        ax.append(dict(traces=[f.add_subplot(gs[i, 0], sharex=raster_ax) for i in range(2)],
                       raster=raster_ax, amp=f.add_subplot(gs[2, 1]),
                       tc=f.add_subplot(gs[2, 3])))

    for c_idx, cell_id in enumerate(cell_ids):
        for d in ax:
            for a in d['traces'] + [d['raster'], d['amp'], d['tc']]:
                a.cla()
                a.set_visible(True)

        # this cell's events, capped per group, sorted into group blocks
        sel = slice(None) if trial_keep is None else trial_keep[c_idx]
        r_sacc, r_align, r_groups, _, _ = subsample_for_raster(
            sacc[sel], align[sel], groups[sel], max_per_group=max_events_per_group,
            rng=np.random.default_rng(raster_seed))
        n_events = r_sacc.shape[0]
        if n_events == 0:
            print(f'  cell {cell_id}: no events survive the drift cut, skipping')
            continue
        sort_fn = sort_events_by_duration if sort_by_lag else sort_events_by_time_group
        order, block_edges = sort_fn(r_sacc, r_align, r_groups)   # duration = lag
        r_groups = r_groups[order]
        lag = (r_align - r_sacc)[order] * dt
        aligned = [r_sacc[order], r_align[order]]

        # per-event amplitudes in each row's raster window, one x limit for both
        raster_amp = [None, None]
        if amp_data is not None:
            raster_amp = [trial_amplitudes(amp_data[c_idx][0], amp_data[c_idx][1],
                                           frames, cfg['hw'], min_spikes=amp_min_spikes)[0]
                          for frames, cfg in zip(aligned, ROWS)]
            finite = np.concatenate(raster_amp)
            finite = finite[np.isfinite(finite)]
            amp_xmax = max(np.ceil(float(finite.max()) * 10) / 10, 0.5) if finite.size else None

        max_fr = shared_ylim(*[p[c_idx, enough[c_idx]] for p in psths],
                             minimum=float(avg_fr[c_idx]) + 1)

        # behavior around each raster event (nan past the session edges), and
        # one y range per trace shared by both blocks
        segs = [[build_raster(np.pad(trace, cfg['hw'], constant_values=np.nan),
                              frames + cfg['hw'], cfg['hw']) for _, trace in behavior_traces]
                for frames, cfg in zip(aligned, ROWS)]
        trace_lims = [(min(0.0, np.nanmin([np.nanmin(s[i]) for s in segs])),
                       nice_max(np.nanmax([np.nanmax(s[i]) for s in segs])))
                      for i in range(len(behavior_traces))]
        rng = np.random.default_rng(int(cell_id) * 7919)

        for row, (cfg, frames, ticks) in enumerate(zip(ROWS, aligned, [lag, -lag])):
            # ── Beak height and angular head speed above the raster ───
            for a, (label, _), seg, (y0, y1) in zip(ax[row]['traces'], behavior_traces,
                                                     segs[row], trace_lims):
                a.plot(cfg['raster_t'], seg.T, color=trace_color, lw=trace_lw)
                with np.errstate(all='ignore'):
                    a.plot(cfg['raster_t'], np.nanmean(seg, axis=0), color='k', lw=mean_lw)
                a.axvline(0, color='k', ls='--', lw=event_lw)
                a.set_ylim(y0, y1)
                a.set_yticks([0, y1])
                a.set_yticklabels(['0', f'{y1:g}'])
                a.set_ylabel(label, fontsize=trace_label_size, rotation=0,
                             ha='right', va='center')
                a.tick_params(labelbottom=False, labelsize=trace_label_size)
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            # ── Raster, colored by group ──────────────────────────────
            a = ax[row]['raster']
            t = cfg['raster_t']
            a.set_xlim(t[0], t[-1])
            a.set_ylim(-0.5, n_events - 0.5)
            a.yaxis.set_major_locator(MaxNLocator(integer=True))
            spk_s = raster_marker_size(a, f, n_events)
            spk_t, spk_row = raster_scatter(build_raster(spike_fr[c_idx], frames, cfg['hw']),
                                            t, dt, rng=rng)
            spk_groups = r_groups[spk_row.astype(int)]
            for g in np.unique(r_groups):
                a.scatter(spk_t[spk_groups == g], spk_row[spk_groups == g],
                          color=group_colors[g], marker='|', lw=0.6, s=spk_s)
            a.axvline(0, color='k', ls='--', lw=event_lw)
            in_win = np.abs(ticks) <= t[-1]
            a.scatter(ticks[in_win], np.flatnonzero(in_win), color='k', marker='|',
                      lw=event_lw, s=spk_s / 2, zorder=2)
            for edge in block_edges:
                a.axhline(edge - 0.5, color='xkcd:gray', lw=divider_lw, zorder=3)
            a.set_xlabel(cfg['xlabel'], fontsize=axis_label)

            # ── Amplitude panel: same rows, colors and dividers ───────
            if raster_amp[row] is not None:
                plot_amp_panel(ax[row]['amp'], raster_amp[row], groups_sorted=r_groups,
                               block_edges=block_edges, colors=group_colors,
                               divider_lw=divider_lw, lw=amp_panel_lw,
                               axis_label=axis_label, xmax=amp_xmax)
                ax[row]['amp'].set_ylim(a.get_ylim())
            else:
                ax[row]['amp'].set_visible(False)

            # ── Tuning curve, one trace per group ─────────────────────
            a = ax[row]['tc']
            for g_idx, g in enumerate(group_ids):
                if enough[c_idx, g_idx]:
                    plot_psth_trace(a, cfg['tc_t'], psths[row][c_idx, g_idx], dt,
                                    group_colors[int(g)], sigma_frames=sigma_frames,
                                    lw=psth_lw)
            a.axvline(0, color='k', ls='--', lw=event_lw)
            a.axhline(avg_fr[c_idx], color='xkcd:gray', ls='--', lw=event_lw)
            a.set_xlim(cfg['tc_t'][0], cfg['tc_t'][-1] + dt)
            a.set_ylim(0, max_fr)
            a.set_yticks([0, max_fr])
            a.set_xlabel(cfg['xlabel'], fontsize=axis_label)
            a.set_ylabel('firing rate (Hz)', fontsize=axis_label)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.grid(True)

        # ── Colored group labels: tick labels have to be final first ──
        ax[0]['raster'].set_ylabel(f'preceding saccade',
                         fontsize=axis_label, labelpad=ylabel_pad)
        ax[1]['raster'].set_ylabel(f'{event_key}s (n={np.asarray(sacc[sel]).shape[0]})',
                         fontsize=axis_label, labelpad=ylabel_pad)
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        for row in range(2):
            draw_group_labels(f, ax[row]['raster'], renderer, r_groups, block_edges,
                              group_colors, plot_names, fontsize=grp_label_size,
                              gap=grp_label_gap)

        f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
                   f'(baseline {avg_fr[c_idx]} Hz)', fontsize=title_size, y=0.93)
        f.savefig(f'{save_folder}{session_id}_saccade_{event_key}_{align_event}_cell{cell_id}.png',
                  dpi=400, bbox_inches='tight')

    plt.close(f)
