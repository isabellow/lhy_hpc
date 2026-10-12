import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import sys
sys.path.append("..//utils/")
sys.path.append("..//stim/")
sys.path.append("..//neural/")
from format_waveform_data import pop_normalize
from cell_filters import filter_cells, apply_cell_filter
from format_behavior_data import (load_behavior_data, get_retrieve_ints,
                                  get_expectation_status, SITE_BAITED, SITE_CACHED)

'''
Average retrieval onset-aligned activity across all LHy cells from all sessions,
split by whether the bird retrieved a baited or a cached seed.

Grouping is from get_expectation_status, as in plot_baited_cached_activity:
    baited   a seed we placed, retrieved on the bird's first interaction with
             the site (it has no way to expect a seed there)
    cached   any seed the bird should expect: its own caches, plus baits it
             already found in an earlier interaction with the site

Each cell's whole-session trace is normalized with pop_normalize (SD units
relative to a running baseline, as in plot_summary_check_retrieve_responses),
averaged over that session's baited and cached retrievals separately, then
averaged across cells (mean +/- SEM across cells).

A session contributes only if it has >= min_retrievals of BOTH groups being compared, so the
two curves are made from the same cells.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

# load the data dictionary
data_dict = np.load(data_file, allow_pickle=True).item()
exclude_birds = []

''' Data params '''
fps = 50 # Hz
dt = 1/fps

# same cell filters and normalization as the check/retrieve summary
CELL_FILTERS = dict(use_stim_filter=False, fr_thresh=0.1, cell_type='all')
pop_norm_kw = dict(std_reg=1e-2, baseline_window=30)

min_retrievals = 3 # per retrieval type, per session

# time-match the two retrieval types, so they sample the same parts of the session
#   None      no matching
#   'window'  keep retrievals with a retrieval of the other type within time_match_window
#   'paired'  keep 1:1 pairs (retrieval of each type) within time_match_window
#   'same'    no matching; instead compare one retrieval type (same_type) against
#             itself, first half vs last half of the session (by onset frame)
time_match = 'same'
time_match_window = 120 # seconds
same_type = 'cached' # 'cached' or 'baited'; only used when time_match = 'same'
if time_match not in (None, 'window', 'paired', 'same'):
    raise ValueError("time_match must be None, 'window', 'paired' or 'same'")

# window to take around retrieval onset
pre_onset = 1  # seconds
post_onset = 1 # seconds
pre_frames = int(pre_onset/dt)
post_frames = int(post_onset/dt)
time_points_seconds = np.arange(-pre_frames, post_frames) * dt
sigma_frames = 1 # Gaussian smoothing of each cell's PSTH, in frames

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9
cached_color = 'xkcd:purple'
baited_color = 'xkcd:dark grey'

# the two curves to plot
if time_match == 'same':
    curve_names = [f'{same_type}, first half', f'{same_type}, last half']
    curve_colors = ['xkcd:lavender', {'cached': cached_color, 'baited': baited_color}[same_type]]
else:
    curve_names = ['cached', 'baited']
    curve_colors = [cached_color, baited_color]

''' Collect retrieval-aligned activity for every cell '''
all_psth = [[], []]  # per curve: list of (n_cells, n_window_frames) arrays, one per session
n_events = [0, 0]    # per curve: total retrievals
n_sessions = 0
for bird in data_dict.keys():
    if bird in exclude_birds:
        continue
    for session_id in data_dict[bird]['all_sessions']:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if not (('behavior' in preprocessed_data) & ('ephys' in preprocessed_data)):
            continue
        print(f'collecting retrieval responses for {bird}_{session_id}')
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"

        ''' Load/format the neural data '''
        spike_fr = np.load(f"{data_dir}aligned_spikes.npy") # cells x video frames
        n_cells_raw, n_frames = spike_fr.shape
        keep_cells, _ = filter_cells(data_dict, bird, session_id, n_cells_raw, **CELL_FILTERS)
        spike_fr = apply_cell_filter(keep_cells, spike_fr=spike_fr)['spike_fr']
        if spike_fr.shape[0] == 0:
            print('  no cells survived the filters, skipping')
            continue

        # normalize each cell's whole-session trace
        norm_fr = pop_normalize(spike_fr, dt=dt, **pop_norm_kw)

        ''' Load behavior data and split retrievals by expectation '''
        seed_struct, count_data = load_behavior_data(data_dir)
        ret_on, _, ret_site = get_retrieve_ints(count_data, seed_struct, return_site_idx=True)
        ret_on = np.asarray(ret_on).astype(int)
        status, _ = get_expectation_status(count_data, seed_struct, ret_on, ret_site)

        # remove retrievals whose window falls outside the session
        in_session = (ret_on - pre_frames >= 0) & (ret_on + post_frames <= n_frames)
        cached_on = ret_on[in_session & (status == SITE_CACHED)]
        baited_on = ret_on[in_session & (status == SITE_BAITED)]

        # time-match the two retrieval types (see time_match above)
        if (time_match in ('window', 'paired')) & (cached_on.shape[0] > 0) & (baited_on.shape[0] > 0):
            dist = np.abs(cached_on[:, None] - baited_on[None, :]) * dt # seconds, cached x baited
            if time_match == 'window':
                cached_on = cached_on[np.min(dist, axis=1) <= time_match_window]
                baited_on = baited_on[np.min(dist, axis=0) <= time_match_window]
            elif time_match == 'paired':
                # 1:1 pairs: as many pairs within the window as possible, then the
                # smallest total gap (out-of-window pairs cost more than any in-window set)
                cost = np.where(dist <= time_match_window, dist, dist.size * time_match_window + 1)
                cached_idx, baited_idx = linear_sum_assignment(cost)
                in_window = dist[cached_idx, baited_idx] <= time_match_window
                cached_on = cached_on[cached_idx[in_window]]
                baited_on = baited_on[baited_idx[in_window]]

        # the two sets of retrievals to compare
        if time_match == 'same':
            same_on = {'cached': cached_on, 'baited': baited_on}[same_type]
            curve_on = [same_on[same_on < n_frames/2], same_on[same_on >= n_frames/2]]
        else:
            curve_on = [cached_on, baited_on]
        print(f'  {curve_names[0]}: {curve_on[0].shape[0]}, {curve_names[1]}: {curve_on[1].shape[0]} retrievals'
              + (f' ({time_match} time-matched)' if time_match in ('window', 'paired') else ''))
        if (curve_on[0].shape[0] < min_retrievals) | (curve_on[1].shape[0] < min_retrievals):
            print('  too few retrievals in one group, skipping')
            continue

        ''' Average onset-aligned activity across retrievals, for each cell '''
        for i in range(2):
            psth = np.mean([norm_fr[:, on-pre_frames:on+post_frames] for on in curve_on[i]], axis=0)
            all_psth[i].append(psth) # save across sessions/birds
            n_events[i] += curve_on[i].shape[0]
        n_sessions += 1

# cells x time, smoothed
all_psth = [gaussian_filter1d(np.vstack(psth), sigma_frames, axis=1) for psth in all_psth]
n_cells = all_psth[0].shape[0]

''' Plot the average/sem retrieval response across cells '''
f, ax = plt.subplots(1, 1, figsize=(4, 4))
for psth, color, name, n in zip(all_psth, curve_colors, curve_names, n_events):
    avg = np.mean(psth, axis=0)
    sem = stats.sem(psth, axis=0)
    ax.plot(time_points_seconds, avg, color=color, lw=1, zorder=1, label=f'{name} ({n} retrievals)')
    ax.fill_between(time_points_seconds, avg+sem, avg-sem, color=color, lw=0, alpha=0.4)

# vertical line at retrieval onset
ax.axvline(0, color='xkcd:gray', linestyle='dashed', lw=0.5)

# formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=tick_label)
ax.set_xlabel('time from retrieval onset (sec.)', size=axis_label)
ax.set_ylabel('normalized FR (SD)', size=axis_label)
ax.set_title(f'all cells (N = {n_cells}, {n_sessions} sessions)', fontsize=title_size)
ax.legend(frameon=False, fontsize=tick_label)

plt.show()
save_name = {None: 'cached_baited', 'window': 'cached_baited_time_match',
             'paired': 'cached_baited_paired', 'same': f'{same_type}_same'}[time_match]
f.savefig(f'{save_figs_dir}aggregate_retrieve_psth_{save_name}.png', dpi=600, bbox_inches='tight')
