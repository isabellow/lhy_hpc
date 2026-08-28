import numpy as np

import os
import sys
sys.path.append("..//utils/")
sys.path.append("..//neural/")
from format_waveform_data import cluster_ids_for_session
from format_behavior_data import (load_behavior_data, get_checks_raw,
                                  get_site_occupancy)
from helpers import moving_avg

import matplotlib.pyplot as plt

'''
Pre-onset firing rate vs. time in session, one point per check, split by
whether the site was occupied (any seeds) vs. empty.

Sanity check on plot_check_visit_activity.py: if occupied checks are
concentrated later in the session and the cell's local baseline drifts (state
change, probe drift), an apparent occupancy difference in the onset PSTH could
be a baseline effect. This plot shows the occupied/empty distribution over the
session alongside the running baseline (dashed grey).

Events come from get_checks_raw, matching plot_check_visit_activity.py.
'''

''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Data params '''
bird = 'LMN88'          # update as needed
session_id = '260720'         # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
fps = 50  # Hz behavioral video frames
dt = 1 / fps

# example cells, by phy cluster ID (as in the check/visit plot title)
cell_ids_to_plot = np.array([44, 49, 56, 61, 66, 90, 91, 97, 107, 108, 116, 122, 125, 126, 131, 132, 144, 147, 148, 152])

''' Event params '''
# site interactions longer than this are probably not checks (SC, EM 2024)
max_check_dur = 1.5     # seconds

# window for the per-check firing rate, relative to check onset
titles = ["pre-onset", "post-onset"]
window_1 = (-0.2, 0.0)     # seconds
window_2 = (0.1, 0.3)     # seconds

# window for the running baseline FR
baseline_window = 5     # minutes

# bin width for the per-group mean, and the events a bin needs to be drawn
mean_bin_width = 10     # minutes
min_checks_per_bin = 5

''' Plotting params '''
# style — checks green, matching plot_check_visit_activity.py
# group index is the occupancy flag itself: 0 = empty, 1 = occupied
check_colors = ['xkcd:apple green', 'xkcd:deep green']
group_names = ['empty', 'occupied']
marker_size = 5
marker_alpha = 0.2
baseline_lw = 1
mean_lw = 1
title_size = 14
axis_label = 12
legend_size = 8

''' Define/create the save folder '''
save_folder = f"{save_figs_dir}/{bird}/avg_check_fr/"
os.makedirs(save_folder, exist_ok=True)

''' Load and format the neural data '''
session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
data_dir = f"{session_dir}/behavior_data/"

# spike counts per 20 ms video frame, cells x video frames
spike_counts = np.load(f"{data_dir}aligned_spikes.npy")
n_cells_raw, n_frames = spike_counts.shape

# session average firing rate (waveform_props is [asymm, width, log10 fr])
avg_firing_rate = 10 ** data_dict[bird][session_id]['waveform_props'][2]

''' Label cells by their KS cluster ID '''
all_ids = cluster_ids_for_session(data_dict, bird, session_id, root_dir)

''' Load and format behavior data '''
seed_struct, count_data = load_behavior_data(data_dir)

check_onsets, check_offsets, check_site_idx = get_checks_raw(count_data, seed_struct, max_check_dur=max_check_dur, dt=dt)
check_occupied = get_site_occupancy(count_data, seed_struct, check_onsets, check_site_idx, use_init_counts=True)
groups = check_occupied.astype(int)     # 0 empty, 1 occupied
init_count = np.sum(np.asarray(seed_struct['initSeedCounts'], dtype=int))
print(f'{check_onsets.shape[0]} checks '
      f'({int(np.sum(groups == 1))} occupied, {int(np.sum(groups == 0))} empty)')

''' Frames in each window, one row per check '''
window_index, window_usable, window_quantum = [], [], []
for fr_window, title in zip([window_1, window_2], titles):
    win_frames = np.arange(int(round(fr_window[0] / dt)),
                           int(round(fr_window[1] / dt)))
    win_idx = check_onsets[:, None] + win_frames[None, :]

    # drop checks whose window runs off either end of the session
    usable = np.all((win_idx >= 0) & (win_idx < n_frames), axis=1)
    win_idx = np.clip(win_idx, 0, n_frames - 1)
    if np.sum(~usable):
        print(f'  dropping {int(np.sum(~usable))} checks with a truncated '
              f'{title} window')

    window_index.append(win_idx)
    window_usable.append(usable)

    # a window of n frames can only return multiples of this many Hz
    window_quantum.append(1 / (win_frames.shape[0] * dt))
    print(f'  {title} window is {win_frames.shape[0]} frames, '
          f'{window_quantum[-1]:.1f} Hz resolution')


''' Time axes, in minutes from session start '''
check_t = check_onsets * dt / 60
session_t = np.arange(n_frames) * dt / 60

''' Bins for the per-group mean '''
bin_edges = np.arange(0, session_t[-1] + mean_bin_width, mean_bin_width)
bin_centers = bin_edges[:-1] + mean_bin_width / 2
check_bin = np.digitize(check_t, bin_edges) - 1


def binned_mean(values, sel, n_bins):
    '''
    Mean of `values` over the selected checks in each time bin.  Bins with
    fewer than min_checks_per_bin events are left nan, so the line breaks
    rather than drawing a mean over one or two events.
    '''
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = sel & (check_bin == b)
        if np.sum(in_bin) >= min_checks_per_bin:
            means[b] = np.mean(values[in_bin])
    return means

''' Plot pre/post-onset firing rate over the session for each example cell '''
f, axes = plt.subplots(2, 1, figsize=(6, 3.5), sharey=False,
                        gridspec_kw=dict(hspace=0.45))

for cell_id in cell_ids_to_plot:
    match = np.flatnonzero(all_ids == cell_id)
    if match.shape[0] == 0:
        print(f'  cell {cell_id} not found in this session - confirm that cell_ids_to_plot matches session_id')
        continue
    c_idx = match[0]

    # deterministic jitter, so a cell looks the same each time it is drawn
    rng = np.random.default_rng(int(cell_id) * 7919)

    # running baseline over the whole session
    running_fr = moving_avg(spike_counts[c_idx], fs=fps,
                            window=baseline_window) / dt

    for row in range(2):
        axes[row].cla()

    for row in range(2):
        ax = axes[row]
        win_idx = window_index[row]
        usable = window_usable[row]

        # mean firing rate in the window, per check
        check_fr = np.mean(spike_counts[c_idx][win_idx], axis=1) / dt

        # jitter spikes to random possible fr given the measurement window
        check_fr_jit = check_fr + rng.uniform(0.0, window_quantum[row],
                                              size=check_fr.shape[0])

        # plot occupied/empty checks
        for g_id in [0, 1]:
            sel = (groups == g_id) & usable
            ax.scatter(check_t[sel], check_fr_jit[sel], s=marker_size,
                       alpha=marker_alpha, color=check_colors[g_id],
                       edgecolors='none')

        # per-group mean in mean_bin_width bins, from the unjittered rates
        for g_id in [0, 1]:
            sel = (groups == g_id) & usable
            ax.plot(bin_centers, binned_mean(check_fr, sel, bin_centers.shape[0]),
                    color=check_colors[g_id], lw=mean_lw, zorder=4, label=f'{group_names[g_id]}')

        ax.plot(session_t, running_fr, color='xkcd:gray', linestyle='dashed',
                lw=baseline_lw, zorder=3,
                label=f'avg FR')

        ax.set_xlim(0, session_t[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(titles[row], fontsize=title_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row==0:
            ax.legend(fontsize=legend_size, frameon=False)
        else:
            ax.set_xlabel('time in session (min)', fontsize=axis_label)


    f.suptitle(f'{bird} {session_id}  —  cell {cell_id}  '
               f'({np.round(avg_firing_rate[c_idx], 2)} Hz)',
               fontsize=title_size, y=1)
    f.supylabel(f'firing rate (Hz)', fontsize=axis_label)

    f.savefig(f'{save_folder}/{session_id}_check_fr_cell{cell_id}.png',
              dpi=400, bbox_inches='tight')

plt.close(f)
