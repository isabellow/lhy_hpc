import numpy as np
import pandas as pd

import os 
import sys
sys.path.append("..//utils/")
import color_utils
from format_behavior_data import load_behavior_data, format_feeder_ints
from load_matlab_data import loadmat_sbx
import matplotlib.pyplot as plt

'''
Plot the firing rates of all cells over time in the session
- chunk by excitatory vs. inhibitory
- consider different sorting methods
	- by fr during feeder interaction?
	- unsupervised ala Stringer?

Also plot:
- foot speed (cm/s)
- caches, retrievals?
- eating bouts
- feeder open/close
- feeder interactions
- stim on/off

Ideally make it zoomable and scrollable
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Session params (update as needed) '''
bird = 'AMB154'
session_id = '241122'
pred_date = '260409'

''' Define the save folder'''
save_dir = f"{save_figs_dir}/{bird}/"
if os.path.isdir(save_dir):
    print('save directory exists')
else:
    os.mkdir(save_dir)
save_folder = f"{save_dir}/{bird}_{session_id}/"
if os.path.isdir(save_folder):
    print('save folder exists')
else:
    os.mkdir(save_folder)

''' Load the frame times '''
framet_raw = np.load(f'{data_dir}frame_times.npy')
framet_raw = np.squeeze(framet_raw)
dt = np.unique(np.round(np.diff(framet_raw), 4))

''' Load and format behavior data '''
data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
seed_struct, count_data = load_behavior_data(data_dir)

# get cache/retrieval offsets
all_int_end = count_data['endSite']
all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
cache_offsets = all_int_end[all_int_changes > 0]
retrieval_offsets = all_int_end[all_int_changes < 0]

# eating bouts
eat_onsets = count_data['newBeakPerch']
eat_offsets = count_data['endBeakPerch']

# get all feeder interactions
feeder_int_start, feeder_int_end, feeder_idx = get_feeder_ints(count_data, use_beak=True)

# get the foot speed (cm/sec)
posture_vel_file = 'posture_vel_smooth.npy'
smooth_posture_vel = np.load(f'{data_dir}{posture_vel_file}')
foot_speed = np.sqrt(np.sum(np.mean(smooth_posture_vel[:, [10, 14]], axis=1)**2, axis=1))
foot_speed_cm = foot_speed*13*2.54

''' Get feeder open/close times '''
session_info = pd.read_excel(session_info_file, sheet_name=bird, header=1)
session_info["id"] = session_info["date"].dt.strftime("%y%m%d")
# todo
probe_depth = session_info.loc[session_info["id"] == session_id,
                                               "approx. depth (um)"].iloc[0]

''' Load and format neural data '''
# smooth and zscore firing rates
smooth_spike_fr = gaussian_filter1d(spike_fr, sigma=250, axis=1)
zscore_spike_fr = stats.zscore(smooth_spike_fr, axis=1)

'''
Params
------
zscore_spike_fr
--> could also try black/white 0 to max?
foot_speed_cm

timepoints:
- stim_onset, stim_offset
- feeder_open, feeder_close
- feeder_int_start, feeder_int_end (todo)
--> feet contacting/leaving feeder perch
- eat_onsets, eat_offsets
- cache_offsets
- retrieve_offsets (todo)

for clustering/sorting:
- clu1_idx, clu2_idx
- ex_ch_sort, in_ch_sort
'''
# todo compute height ratio for inhib vs. excite cells 

f, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                     gridspec_kw=dict(height_ratios=[45, 10, 6, 6], hspace=0.1))
n_cache = cache_offsets.shape[0]

# plot firing rates separately for excitatory and inhibitory
ax[0].imshow(zscore_spike_fr[clu1_idx][ex_ch_sort],
              clim=[-4, 4],
              aspect='auto', cmap='bwr',
              interpolation='none')
ax[1].imshow(zscore_spike_fr[clu2_idx][in_ch_sort][1:],
              clim=[-4, 4],
              aspect='auto', cmap='bwr',
              interpolation='none')
ax[0].set_ylim(ax[0].get_ylim()[::-1])
ax[1].set_ylim(ax[1].get_ylim()[::-1])

# plot foot speed
ax[2].plot(np.arange(foot_speed_cm.shape[0]), foot_speed_cm, lw=0.1, c='k')

# plot caching, eating, feeder periods, stim periods
ylim_1 = np.asarray([0, 1])
for s_start, s_end in zip(stim_onset*50, stim_offset*50): # seconds
    ax[3].fill_betweenx(ylim_1, np.full(2, s_start), np.full(2, s_end), 
                        color='xkcd:scarlet', lw=0, zorder=0)
for fo, fc in zip(feeder_open*60*50, feeder_close*60*50): # minutes
    ax[3].fill_betweenx(ylim_1, np.full(2, fo), np.full(2, fc), 
                        color='k', alpha=0.2, lw=0, zorder=0)
ylim_2 = np.asarray([1, 2])
for e_start, e_end in zip(eat_onsets, eat_offsets): # frames
    ax[3].fill_betweenx(ylim_2, np.full(2, e_start), np.full(2, e_end), 
                        color='xkcd:cobalt blue', alpha=1, lw=0, zorder=0)
ax[3].scatter(cache_offsets, np.full(n_cache, 2.6), 
              c='xkcd:orange', marker='|', lw=0.5, s=200)

# ticks, labels, etc.
ax[2].set_ylim([0, 180])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['left'].set_bounds(0, 150)
ax[2].set_ylim([0, 180])
ax[2].set_yticks(np.arange(0, 180, 50))

ax[3].set_ylim([0, 3])
ax[3].spines['top'].set_visible(False)
ax[3].spines['left'].set_visible(False)
ax[3].spines['right'].set_visible(False)
ax[3].set_yticks([])

n_mins = np.floor(n_frames/50/60/30)*30
ax[3].set_xticks(np.arange(0, n_mins*50*60 + 1, 30*50*60))
ax[3].set_xticklabels(np.arange(0, n_mins + 1, 30).astype(int))
ax[3].set_xlim([0, n_frames])

ax[0].set_ylabel('excitatory cells')
ax[1].set_ylabel('inhib. cells')
ax[2].set_ylabel('foot speed\n(cm/s)')
ax[3].set_xlabel('time (minutes)')

# TODO add labels for caches, retrievals, etc.

plt.show()
f.savefig(f'{save_folder}/ethogram_full.png', 
                      dpi=600, bbox_inches='tight')