import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import os 
import sys
from scipy.io import loadmat

from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel
sys.path.append("..//behavior/")
from format_behavior_data import load_behavior_data, get_cache_ints, spikes_by_cache
sys.path.append("..//utils/")
import helpers
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

'''
Compute the offset-aligned psth for all excitatory cells across all sessions/birds
as in DA's talk

Grab a window of activity around cache offset and compute the psth for each cell
Take the log (w/ added 0.1 Hz to regularize)
Average across excitatory units
'''

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# load the data dictionary and get bird ids
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = []
for bird in data_dict.keys():
    bird_ids.append(bird)

# window to take around cache offset
pre_offset = 9 # seconds
post_offset = 5 # seconds

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9

''' To plot example rasters '''
ex_sessions = ['RBY94_241129', 'AMB154_241202', 'SLV132_250310', 'IND67_251003', 'LMN146_251114']

cache_window = 20 # seconds
cache_color = 'xkcd:orange'
spk_s = 5
psth_lw = 2
on_lw = 1
off_lw = 1
on_s = 1
time_int = 5
f = plt.figure(figsize=(10, 10))
h_ratio = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
h_ratio = np.ones(10)
gs = f.add_gridspec(10, 5, height_ratios=h_ratio, hspace=0.1,
                    width_ratios=[1, 1, 1, 1, 1], wspace=0.3)
row_idx = 0

''' List of behavior sessions '''
all_behavior_sessions = []
for i, bird in enumerate(bird_ids):
    behavior_sessions = []
    for session_id in data_dict[bird]['all_sessions']:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)
    all_behavior_sessions.append(behavior_sessions)

all_cache_psth = np.asarray([])
all_excitatory_idx = np.asarray([]).astype(bool)
all_modulation_idx = np.asarray([])
for bird in bird_ids:
    if bird=='LIM63':
        continue
    bird_idx = bird_ids.index(bird)

    ''' Define/create the save folder'''
    save_dir = f"{save_figs_dir}/{bird}/"
    if os.path.isdir(save_dir):
        print('save directory exists')
    else:
        os.mkdir(save_dir)
    save_folder = f"{save_dir}/barcodes/"
    if os.path.isdir(save_folder):
        print('save folder exists')
    else:
        os.mkdir(save_folder)

    # to collect data by bird
    bird_cache_psth = np.asarray([])
    bird_excitatory_idx = np.asarray([]).astype(bool)
    bird_modulation_idx = np.asarray([])

    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions:
        print(f'collecting average cache responses for {bird}_{session_id}')

        ''' Get the file params '''
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        data_dir = f"{session_dir}/behavior_data/"
        pred_date = data_dict[bird][session_id]['pred_date']
        ephys_id = data_dict[bird][session_id]['ephys_id']
        ephys_folder= f"{root_dir}{bird}/{bird}_{session_id}/{bird}_{ephys_id}/"
        for folder in sorted(os.listdir(ephys_folder)):
            if 'kilosort4' in folder:
                for file in sorted(os.listdir(f"{ephys_folder}{folder}")):
                    if 'waveformStruct' in file:
                        ks_dir = f"{bird}_{ephys_id}/{folder}/"

        ''' Load the frame times '''
        sampling_rate = 30000 # intan
        framet_raw = np.load(f'{data_dir}frame_times.npy')
        framet_raw = np.squeeze(framet_raw)
        dt = np.unique(np.round(np.diff(framet_raw), 2))
        dt = dt[0]

        # align so that 0 is the video start time
        start_t = framet_raw[0]
        frame_t = framet_raw - start_t
        frame_samples = np.append(frame_t, frame_t[-1] + dt)*sampling_rate
        n_frames = frame_t.shape[0]

        ''' Load/format the neural data '''
        # get the cell IDs and raw spike times
        good_clusters, spike_id, spike_samp_raw = get_spike_times(session_dir, ks_dir=ks_dir)
        n_cells = good_clusters.shape[0]
        
        # keep only spikes from within the session
        spike_t = spike_samp_raw - start_t*sampling_rate
        spike_id = spike_id[(spike_t >= 0) & (spike_t <= frame_samples[-1])]
        spike_t = spike_t[(spike_t >= 0) & (spike_t <= frame_samples[-1])]

        # spikes per frame and spike bool
        spike_frame = np.zeros((n_cells, n_frames))
        i = -1
        for c_idx, cell in enumerate(good_clusters):
            i += 1
            spk_times = spike_t[spike_id==cell]       
            spike_frame[i], _ = np.histogram(spk_times, frame_samples)

        # keep only cells in nucleus
        stim_idx = idx_cells_by_stim(data_dict, bird, session_id)
        spike_frame = spike_frame[stim_idx]
        n_cells = spike_frame.shape[0]

        # for plotting rasters
        spike_bool = spike_frame.astype(bool)

        # instantaneous firing rate
        inst_firing_rate = spike_frame/dt

        # index for excitatory cells
        excitatory_idx = data_dict[bird][session_id]['excitatory_idx'][stim_idx]

        # cache modulation index
        modulation_idx = data_dict[bird][session_id]['barcode_dict']['cache_modulated'][stim_idx]
        
        ''' Load and format behavior data '''
        # load behavioral data
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
        seed_struct, count_data = load_behavior_data(data_dir)

        # get cache offset times
        cache_onsets, cache_offsets = get_cache_ints(count_data, seed_struct)
        n_caches = cache_offsets.shape[0]
        if n_caches == 0:
            continue

        ''' Average cache offset-aligned activity '''
        pre_frames = int(pre_offset/dt) # frames
        post_frames = int(post_offset/dt) # frames
        n_cache_frames = pre_frames + post_frames

        # get offset-aligned activity for each cell
        cache_activity = np.full((n_caches, n_cells, n_cache_frames), np.nan)
        for i, cache_off in enumerate(cache_offsets):
            start_frame = cache_off - pre_frames
            end_frame = cache_off + post_frames
            if (start_frame > 0) & (end_frame < n_frames):
                cache_activity[i] = inst_firing_rate[:, start_frame:end_frame]

        # remove caches that start or end outside session time
        nan_idx = np.isnan(np.sum(cache_activity, axis=(1, 2)))
        cache_activity = cache_activity[~nan_idx]

        # average across caches for each cell
        avg_cache_activity = np.mean(cache_activity, axis=0)

        # save across birds
        if all_cache_psth.shape[0] == 0:
            all_cache_psth = avg_cache_activity
        else:
            all_cache_psth = np.row_stack((all_cache_psth, avg_cache_activity))
        all_excitatory_idx = np.append(all_excitatory_idx, excitatory_idx)
        all_modulation_idx = np.append(all_modulation_idx, modulation_idx)

        # save across sessions (within bird)
        if bird_cache_psth.shape[0] == 0:
            bird_cache_psth = avg_cache_activity
        else:
            bird_cache_psth = np.row_stack((bird_cache_psth, avg_cache_activity))
        bird_excitatory_idx = np.append(bird_excitatory_idx, excitatory_idx)
        bird_modulation_idx = np.append(bird_modulation_idx, modulation_idx)

    # ''' Plot suppression by bird '''
    # # regularize and log
    # log_cache_psth = np.log10(bird_cache_psth + 0.1)

    # # average across all excitatory cells
    # avg_cache_resp_all = np.mean(log_cache_psth[bird_excitatory_idx], axis=0)
    # sem_cache_resp_all = stats.sem(log_cache_psth[bird_excitatory_idx], axis=0)

    # # fig params
    # f, ax = plt.subplots(1, 1, figsize=(4, 4))
    # time_points_seconds = np.linspace(-pre_offset, post_offset+1, 
    #                                     avg_cache_resp_all.shape[0])

    # # plot the cache response
    # ax.plot(time_points_seconds, avg_cache_resp_all, color='xkcd:orange', lw=1, zorder=1)
    # ax.fill_between(time_points_seconds, 
    #                     avg_cache_resp_all+sem_cache_resp_all, 
    #                     avg_cache_resp_all-sem_cache_resp_all, 
    #                     color='xkcd:orange', lw=0, alpha=0.4)

    # # add a scale bar
    # sec_scale = 2
    # fr_scale = 0.1
    # scale_0 = np.min(avg_cache_resp_all)
    # ax.hlines(scale_0, -pre_offset, -pre_offset+sec_scale, colors='k', lw=2)
    # ax.vlines(-pre_offset, scale_0, scale_0 + fr_scale, colors='k', lw=2)
    # ax.text(-pre_offset+(sec_scale/2), scale_0-(fr_scale/10), f'{sec_scale} sec.',
    #                     size=axis_label, ha='center', va='top')
    # ax.text(-(pre_offset+1.2), scale_0+(fr_scale/2), f'{fr_scale}\n$log_{{10}}$ FR', rotation='vertical',
    #                     size=axis_label, ha='center', va='center')

    # # remove axes etc
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # labels
    # ax.set_title('all excitatory cells', fontsize=title_size)
    # ax.text(post_offset-1, np.max(avg_cache_resp_all)+fr_scale/5, f'N cells = {np.sum(bird_excitatory_idx).astype(int)}', size=axis_label)

    # plt.show()
    # f.savefig(f'{save_folder}aggregate_cache_psth.png', dpi=600, bbox_inches='tight')

# show and save the example cell plot
plt.show()
f.savefig(f'{save_figs_dir}ex_cache_enhanced.png', dpi=600, bbox_inches='tight')


''' Average across all sessions/birds '''
# regularize and log
log_cache_psth = np.log10(all_cache_psth + 0.1)

# average across all excitatory cells
avg_cache_resp_all = np.mean(log_cache_psth[all_excitatory_idx], axis=0)
sem_cache_resp_all = stats.sem(log_cache_psth[all_excitatory_idx], axis=0)

''' Plot the average/sem cache response across all excitatory cells '''
f, ax = plt.subplots(1, 1, figsize=(4, 4))
time_points_seconds = np.linspace(-pre_offset, post_offset+1, 
                                    avg_cache_resp_all.shape[0])

ax.plot(time_points_seconds, avg_cache_resp_all, color='xkcd:orange', lw=1, zorder=1)
ax.fill_between(time_points_seconds, 
                    avg_cache_resp_all+sem_cache_resp_all, 
                    avg_cache_resp_all-sem_cache_resp_all, 
                    color='xkcd:orange', lw=0, alpha=0.4)

# add a scale bar
sec_scale = 2
fr_scale = 0.04
scale_0 = np.min(avg_cache_resp_all)
ax.hlines(scale_0, -pre_offset, -pre_offset+sec_scale, colors='k', lw=2)
ax.vlines(-pre_offset, scale_0, scale_0 + fr_scale, colors='k', lw=2)
ax.text(-pre_offset+(sec_scale/2), scale_0-(fr_scale/10), f'{sec_scale} sec.',
                    size=axis_label, ha='center', va='top')
ax.text(-(pre_offset+1.2), scale_0+(fr_scale/2), f'{fr_scale}\n$log_{{10}}$ FR', rotation='vertical',
                    size=axis_label, ha='center', va='center')

# remove axes etc
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# labels
ax.set_title('all excitatory cells', fontsize=title_size)
ax.text(post_offset-1, np.max(avg_cache_resp_all)+fr_scale/5, f'N cells = {np.sum(all_excitatory_idx).astype(int)}', size=axis_label)

plt.show()
f.savefig(f'{save_figs_dir}aggregate_cache_psth.png', dpi=600, bbox_inches='tight')


''' Average across all sessions/birds '''
# average across all enhanced excitatory cells
avg_cache_resp_enh = np.mean(log_cache_psth[all_excitatory_idx & (all_modulation_idx==1)], axis=0)
sem_cache_resp_enh = stats.sem(log_cache_psth[all_excitatory_idx & (all_modulation_idx==1)], axis=0)

# average across all suppressed excitatory cells
avg_cache_resp_sup = np.mean(log_cache_psth[all_excitatory_idx & (all_modulation_idx==-1)], axis=0)
sem_cache_resp_sup = stats.sem(log_cache_psth[all_excitatory_idx & (all_modulation_idx==-1)], axis=0)


''' Plot split by significant suppression/enhancement '''
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
time_points_seconds = np.linspace(-pre_offset, post_offset+1, 
                                    avg_cache_resp_all.shape[0])

ax[0].plot(time_points_seconds, avg_cache_resp_enh, color='xkcd:orange', lw=1, zorder=1)
ax[0].fill_between(time_points_seconds, 
                    avg_cache_resp_enh+sem_cache_resp_enh, 
                    avg_cache_resp_enh-sem_cache_resp_enh, 
                    color='xkcd:orange', lw=0, alpha=0.4)
ax[0].set_title('enhanced exc. cells', fontsize=title_size)

ax[1].plot(time_points_seconds, avg_cache_resp_sup, color='xkcd:cerulean', lw=1, zorder=1)
ax[1].fill_between(time_points_seconds, 
                    avg_cache_resp_sup+sem_cache_resp_sup, 
                    avg_cache_resp_sup-sem_cache_resp_sup, 
                    color='xkcd:cerulean', lw=0, alpha=0.4)
ax[1].set_title('suppressed exc. cells', fontsize=title_size)


# formatting
sec_scale = 2
fr_scale = 0.05
scale_0 = np.min(avg_cache_resp_sup)
ylims = ax[0].get_ylim()

for i in range(2):
#     # add a scale bar
#     ax[i].hlines(scale_0, -pre_offset, -pre_offset+sec_scale, colors='k', lw=2)
#     ax[i].vlines(-pre_offset, scale_0, scale_0 + fr_scale, colors='k', lw=2)
#     ax[i].text(-pre_offset+(sec_scale/2), scale_0, f'{sec_scale} sec.',
#                         size=axis_label, ha='center', va='top')
#     ax[i].text(-(pre_offset+1), scale_0, f'{fr_scale}\n$log_{{10}}$ FR', rotation='vertical',
#                         size=axis_label, ha='center', va='center')

    # remove axes etc
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
#     # ax[i].spines['left'].set_visible(False)
#     ax[i].spines['bottom'].set_visible(False)
#     ax[i].set_xticks([])
#     # ax[i].set_yticks([])

    # add a vertical line at cache offset
    ax[i].vlines(0, ylims[0], ylims[1], colors='xkcd:gray', linestyles='dashed', lw=0.5)
ax[0].set_ylim(ylims)
ax[0].set_ylabel(f'$log_{{10}}$ FR', size=axis_label)
f.supxlabel(f'time from cache offset (sec.)', size=axis_label)
ax[0].text(post_offset-2, np.max(avg_cache_resp_enh), 
            f'N cells = {np.sum(all_excitatory_idx & (all_modulation_idx==1)).astype(int)}', size=axis_label)
ax[1].text(post_offset-2, np.max(avg_cache_resp_enh),
            f'N cells = {np.sum(all_excitatory_idx & (all_modulation_idx==-1)).astype(int)}', size=axis_label)

plt.show()
f.savefig(f'{save_figs_dir}aggregate_cache_psth_split.png', dpi=600, bbox_inches='tight')


''' Plot the average/sem cache response suppressed only '''
f, ax = plt.subplots(1, 1, figsize=(4, 4))
time_points_seconds = np.linspace(-pre_offset, post_offset+1, 
                                    avg_cache_resp_all.shape[0])

ax.plot(time_points_seconds, avg_cache_resp_sup, color='xkcd:orange', lw=1, zorder=1)
ax.fill_between(time_points_seconds, 
                    avg_cache_resp_sup+sem_cache_resp_sup, 
                    avg_cache_resp_sup-sem_cache_resp_sup, 
                    color='xkcd:orange', lw=0, alpha=0.4)

# add a scale bar
sec_scale = 2
fr_scale = 0.06
scale_0 = np.min(avg_cache_resp_sup)
ax.hlines(scale_0, -pre_offset, -pre_offset+sec_scale, colors='k', lw=2)
ax.vlines(-pre_offset, scale_0, scale_0 + fr_scale, colors='k', lw=2)
ax.text(-pre_offset+(sec_scale/2), scale_0-(fr_scale/10), f'{sec_scale} sec.',
                    size=axis_label, ha='center', va='top')
ax.text(-(pre_offset+1.2), scale_0+(fr_scale/2), f'{fr_scale}\n$log_{{10}}$ FR', rotation='vertical',
                    size=axis_label, ha='center', va='center')

# remove axes etc
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# labels
ax.set_title('suppressed exc. cells', fontsize=title_size)
ax.text(post_offset-1, np.max(avg_cache_resp_sup)+fr_scale/5,
            f'N cells = {np.sum(all_excitatory_idx & (all_modulation_idx==-1)).astype(int)}', size=axis_label)

plt.show()
f.savefig(f'{save_figs_dir}aggregate_cache_psth_sup.png', dpi=600, bbox_inches='tight')