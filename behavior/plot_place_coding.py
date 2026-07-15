'''
Vanilla place cell analysis

Based on place coding analysis in Payne et al. 2021
but excluding caching, retrieving, checking as in
Chettih, Mackevicius et al 2024.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim
sys.path.append("..//utils/")
from color_utils import parula_colormap
import format_behavior_data as format_data 
import analyze_place_codes

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9
parula_cmap = parula_colormap()

''' Recording params '''
fps = 50 # Hz
sampling_rate = 30000 # ephys
dt = 1/fps

fr_thresh = 0.05 # Hz, threshold for excluding low firing cells

''' Load and format the behavioral data '''
# load the data dictionary and get bird ids
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

# define sessions with behavior & ephys
all_behavior_sessions = []
for i, bird in enumerate(bird_ids):
    behavior_sessions = []
    for session_id in data_dict[bird]['all_sessions']:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)
    all_behavior_sessions.append(behavior_sessions)

# to save data across sessions
all_pos_fr = []
all_spatial_info = np.asarray([])
all_exc_idx = np.asarray([]).astype(bool)
all_inhib_idx = np.asarray([]).astype(bool)

for bird in bird_ids:
    print(f'computing place coding for {bird}')
    bird_idx = bird_ids.index(bird)
    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions:
        # get the file params
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        data_dir = f"{session_dir}/behavior_data/"
        pred_date = data_dict[bird][session_id]['pred_date']

        ephys_id = data_dict[bird][session_id]['ephys_id']
        ephys_folder= f"{session_dir}{bird}_{ephys_id}/"
        for folder in sorted(os.listdir(ephys_folder)):
            if 'kilosort4' in folder:
                for file in sorted(os.listdir(f"{ephys_folder}{folder}")):
                    if 'waveformStruct' in file:
                        ks_dir = f"{bird}_{ephys_id}/{folder}/"

        ''' load and format the neural data '''
        # to separate excitatory vs. inhibitory cells
        excitatory_idx = data_dict[bird][session_id]['excitatory_idx']
        inhibitory_idx = data_dict[bird][session_id]['inhibitory_idx']

        # spikes per video frame
        spike_fr = np.load(f"{data_dir}aligned_spikes.npy") # cells x video frames
        avg_firing_rate = np.mean(spike_fr, axis=1) / dt       # session average firing rate

        # filter out low-firing cells and cells not in the projection nucleus
        high_fr = avg_firing_rate > fr_thresh
        if 'stim_resp_idx_ch' in data_dict[bird][session_id].keys():
            stim_idx_cell = idx_cells_by_stim(data_dict, bird, session_id)
        else:
            print(f'warning! no stim data for {bird}_{session_id}')
            stim_idx_cell = np.ones(spike_fr.shape[0]).astype(bool)
        cell_filt_idx = high_fr & stim_idx_cell
        spike_fr = spike_fr[cell_filt_idx]
        avg_firing_rate = avg_firing_rate[cell_filt_idx]
        excitatory_idx = excitatory_idx[cell_filt_idx]
        inhibitory_idx = inhibitory_idx[cell_filt_idx]
        n_cells, n_frames_spk = spike_fr.shape

        ''' load and format the behavior data '''
        # load the seed struct and get session info
        seed_struct, count_data = format_data.load_behavior_data(data_dir)

        # position and speed by video frames
        pos_xy = format_data.get_position(data_dir) # n_timepoints x xy
        abs_speed_xy = format_data.get_speed(pos_xy)
        n_frames_beh = pos_xy.shape[0]

        # check that n_frames is consistent across neural and behavioral data
        if n_frames_beh == n_frames_spk:
            n_frames = n_frames_spk
        else:
            print(f"warning! {n_frames_spk} neural frames vs. {n_frames_beh} behavioral frames")
            n_frames = np.min([n_frames_beh, n_frames_spk])
            pos_xy    = pos_xy[:n_frames]
            abs_speed_xy = abs_speed_xy[:n_frames]
            spike_fr  = spike_fr[:, :n_frames]

        # event times
        cache_onsets, cache_offsets, _ = format_data.get_caches_refined(count_data, seed_struct, n_frames)
        retrieve_onsets, retrieve_offsets, _ = format_data.get_retrievals_refined(count_data, seed_struct, n_frames)
        check_onsets, check_offsets, _ = format_data.get_checks_refined(count_data, seed_struct, n_frames)
        visit_onsets, visit_offsets, _ = format_data.get_visits_refined(count_data, n_frames, exclude_feeders=False)

        ''' get the 2D spatial firing rates and SI '''
        # choose a reasonable speed threshold
        # plt.hist(abs_speed_xy*33.02, bins=200)
        # plt.hist(np.log10(abs_speed_xy), bins=200)
        # plt.show()
        # log_speed_thresh = float(input("input speed threshold: "))
        # speed_thresh_cm = (10**log_speed_thresh)*33.02
        # print(f'speed threshold = {speed_thresh_cm} cm/s')

        # make boolean masks to include only active movement, non-caching etc.
        speed_mask = analyze_place_codes.make_speed_mask(abs_speed_xy, speed_threshold=2)
        # print(f'{np.sum(speed_mask)} out of {n_frames} frames included')
        event_mask = analyze_place_codes.make_event_exclusion_mask(np.append(cache_onsets, retrieve_onsets), 
                                                                    np.append(cache_offsets, retrieve_offsets), 
                                                                    n_frames)
        # event_mask = analyze_place_codes.make_event_exclusion_mask(np.concatenate([cache_onsets, retrieve_onsets, check_onsets]), 
        #                                                             np.concatenate([cache_offsets, retrieve_offsets, check_offsets]), 
        #                                                             n_frames)
        not_visit_mask = analyze_place_codes.make_event_exclusion_mask(visit_onsets, visit_offsets, n_frames)

        mask = speed_mask & event_mask
        # mask = ~not_visit_mask
        # mask = speed_mask

        # smoothed firing rate by xy position bins
        smooth_pos_fr, smooth_pos_time, pos_edges, centers = analyze_place_codes.get_firing_by_pos(pos_xy, spike_fr, mask)

        # spatial information for each cell
        spatial_info = analyze_place_codes.get_spatial_info(smooth_pos_fr, smooth_pos_time)

        # plot for this session
        f, ax, si_sort_idx = analyze_place_codes.plot_place_maps(smooth_pos_fr[excitatory_idx], spatial_info[excitatory_idx], centers,
                                                n_cols=10, cmap=parula_cmap, figsize_per_cell=(1.4, 1.6))
        analyze_place_codes.suptitle_fixed_pad(f, f"{bird} {session_id} (excitatory cells)\ncells sorted by spatial information")
        f.savefig(f'{save_figs_dir}{bird}/place_coding/{bird}_{session_id}_excitatory_maps.png', dpi=600, bbox_inches='tight')
        plt.close()

        if np.sum(inhibitory_idx) > 0:
            f, ax, si_sort_idx = analyze_place_codes.plot_place_maps(smooth_pos_fr[inhibitory_idx], spatial_info[inhibitory_idx], centers,
                                                    n_cols=10, cmap=parula_cmap, figsize_per_cell=(1.4, 1.6))
            analyze_place_codes.suptitle_fixed_pad(f, f"{bird} {session_id} (inhibitory cells)\ncells sorted by spatial information")
            f.savefig(f'{save_figs_dir}{bird}/place_coding/{bird}_{session_id}_inhibitory_maps.png', dpi=600, bbox_inches='tight')
            plt.close()

        # save across all sessions
        if len(all_pos_fr) == 0:
            all_pos_fr = smooth_pos_fr
        else:
            all_pos_fr = np.row_stack([all_pos_fr, smooth_pos_fr])            
        all_spatial_info = np.append(all_spatial_info, spatial_info)
        all_exc_idx = np.append(all_exc_idx, excitatory_idx)
        all_inhib_idx = np.append(all_inhib_idx, inhibitory_idx)

''' Plot the place maps from all the birds '''
f, ax, si_sort_idx = analyze_place_codes.plot_place_maps(all_pos_fr[all_exc_idx], all_spatial_info[all_exc_idx], centers,
                                        n_cols=20, cmap=parula_cmap, figsize_per_cell=(0.7, 0.8))
analyze_place_codes.suptitle_fixed_pad(f, f"all excitatory cells\ncells sorted by spatial information")
f.savefig(f'{save_figs_dir}excitatory_place_maps.png', dpi=600, bbox_inches='tight')

f, ax, si_sort_idx = analyze_place_codes.plot_place_maps(all_pos_fr[all_inhib_idx], all_spatial_info[all_inhib_idx], centers,
                                        n_cols=20, cmap=parula_cmap, figsize_per_cell=(0.7, 0.8))
analyze_place_codes.suptitle_fixed_pad(f, f"all inhibitory cells\ncells sorted by spatial information")
f.savefig(f'{save_figs_dir}inhibitory_place_maps.png', dpi=600, bbox_inches='tight')