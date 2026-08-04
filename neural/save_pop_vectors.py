import numpy as np
from scipy import stats

import os 
import sys
from scipy.io import loadmat

from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel
sys.path.append("..//behavior/")
from format_behavior_data import load_behavior_data, get_caches_refined, get_visits_refined, get_retrievals_refined
sys.path.append("..//utils/")
import helpers
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

'''
Compute population vectors associated with different events as in Chettih, Mackevicius et al, 2024

Add to the data dictionary for future use.
'''
# Include only cells bounded by channels with stim response?
proj_only = True
subtract_baseline = True

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# behavioral data frames per second

# to load the arena objects
# arena_dir = 'C:/Users/ilow1/Documents/code/il_rig_control/arena_alignment/' # laptop
arena_dir = 'C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/' # rig computer
arena_items_file = 'arena_items_2.mat'


''' Data params '''
long_thresh = 2 # seconds for long events
baseline_window = 30 # minutes for FR moving average
fps=50 # video frame rate

# for computing/binning distances between perches
convert_norm_to_cm = 13 * 2.54 # conversion factor normalized coordinates to cm
perch_dist_bins = np.asarray([0, 0.17, 0.22, 0.35, 0.49, 0.52, 0.69, 0.77, 0.98, np.inf])
perch_dist_centers = (perch_dist_bins[:-1] + perch_dist_bins[1:]) / 2
perch_dist_centers[-1] = 1.5
perch_dist_centers_cm = perch_dist_centers*convert_norm_to_cm
n_bins = perch_dist_centers.shape[0]

# get the x, y coordinates of each perch center
arena_data = loadmat(f'{arena_dir}{arena_items_file}', squeeze_me=True)
n_sites = arena_data["perch_w_site"].shape[0]
perch_loc = np.zeros((n_sites, 2))
for site in range(n_sites):
    perch_loc[site] = arena_data["perch_w_site"][site]['Centroid']

# load the data dictionary and get bird ids
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = []
for bird in data_dict.keys():
    bird_ids.append(bird)


''' List of behavior sessions '''
all_behavior_sessions = []
for i, bird in enumerate(bird_ids):
    behavior_sessions = []
    for session_id in data_dict[bird]['all_sessions']:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)
    all_behavior_sessions.append(behavior_sessions)

for bird in bird_ids:
    bird_idx = bird_ids.index(bird)
    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions:
        print(f'collecting population vectors for {bird}_{session_id}')

        ''' Get the file params '''
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        data_dir = f"{session_dir}/behavior_data/"
        pred_date = data_dict[bird][session_id]['pred_date']

        ''' Load/format the neural data '''
        # load spike times and get firing rate per cell
        dt = 1 / fps
        spike_frame = np.load(f'{data_dir}aligned_spikes.npy')
        n_cells, n_frames = spike_frame.shape
        inst_firing_rate = spike_frame / dt
        
        ''' Load and format behavior data '''
        # load behavioral data
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
        seed_struct, count_data = load_behavior_data(data_dir)

        # get event onsets, offsets, and perch_ids
        cache_onsets, cache_offsets, cache_ids = get_caches_refined(count_data, seed_struct, n_frames)
        ret_onsets, ret_offsets, ret_ids = get_retrievals_refined(count_data, seed_struct, n_frames)
        visit_onsets, visit_offsets, visit_ids = get_visits_refined(count_data, n_frames)

        # adjust for pythonic indexing
        cache_ids -= 1
        ret_ids -= 1
        visit_ids -= 1

        # exclude visits to feeder perches
        visit_onsets = visit_onsets[visit_ids < n_sites]
        visit_offsets = visit_offsets[visit_ids < n_sites]
        visit_ids = visit_ids[visit_ids < n_sites]

        n_caches = cache_onsets.shape[0]
        n_retrieve = ret_onsets.shape[0]
        n_visits = visit_onsets.shape[0]
        if n_caches == 0:
            print(f'    no caches found, skipping')
            continue

        # get the x, y coordinates of each event
        cache_loc = np.zeros((n_caches, 2))
        for cache_idx, cache_id in enumerate(cache_ids):
            cache_loc[cache_idx] = perch_loc[cache_id]

        ret_loc = np.zeros((n_retrieve, 2))
        for ret_idx, ret_id in enumerate(ret_ids):
            ret_loc[ret_idx] = perch_loc[ret_id]
            
        visit_loc = np.zeros((n_visits, 2))
        for visit_idx, visit_id in enumerate(visit_ids):
            visit_loc[visit_idx] = perch_loc[visit_id]

        ''' Normalize activity for population analysis '''
        moving_avg_fr = np.zeros_like(inst_firing_rate)
        for cell in range(n_cells):
            moving_avg_fr[cell] = helpers.moving_avg(inst_firing_rate[cell], window=baseline_window)
        st_dev_fr = stats.tstd(inst_firing_rate, axis=1) + 0.6
        norm_fr = inst_firing_rate.copy() - moving_avg_fr
        for cell in range(n_cells):
            norm_fr[cell] /= st_dev_fr[cell]

        ''' Compute the population activity vectors '''
        # account for long events as in SC, EM 2024
        long_thresh_frames = int(long_thresh/dt)
        long_window = int(long_thresh/2/dt) # frames

        # get the raw vectors
        def _vectors(onsets, offsets, n_events):
            vecs = np.zeros((n_events, n_cells))
            for i, (s, e) in enumerate(zip(onsets, offsets)):
                if e - s < long_thresh_frames:
                    vecs[i] = np.mean(norm_fr[:, s:e], axis=1)
                else:
                    activity = np.column_stack((norm_fr[:, s:s + long_window], norm_fr[:, e - long_window:e]))
                    vecs[i] = np.mean(activity, axis=1)
            return vecs
        visit_vectors_raw = _vectors(visit_onsets, visit_offsets, n_visits)
        cache_vectors_raw = _vectors(cache_onsets, cache_offsets, n_caches)
        ret_vectors_raw = _vectors(ret_onsets, ret_offsets, n_retrieve)

        ''' Optionally, only keep cells in the projection nucleus '''
        if proj_only:
            stim_idx_cell = idx_cells_by_stim(data_dict, bird, session_id)
            visit_vectors_raw = visit_vectors_raw[:, stim_idx_cell]
            cache_vectors_raw = cache_vectors_raw[:, stim_idx_cell]
            ret_vectors_raw = ret_vectors_raw[:, stim_idx_cell]

        ''' Optionally subtract the baseline for each event type '''
        if subtract_baseline:
            # compute avg population vectors for all events
            avg_visit_vector = np.mean(visit_vectors_raw, axis=0, keepdims=True)
            avg_cache_vector = np.mean(cache_vectors_raw, axis=0, keepdims=True)
            avg_retrieve_vector = np.mean(ret_vectors_raw, axis=0, keepdims=True)

            # subtract the means across all events
            visit_vectors = visit_vectors_raw - avg_visit_vector
            cache_vectors = cache_vectors_raw - avg_cache_vector
            retrieve_vectors = ret_vectors_raw - avg_retrieve_vector
        else:
            # keep only excitatory cells
            exc_idx = data_dict[bird][session_id]['excitatory_idx']
            if proj_only:
                exc_idx = exc_idx[stim_idx_cell]

            visit_vectors = visit_vectors_raw[:, exc_idx]
            cache_vectors = cache_vectors_raw[:, exc_idx]
            retrieve_vectors = ret_vectors_raw[:, exc_idx]

        ''' Save data for future use '''
        if 'barcode_dict' in data_dict[bird][session_id].keys():
            barcode_dict = data_dict[bird][session_id]['barcode_dict']
        else:
            barcode_dict = {}

        barcode_dict['cache_vectors'] = cache_vectors
        barcode_dict['retrieve_vectors'] = retrieve_vectors
        barcode_dict['visit_vectors'] = visit_vectors

        barcode_dict['cache_loc'] = cache_loc
        barcode_dict['retrieve_loc'] = ret_loc
        barcode_dict['visit_loc'] = visit_loc

        data_dict[bird][session_id]['barcode_dict'] = barcode_dict

np.save(data_file, data_dict)