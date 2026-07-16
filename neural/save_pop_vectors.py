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
proj_only = False
subtract_baseline = True

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# to load the arena objects
# arena_dir = 'C:/Users/ilow1/Documents/code/il_rig_control/arena_alignment/' # laptop
arena_dir = 'C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/' # rig computer
arena_items_file = 'arena_items_2.mat'


''' Data params '''
# for grabbing activity around cache events
long_thresh = 2 # seconds

# for computing/binning distances between perches
convert_norm_to_cm = 13 * 2.54 # conversion factor normalized coordinates to cm
perch_dist_bins = np.asarray([0, 0.01, 0.25, 0.45, 0.62, 0.76, 0.92, 1.12, 1.29, np.inf])
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
        spike_bool = spike_frame.astype(bool)

        # instantaneous firing rate
        inst_firing_rate = spike_frame/dt

        # session average firing rate
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]
        avg_firing_rate = 10**log_fr
        
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

        ''' Average neural activity during each cache '''
        # account for long caches as in SC, EM 2024
        long_window = int(long_thresh/2/dt) # frames

        # get average activity during cache window & PSTH
        avg_cache = np.zeros((n_cells, n_caches))
        for i, (cache_on, cache_off) in enumerate(zip(cache_onsets, cache_offsets)):
            if cache_off - cache_on < long_thresh:
                spike_count = np.sum(spike_frame[:, cache_on:cache_off], axis=1)
                occupancy = dt*(cache_off-cache_on)
                avg_cache[:, i] = spike_count/occupancy
            else:
                begin_count = np.sum(spike_frame[:, cache_on:cache_on+long_window], axis=1)
                end_count = np.sum(spike_frame[:, cache_off-long_window:cache_off], axis=1)
                spike_count = begin_count + end_count
                avg_cache[:, i] = spike_count/long_thresh

        # fraction of caches w/ FR > session avg
        active_cache = np.zeros_like(avg_cache)
        for c_idx in range(n_cells):
            active_cache[c_idx] = avg_cache[c_idx] > avg_firing_rate[c_idx]
        active_cache_frac = np.sum(active_cache, axis=1) / n_caches


        ''' Normalize activity for population analysis '''
        # get the baseline rate for each cell (running 30min avg activity)
        baseline_window = 30 # minutes
        moving_avg_fr = np.zeros_like(inst_firing_rate)
        for cell in range(n_cells):
            moving_avg_fr[cell] = helpers.moving_avg(inst_firing_rate[cell], window=baseline_window)

        # get the standard deviation (regularize by adding 0.6 Hz)
        st_dev_fr = stats.tstd(inst_firing_rate, axis=1) + 0.6
        assert st_dev_fr.shape[0] == n_cells

        # normalize
        norm_fr = inst_firing_rate.copy()
        norm_fr -= moving_avg_fr
        for cell in range(n_cells):
            norm_fr[cell] /= st_dev_fr[cell]

        ''' Compute the population activity vectors '''
        # collect all the population vectors for this session
        visit_vectors_raw = np.zeros((n_visits, n_cells))
        for i, (vs, ve) in enumerate(zip(visit_onsets, visit_offsets)):
            visit_vectors_raw[i] = np.mean(norm_fr[:, vs:ve], axis=1)

        cache_vectors_raw = np.zeros((n_caches, n_cells))    
        for i, (cs, ce) in enumerate(zip(cache_onsets, cache_offsets)):
            if ce - cs < long_thresh:
                cache_vectors_raw[i] = np.mean(norm_fr[:, cs:ce], axis=1)
            else:
                cache_activity_start = norm_fr[:, cs:cs+long_window]
                cache_activity_end = norm_fr[:, ce-long_window:ce]
                cache_activity = np.column_stack((cache_activity_start, cache_activity_end))
                cache_vectors_raw[i] = np.mean(cache_activity, axis=1)

        ret_vectors_raw = np.zeros((n_retrieve, n_cells))    
        for i, (cs, ce) in enumerate(zip(ret_onsets, ret_offsets)):
            if ce - cs < long_thresh:
                ret_vectors_raw[i] = np.mean(norm_fr[:, cs:ce], axis=1)
            else:
                cache_activity_start = norm_fr[:, cs:cs+long_window]
                cache_activity_end = norm_fr[:, ce-long_window:ce]
                cache_activity = np.column_stack((cache_activity_start, cache_activity_end))
                ret_vectors_raw[i] = np.mean(cache_activity, axis=1)


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

        barcode_dict['active_cache_frac'] = active_cache_frac

        data_dict[bird][session_id]['barcode_dict'] = barcode_dict

np.save(data_file, data_dict)