import numpy as np
from scipy import stats

import os 
import sys
from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel
sys.path.append("..//behavior/")
from format_behavior_data import load_behavior_data, get_caches_refined

'''
Circularly permute cache times relative to neural data and recompute cache activity 1000x

Select a random number of frames, from 0 to n_frames with replacement,
to shift the cache onset/offset times by.

If the new "cache" extends past the beginning or end of the session, wrap activity around.
'''
''' Data params '''
n_shuffles = 1000

# account for long caches as in SC, EM 2024
long_thresh = 2 # seconds

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"


''' Load the data dictionary and get bird ids '''
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
        print(f'collecting shuffled cache activity for {bird}_{session_id}')

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
        n_caches = cache_onsets.shape[0]

        ''' Get average cache activity for real cache times '''
        long_window = int(long_thresh/2/dt) # frames

        # get average activity during the cache window
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

        ''' Activity during shuffled cache windows '''
        # n frames to shift by
        frame_shifts = np.random.choice(np.arange(n_frames), size=n_shuffles)
        shuff_onsets = np.zeros((n_caches, n_shuffles)).astype(int)
        shuff_offsets = np.zeros((n_caches, n_shuffles)).astype(int)
        for cache_idx, (cache_on, cache_off) in enumerate(zip(cache_onsets, cache_offsets)):
            shuff_onsets[cache_idx] = cache_on + frame_shifts
            shuff_offsets[cache_idx] = cache_off + frame_shifts

        # to store shuffled cache activity
        shuff_avg_cache = np.zeros((n_cells, n_caches, n_shuffles))
        for shuff_idx in range(n_shuffles):
            shuff_ons = shuff_onsets[:, shuff_idx]
            shuff_offs = shuff_offsets[:, shuff_idx]

            # get average activity during fake cache window
            for cache_idx, (cache_on, cache_off) in enumerate(zip(shuff_ons, shuff_offs)):
                if cache_on > n_frames:
                    cache_on = cache_on - n_frames
                    cache_off = cache_off - n_frames
                    spike_frame_cache = spike_frame[:, cache_on:cache_off]
                elif cache_off > n_frames:
                    spike_frame_start = spike_frame[:, cache_on:]
                    spike_frame_end = spike_frame[:, :(cache_off - n_frames)]
                    spike_frame_cache = np.column_stack((spike_frame_start, spike_frame_end))
                else:
                    spike_frame_cache = spike_frame[:, cache_on:cache_off]

                if cache_off - cache_on < long_thresh:
                    spike_count = np.sum(spike_frame_cache, axis=1)
                    occupancy = dt*(cache_off-cache_on)
                    shuff_avg_cache[:, cache_idx, shuff_idx] = spike_count/occupancy
                else:
                    begin_count = np.sum(spike_frame_cache[:, :long_window], axis=1)
                    end_count = np.sum(spike_frame_cache[:, long_window:], axis=1)
                    spike_count = begin_count + end_count
                    shuff_avg_cache[:, cache_idx, shuff_idx] = spike_count/long_thresh

        # fraction of caches w/ FR > session avg
        active_cache_shuff = np.zeros_like(shuff_avg_cache)
        for c_idx in range(n_cells):
            active_cache_shuff[c_idx] = shuff_avg_cache[c_idx] > avg_firing_rate[c_idx]
        active_cache_frac_shuff = np.sum(active_cache_shuff, axis=1) / n_caches

        ''' Compare to shuffle '''
        # -1 = sig suppressed, 1 = sig enhanced, 0 = no change
        cache_modulated = np.zeros(n_cells).astype(int)
        for c_idx in range(n_cells):
            pcts = np.percentile(active_cache_frac_shuff[c_idx], [5, 95])
            if active_cache_frac[c_idx] < pcts[0]:
                cache_modulated[c_idx] = -1
            elif active_cache_frac[c_idx] > pcts[1]:
                cache_modulated[c_idx] = 1

        ''' Save everything '''
        if 'barcode_dict' in data_dict[bird][session_id].keys():
            barcode_dict = data_dict[bird][session_id]['barcode_dict']
        else:
            barcode_dict = {}
        barcode_dict['active_cache_frac'] = active_cache_frac
        barcode_dict['shuff_avg_cache'] = shuff_avg_cache
        barcode_dict['cache_modulated'] = cache_modulated

        data_dict[bird][session_id]['barcode_dict'] = barcode_dict
        
np.save(data_file, data_dict)