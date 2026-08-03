'''
Create/update a data struct of good sessions for analysis and plotting.

Order of operations:
--------------------
1. make_data_dict.modify_data_dict()
   -> creates the dict if it doesn't exist, adds new birds/sessions,
    
    data_dict[bird]['all_sessions']
        list of existing sessions for this bird
    data_dict[bird][session]['preprocessed_data']  
        list of 'ephys', 'behavior', 'stim'
        if that data exists for this session

2. collect_waveform_data()
    -> ported from neural/save_all_wf_data.py
    -> needs: 'preprocessed_data', 'all_sessions' (step 1)
    
    data_dict[bird][session]['ephys_id'], ['waveform_props'], ['excitatory_idx'], ['inhibitory_idx']
    data_dict[bird][session]['ephys_id']['ks_path'] # todo
    data_dict[bird]['all_waveform_props']

3. get_probe_coords.get_anatomy_info() + save_cell_positions()
    -> existing functions, anatomy/get_probe_coords.py
    -> needs: 'all_sessions', 'preprocessed_data' (step 1)
    -> needs: good_sessions.xlsx spreadsheet (session depths + probe insertion coords)
    
    data_dict[bird]['insert_coords'],
    data_dict[bird][session]['depth'], ['channel_pos'], ['cell_pos'], ['shank_A_idx']

4. align_behavior_spikes()
    -> ported from behavior/save_aligned_spikes.py
    -> needs 'preprocessed_data'  (step 1)

    writes aligned_spikes.npy file per session (does not modify the dict itself)

5. collect_stim_data()
    -> ported from stim/save_all_stim_data.py
    -> needs: 'preprocessed_data', 'all_sessions' (step 1), 'channel_pos' (step 2)

    data_dict[bird][session]['worm_ch_idx'], ['stim_resp_idx_ch'], ['nucleus_dvs'], ['proj_cell_IDs'], ['proj_cell_idx']
    data_dict[bird]['nucleus_dvs']

6. collect_population_vectors()
    # todo update for all events (caches, retrievals, visits, checks)
    -> ported from neural/save_pop_vectors.py
    -> needs: 
        'pred_date' (step 1),
        'ephys_id' + 'waveform_props' + 'excitatory_idx' (step 2)
        optionally 'channel_pos' (step 3, if proj_only=True) # todo check if also needs stim_resp_idx
    
    data_dict[bird][session]['barcode_dict'] 
        (cache/retrieve/visit population vectors + locations, active_cache_frac)

Params:
-------
overwrite : bool
    default False skips sessions that already have the relevant field
    if True, recomputes all fields for each bird/session

TODO:
    - move each `collect_*` function below into the module it was ported from
        (e.g. `collect_waveform_data` -> neural/save_all_wf_data.py)
        so this master script just imports and calls them, and the individual
        scripts become thin CLI wrappers around the same functions
'''

import numpy as np
from scipy import stats
from scipy.io import loadmat
import os
import sys

# ---------------------------------------------------------------------------
# Set paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # adjust if this script lives outside the repo
for sub in ["utils", "neural", "behavior", "stim", "anatomy"]:
    sys.path.append(os.path.join(REPO_ROOT, sub))

import make_data_dict
import format_waveform_data
import waveform_analysis
import get_probe_coords_lhy
import format_behavior_data
import format_chronic_stim
import helpers
import neural_analysis

# ---------------------------------------------------------------------------
# Set root paths
# ---------------------------------------------------------------------------
ROOT_DIR = "Z:/Isabel/data/lhy_implants/"
DATA_FILE = f"{ROOT_DIR}good_session_data.npy"
SESSION_INFO_FILE = f"{ROOT_DIR}good_sessions.xlsx"

# only needed for collect_population_vectors
ARENA_DIR = "C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/"
ARENA_ITEMS_FILE = "arena_items_2.mat"


# ---------------------------------------------------------------------------
# Step 1: create the dict / add birds & sessions
# ---------------------------------------------------------------------------
def get_or_create_data_dict(root_dir, data_file, new_bird_ids=None):
    '''
    Loads the existing dict (or creates a new one)
    Registers any new birds
    Session-level bookkeeping:
        ('all_sessions', 'preprocessed_data')
        is (re)computed for every bird currently in the dict,
        so this function also adds new sessions for existing birds.

    new_bird_ids : list of str, optional
        Birds to add that aren't in the dict yet
        If the dict doesn't exist yet, this must include every bird you want to add
    '''
    if os.path.isfile(data_file):
        data_dict = np.load(data_file, allow_pickle=True).item()
    else:
        data_dict = {}

    existing_birds = list(data_dict.keys())
    print(f"current birds with saved data: {existing_birds}")

    for bird in (new_bird_ids or []):
        if bird not in data_dict:
            data_dict[bird] = {}

    bird_ids = list(data_dict.keys())
    if not bird_ids:
        raise ValueError("No birds in the dict yet! Pass new_bird_ids to add birds.")

    # check each bird's session folders for preprocessed data
    for bird in bird_ids:
        bird_dir = f"{root_dir}{bird}/"
        session_dirs = sorted(os.listdir(bird_dir))
        all_sessions, behavior_sessions = [], []
        ephys_sessions, waveform_sessions, stim_sessions = [], [], []

        for session_folder in session_dirs:
            if bird not in session_folder:
                continue
            session_id = session_folder.split('_')[1]
            all_sessions.append(session_id)
            if session_id not in data_dict[bird]:
                data_dict[bird][session_id] = {}

            # get file paths for preprocessed data
            data_list = []
            for folder in os.listdir(f'{bird_dir}/{session_folder}'):
                if f'{bird}_{session_id}' in folder:
                    # ephys data
                    # todo update logic for multiple recordings?
                    ephys_id = folder[-13:]
                    ephys_sessions.append(ephys_id)
                    for file in os.listdir(f'{bird_dir}/{session_folder}/{folder}'):
                        if 'kilosort4' in file:
                            for f in os.listdir(f'{bird_dir}/{session_folder}/{folder}/{file}'):
                                if 'waveformStruct' in f:
                                    waveform_sessions.append(session_id)
                                    data_list.append('ephys')
                                    data_dict[bird][session_id]['ephys_id'] = ephys_id
                                    data_dict[bird][session_id]['ks_folder'] = file
                        
                        # stim data
                        if 'raw_ephys_output' in file:
                            stim_sessions.append(session_id)
                            data_list.append('stim')
                
                # behavior data
                if 'behavior_data' in folder:
                    behavior_dir = f'{bird_dir}/{session_folder}/{folder}'
                    flag = 0
                    for file in sorted(os.listdir(behavior_dir)):
                        if 'annotatedSeeds' in file:
                            behavior_sessions.append(session_id)
                            data_list.append('behavior')
                        if 'posture_2stage_face.npy' in file:
                            if flag == 1:
                                print(f'Warning! 2 pose tracking files found for {bird}_{session_id}')
                            data_dict[bird][session_id]['pred_date'] = file[:6]
                            flag = 1

            # keep track of preprocessed data
            data_dict[bird][session_id]['preprocessed_data'] = data_list

        ephys_behavior_sessions = list(set(behavior_sessions) & set(waveform_sessions))
        print(f"{bird}: {len(all_sessions)} total sessions, {len(ephys_behavior_sessions)} have behavior & ephys, {len(stim_sessions)} have stim")

        data_dict[bird]['all_sessions'] = all_sessions

    np.save(data_file, data_dict)
    return data_dict, bird_ids


# ---------------------------------------------------------------------------
# Step 2: waveform properties (ported from neural/save_all_wf_data.py)
# ---------------------------------------------------------------------------
def collect_waveform_data(data_dict, bird_ids, root_dir, overwrite=False):
    for bird in bird_ids:
        print(f'\ncollecting waveform data for {bird}')
        bird_dir = f"{root_dir}{bird}/"
        session_list = data_dict[bird]['all_sessions']
        all_props = data_dict[bird].get('all_waveform_props', [])

        for session_id in session_list:
            if (not overwrite) and ('waveform_props' in data_dict[bird][session_id]):
                continue
            if 'ephys' not in data_dict[bird][session_id]['preprocessed_data']:
                continue

            # set paths
            session_dir = f'{bird_dir}/{bird}_{session_id}/'
            ephys_id = data_dict[bird][session_id]['ephys_id']
            ks_id = data_dict[bird][session_id]['ks_folder']
            ks_dir = f"{bird}_{ephys_id}/{ks_id}/"
            if 'lhy' in root_dir:
                ephys_dir = f"{session_dir}{bird}_{ephys_id}/"
            else:
                ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"

            # load and format waveform struct
            waveform_struct = format_waveform_data.load_wf_data(session_dir, ks_dir=ks_dir)
            mean_waveforms, wf_channels, _, ch_names = format_waveform_data.sort_wf_by_channel(
                '', waveform_struct, data_dir=ephys_dir, return_ch_names=True)
            n_cells = mean_waveforms.shape[0]
            wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])

            # collect waveform properties
            fr = waveform_struct['meanRate']
            log_fr = np.log10(fr)
            width = np.zeros(n_cells)
            asymm = np.zeros(n_cells)
            for wf_idx in range(n_cells):
                best_ch = wf_ch_idx[wf_idx]
                width[wf_idx] = waveform_analysis.calc_spike_width(mean_waveforms[wf_idx, best_ch])
                asymm[wf_idx] = waveform_analysis.calc_amp_assym(mean_waveforms[wf_idx, best_ch])

            waveform_props = np.row_stack([asymm, width, log_fr])
            data_dict[bird][session_id]['waveform_props'] = waveform_props
            all_props = waveform_props if len(all_props) == 0 else np.column_stack([all_props, waveform_props])
            data_dict[bird]['all_waveform_props'] = all_props

    # re-cluster excitatory/inhibitory across ALL sessions any time new data is added
    all_waveform_props = []
    sess_idx = 0
    session_index = np.asarray([]).astype(int)
    session_keys = []  # (bird, session_id) in the same order as session_index groups
    for bird in bird_ids:
        for session_id in data_dict[bird]['all_sessions']:
            if 'waveform_props' in data_dict[bird][session_id]:
                wp = data_dict[bird][session_id]['waveform_props']
                n_cells = wp.shape[1]
                all_waveform_props = wp if len(all_waveform_props) == 0 else np.column_stack((all_waveform_props, wp))
                session_index = np.append(session_index, np.full(n_cells, sess_idx))
                session_keys.append((bird, session_id))
                sess_idx += 1

    if len(all_waveform_props) > 0:
        asymm, width, log_fr = all_waveform_props[0], all_waveform_props[1], all_waveform_props[2]
        exc_idx_all, inhib_idx_all = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)
        for i, (bird, session_id) in enumerate(session_keys):
            data_dict[bird][session_id]['excitatory_idx'] = exc_idx_all[session_index == i]
            data_dict[bird][session_id]['inhibitory_idx'] = inhib_idx_all[session_index == i]

    return data_dict

# ---------------------------------------------------------------------------
# Step 4: behavior-aligned spikes per session (ported from behavior/save_aligned_spikes.py)
# ---------------------------------------------------------------------------
def align_behavior_spikes(data_dict, bird_ids, root_dir):
    for bird in bird_ids:
        print(f'\naligning spikes to behavior for {bird}')
        for session_id in data_dict[bird]['all_sessions']:
            preprocessed = data_dict[bird][session_id]['preprocessed_data']
            if ('behavior' in preprocessed) and ('ephys' in preprocessed):
                session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
                neural_analysis.align_spikes_behavior(session_dir)

# ---------------------------------------------------------------------------
# Step 5: stim / antidromic response data (ported from stim/save_all_stim_data.py)
# ---------------------------------------------------------------------------
def collect_stim_response_data(data_dict, bird_ids, root_dir, overwrite=False,
                                sampling_rate=30000, t_pre=0.02, t_post=0.03,
                                spk_thresh=25, start_t=5e-3, end_t=15e-3):
    start_idx = np.round((t_pre + start_t) * sampling_rate).astype(int)
    end_idx = np.round((t_pre + end_t) * sampling_rate).astype(int)

    # identify stim sessions and their ephys dirs
    stim_sessions, ephys_dirs = [], []
    for bird in bird_ids:
        for session_id in data_dict[bird]['all_sessions']:
            preprocessed = data_dict[bird][session_id]['preprocessed_data']
            if ('stim' in preprocessed) and ('ephys' in preprocessed):
                session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
                ephys_id = data_dict[bird][session_id]['ephys_id']
                if 'lhy' in root_dir:
                    ephys_dir = f"{session_dir}{bird}_{ephys_id}/"
                else:
                    ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"
                ephys_dirs.append(ephys_dir)
                stim_sessions.append(f'{bird}_{session_id}')

    for bird in bird_ids:
        print(f'\ncollecting stim response data for {bird}')
        for session_id in data_dict[bird]['all_sessions']:
            if (not overwrite) and ('worm_ch_idx' in data_dict[bird][session_id]):
                continue
            key = f'{bird}_{session_id}'
            if key not in stim_sessions:
                continue
            ephys_dir = ephys_dirs[stim_sessions.index(key)]

            # get the stim params
            stim_params = []
            for file in sorted(os.listdir(ephys_dir)):
                if ('neg' in file) and ('amplifier' in file):
                    stim_pol = file.split(sep='_')[-1][:-4]
                    if 'neg' in stim_pol:
                        stim_params.append(stim_pol)

            # get the indices for stim-responsive channels
            worm_ch_idx = None
            for idx, stim_pol in enumerate(stim_params):
                raw_ephys = format_chronic_stim.load_stim(ephys_dir, stim_pol=stim_pol)
                ephys_data, ch_names = format_chronic_stim.sort_stim_by_channel(ephys_dir, raw_ephys)
                n_channels = ephys_data.shape[0]

                filt_data = format_chronic_stim.filter_stim_for_spikes(ephys_data)
                stim_hash = np.moveaxis(filt_data[:, start_idx:end_idx], -1, 0)
                avg_hash = np.mean(stim_hash, axis=0)

                if idx == 0:
                    worm_ch_idx = np.zeros(n_channels).astype(bool)
                for i in range(n_channels):
                    if any(np.abs(avg_hash[i]) >= spk_thresh):
                        worm_ch_idx[i] = True
            if worm_ch_idx is None:
                continue
            print(f'{session_id} has {np.sum(worm_ch_idx)} total channels with stim responses')

            # collision-verified projection cells
            all_sig_cells, all_sig_idx = np.asarray([]), np.asarray([])
            if 'lhy' in root_dir:
                collision_dir = ephys_dir   # no subfolder to strip in the new structure
            else:
                collision_dir = ephys_dir[:-17]
            for file in sorted(os.listdir(collision_dir)):
                if 'collision_props' in file:
                    collision_dict = np.load(f'{collision_dir}{file}', allow_pickle=True).item()
                    all_sig_cells = np.append(all_sig_cells, collision_dict['sig_cell_IDs'])
                    all_sig_idx = np.append(all_sig_idx, collision_dict['sig_cell_idx'])
            all_sig_cells, unique_idx = np.unique(all_sig_cells, return_index=True)
            all_sig_idx = all_sig_idx[unique_idx]
            print(f'{session_id} has {all_sig_cells.shape[0]} cells with significant collisions (p <= 0.01)')

            data_dict[bird][session_id]['worm_ch_idx'] = worm_ch_idx
            if all_sig_cells.shape[0] > 0:
                data_dict[bird][session_id]['proj_cell_IDs'] = all_sig_cells.astype(int)
                data_dict[bird][session_id]['proj_cell_idx'] = all_sig_idx.astype(int)

    # nucleus depth estimates (requires 'channel_pos' from step 3)
    for bird in bird_ids:
        print(f'\napproximating projection nucleus location for {bird}')
        session_list = data_dict[bird]['all_sessions']
        bird_nucleus_dvs = np.full((len(session_list), 2, 2), np.nan)
        for i, session_id in enumerate(session_list):
            key = f'{bird}_{session_id}'
            if key not in stim_sessions or 'worm_ch_idx' not in data_dict[bird][session_id]:
                continue
            if 'channel_pos' not in data_dict[bird][session_id]:
                print(f'  skipping {key}: no channel_pos yet (run the anatomy step first)')
                continue

            stim_idx = data_dict[bird][session_id]['worm_ch_idx']
            n_channels = stim_idx.shape[0]
            ch_pos = data_dict[bird][session_id]['channel_pos']

            shank_idx = n_channels // 2
            stim_idx_adj = np.zeros(n_channels).astype(bool)
            for ch in range(n_channels):
                if stim_idx[ch]:
                    stim_idx_adj[ch] = True
                    continue
                elif ch < shank_idx:
                    dorsal_resp = np.any(stim_idx[:ch])
                    ventral_resp = np.any(stim_idx[ch + 1:shank_idx])
                else:
                    dorsal_resp = np.any(stim_idx[shank_idx:ch])
                    ventral_resp = np.any(stim_idx[ch + 1:])
                if dorsal_resp and ventral_resp:
                    stim_idx_adj[ch] = True
            stim_idx = stim_idx_adj

            shank_A_idx = np.zeros(n_channels).astype(bool)
            shank_A_idx[:shank_idx] = True
            shank_A_dv = ch_pos[stim_idx & shank_A_idx, -1]
            shank_B_dv = ch_pos[stim_idx & ~shank_A_idx, -1]

            nucleus_dvs = np.full((2, 2), np.nan)
            if shank_A_dv.shape[0] > 0:
                nucleus_dvs[0] = [np.min(shank_A_dv), np.max(shank_A_dv)]
            if shank_B_dv.shape[0] > 0:
                nucleus_dvs[1] = [np.min(shank_B_dv), np.max(shank_B_dv)]
            bird_nucleus_dvs[i] = nucleus_dvs

            data_dict[bird][session_id]['stim_resp_idx_ch'] = stim_idx
            data_dict[bird][session_id]['nucleus_dvs'] = nucleus_dvs

        nucleus_dvs_all = np.full((2, 2), np.nan)
        if not np.all(np.isnan(bird_nucleus_dvs)):
            nucleus_dvs_all[0] = [np.nanmin(bird_nucleus_dvs[:, 0, 0]), np.nanmax(bird_nucleus_dvs[:, 0, 1])]
            nucleus_dvs_all[1] = [np.nanmin(bird_nucleus_dvs[:, 1, 0]), np.nanmax(bird_nucleus_dvs[:, 1, 1])]
        data_dict[bird]['nucleus_dvs'] = nucleus_dvs_all

    return data_dict


# ---------------------------------------------------------------------------
# Step 6: population vectors (ported from neural/save_pop_vectors.py)
# ---------------------------------------------------------------------------
def collect_population_vectors(data_dict, bird_ids, root_dir, arena_dir, arena_items_file,
                                proj_only=False, subtract_baseline=True, overwrite=False,
                                long_thresh=2, baseline_window=30, fps=50):
    from format_behavior_data import load_behavior_data, get_caches_refined, get_visits_refined, get_retrievals_refined
    from format_waveform_data import get_spike_times
    from format_chronic_stim import idx_cells_by_stim

    # get perch locations in the arena
    arena_data = loadmat(f'{arena_dir}{arena_items_file}', squeeze_me=True)
    n_sites = arena_data["perch_w_site"].shape[0]
    perch_loc = np.zeros((n_sites, 2))
    for site in range(n_sites):
        perch_loc[site] = arena_data["perch_w_site"][site]['Centroid']

    for bird in bird_ids:
        for session_id in data_dict[bird]['all_sessions']:
            if (not overwrite) and ('barcode_dict' in data_dict[bird][session_id]):
                continue
            preprocessed = data_dict[bird][session_id]['preprocessed_data']
            if not (('behavior' in preprocessed) and ('ephys' in preprocessed)):
                continue
            if 'pred_date' not in data_dict[bird][session_id] or 'ephys_id' not in data_dict[bird][session_id]:
                print(f'  skipping {bird}_{session_id}: missing pred_date/ephys_id '
                      f'(run collect_behavior_tracking / collect_waveform_data first)')
                continue
            print(f'collecting population vectors for {bird}_{session_id}')

            # set paths
            session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
            data_dir = f"{session_dir}/behavior_data/"

            # load spike times and get firing rate per cell
            dt = 1 / fps
            spike_frame = np.load(f'{data_dir}aligned_spikes.npy')
            n_cells, n_frames = spike_frame.shape
            inst_firing_rate = spike_frame / dt
            waveform_props = data_dict[bird][session_id]['waveform_props']
            avg_firing_rate = 10 ** waveform_props[2]

            # get event onsets, offsets, and perch ids
            seed_struct, count_data = load_behavior_data(data_dir)
            cache_onsets, cache_offsets, cache_ids = get_caches_refined(count_data, seed_struct, n_frames)
            ret_onsets, ret_offsets, ret_ids = get_retrievals_refined(count_data, seed_struct, n_frames)
            visit_onsets, visit_offsets, visit_ids = get_visits_refined(count_data, n_frames)
            cache_ids, ret_ids, visit_ids = cache_ids - 1, ret_ids - 1, visit_ids - 1

            # exclude feeder visits
            visit_onsets = visit_onsets[visit_ids < n_sites]
            visit_offsets = visit_offsets[visit_ids < n_sites]
            visit_ids = visit_ids[visit_ids < n_sites]

            # get event locations in XY arena coords
            n_caches, n_retrieve, n_visits = cache_onsets.shape[0], ret_onsets.shape[0], visit_onsets.shape[0]
            cache_loc = np.asarray([perch_loc[c] for c in cache_ids]).reshape(n_caches, 2)
            ret_loc = np.asarray([perch_loc[r] for r in ret_ids]).reshape(n_retrieve, 2)
            visit_loc = np.asarray([perch_loc[v] for v in visit_ids]).reshape(n_visits, 2)

            # get average cache activity
            long_window = int(long_thresh / 2 / dt)
            avg_cache = np.zeros((n_cells, n_caches))
            for i, (cache_on, cache_off) in enumerate(zip(cache_onsets, cache_offsets)):
                if cache_off - cache_on < long_thresh:
                    spike_count = np.sum(spike_frame[:, cache_on:cache_off], axis=1)
                    occupancy = dt * (cache_off - cache_on)
                    avg_cache[:, i] = spike_count / occupancy
                else:
                    begin_count = np.sum(spike_frame[:, cache_on:cache_on + long_window], axis=1)
                    end_count = np.sum(spike_frame[:, cache_off - long_window:cache_off], axis=1)
                    avg_cache[:, i] = (begin_count + end_count) / long_thresh

            # was activity > average for each cache/cell?
            active_cache = np.zeros_like(avg_cache)
            for c_idx in range(n_cells):
                active_cache[c_idx] = avg_cache[c_idx] > avg_firing_rate[c_idx]
            active_cache_frac = np.sum(active_cache, axis=1) / n_caches

            # get the normalized firing rate for each cell
            moving_avg_fr = np.zeros_like(inst_firing_rate)
            for cell in range(n_cells):
                moving_avg_fr[cell] = helpers.moving_avg(inst_firing_rate[cell], window=baseline_window)
            st_dev_fr = stats.tstd(inst_firing_rate, axis=1) + 0.6
            norm_fr = inst_firing_rate.copy() - moving_avg_fr
            for cell in range(n_cells):
                norm_fr[cell] /= st_dev_fr[cell]

            # get the raw population vectors
            def _vectors(onsets, offsets, n_events):
                vecs = np.zeros((n_events, n_cells))
                for i, (s, e) in enumerate(zip(onsets, offsets)):
                    if e - s < long_thresh:
                        vecs[i] = np.mean(norm_fr[:, s:e], axis=1)
                    else:
                        activity = np.column_stack((norm_fr[:, s:s + long_window], norm_fr[:, e - long_window:e]))
                        vecs[i] = np.mean(activity, axis=1)
                return vecs
            visit_vectors_raw = _vectors(visit_onsets, visit_offsets, n_visits)
            cache_vectors_raw = _vectors(cache_onsets, cache_offsets, n_caches)
            ret_vectors_raw = _vectors(ret_onsets, ret_offsets, n_retrieve)

            # optionally filter by stim response
            if proj_only:
                stim_idx_cell = idx_cells_by_stim(data_dict, bird, session_id)
                visit_vectors_raw = visit_vectors_raw[:, stim_idx_cell]
                cache_vectors_raw = cache_vectors_raw[:, stim_idx_cell]
                ret_vectors_raw = ret_vectors_raw[:, stim_idx_cell]

            # optionally subtract off the average population vector
            if subtract_baseline:
                visit_vectors = visit_vectors_raw - np.mean(visit_vectors_raw, axis=0, keepdims=True)
                cache_vectors = cache_vectors_raw - np.mean(cache_vectors_raw, axis=0, keepdims=True)
                retrieve_vectors = ret_vectors_raw - np.mean(ret_vectors_raw, axis=0, keepdims=True)
            
            # otherwise, just take the excitatory cell activity
            else:
                exc_idx = data_dict[bird][session_id]['excitatory_idx']
                if proj_only:
                    exc_idx = exc_idx[stim_idx_cell]
                visit_vectors = visit_vectors_raw[:, exc_idx]
                cache_vectors = cache_vectors_raw[:, exc_idx]
                retrieve_vectors = ret_vectors_raw[:, exc_idx]

            # make a dictionary of cache-related data
            barcode_dict = data_dict[bird][session_id].get('barcode_dict', {})
            barcode_dict.update({
                'cache_vectors': cache_vectors, 'retrieve_vectors': retrieve_vectors, 'visit_vectors': visit_vectors,
                'cache_loc': cache_loc, 'retrieve_loc': ret_loc, 'visit_loc': visit_loc,
                'active_cache_frac': active_cache_frac,
            })
            data_dict[bird][session_id]['barcode_dict'] = barcode_dict

    return data_dict


# ---------------------------------------------------------------------------
# Build or update the data dictionary
# ---------------------------------------------------------------------------
def build_or_update_session_data(new_bird_ids=None, run_pop_vectors=True,
                                    get_stim_data=False, overwrite=False):
    '''
    Full pipeline

    Pass new_bird_ids=['XYZ01', 'XYZ02'] to create the dict from scratch or register new birds.
    Call with new_bird_ids=None to just refresh session lists / pick up new sessions for existing birds.
    '''
    data_dict, bird_ids = get_or_create_data_dict(ROOT_DIR, DATA_FILE, new_bird_ids)

    print("\n=== waveform properties ===")
    data_dict = collect_waveform_data(data_dict, bird_ids, ROOT_DIR, overwrite=overwrite)
    np.save(DATA_FILE, data_dict)

    print("\n=== anatomy / channel positions ===")
    data_dict = get_probe_coords_lhy.get_raw_anatomy_info(SESSION_INFO_FILE, data_dict)
    data_dict = get_probe_coords_lhy.convert_anatomy_info(data_dict)
    data_dict = get_probe_coords_lhy.save_cell_positions(data_dict, ROOT_DIR)
    np.save(DATA_FILE, data_dict)

    print("\n=== behavior-aligned spikes (per-session files) ===")
    align_behavior_spikes(data_dict, bird_ids, ROOT_DIR)

    if get_stim_data:
        print("\n=== stim / antidromic response data ===")
        data_dict = collect_stim_response_data(data_dict, bird_ids, ROOT_DIR, overwrite=overwrite)
        np.save(DATA_FILE, data_dict)

    if run_pop_vectors:
        print("\n=== population vectors ===")
        data_dict = collect_population_vectors(data_dict, bird_ids, ROOT_DIR, ARENA_DIR, ARENA_ITEMS_FILE,
                                                overwrite=overwrite)
        np.save(DATA_FILE, data_dict)

    print(f"\nDone. Saved to {DATA_FILE}")
    return data_dict


if __name__ == "__main__":
    # Example: add a couple of new birds to an existing (or new) struct
    build_or_update_session_data(new_bird_ids=None, overwrite=True)

#     # Example: just pick up new sessions for birds already in the dict
#     build_or_update_session_data(new_bird_ids=None)