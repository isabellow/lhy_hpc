import numpy as np
from scipy import stats
import pandas as pd
import sys
sys.path.append("../utils/")
from load_matlab_data import loadmat_sbx

''' Load and format behavior data '''
def load_behavior_data(data_dir):
    seed_struct = loadmat_sbx(f'{data_dir}annotatedSeeds.mat')['annotatedSeeds']
    count_data = seed_struct['countData']
    return(seed_struct, count_data)

''' Classify arena interactions '''
def get_cache_ints(count_data, seed_struct):
    '''
    Caches are site interactions where a seed is added
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # caches = add a seed
    cache_onsets = all_int_start[all_int_changes > 0]
    cache_offsets = all_int_end[all_int_changes > 0]

    return cache_onsets, cache_offsets

def get_caches_refined(count_data, seed_struct, n_total_frames, dt=0.02):
    '''
    Caches are site interactions where a seed is added

    Returns cache onset and offset times, 
    as well as the perch ID for each cache
    
    Define a cache window as in SC, EM 2024
    - 250 ms before cache onset to 250 ms after cache offset
    - truncated to avoid other interactions
    - caches > 2 sec, only include 1 sec after onset and 1 sec before offset
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_site_idx = count_data['siteNum']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # caches = add a seed
    cache_onsets_raw = all_int_start[all_int_changes > 0]
    cache_offsets_raw = all_int_end[all_int_changes > 0]
    cache_perch_idx = all_site_idx[all_int_changes > 0]

    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    n_perches = all_perch_start.shape[0]

    # +/-240 ms window around cache, avoiding other events
    t_window = 0.25/dt
    cache_onsets = np.asarray([])
    cache_offsets = np.asarray([])
    for cs, ce in zip(cache_onsets_raw, cache_offsets_raw):
        # create the time window
        cache_start = cs - t_window
        cache_end = ce + t_window

        # get the perch index for this cache event
        temp_perch_start = all_perch_start.copy()
        temp_perch_start[temp_perch_start > cs] = 0
        cache_idx = np.argmin(cs-temp_perch_start)

        # check for overlap with other events
        if cache_idx > 0:
            if all_perch_end[cache_idx-1] >= cache_start:
                cache_start = all_perch_end[cache_idx-1]
        if cache_idx < n_perches-1:
            if all_perch_start[cache_idx+1] <= cache_end:
                cache_end = all_perch_start[cache_idx+1]

        # check session ends
        if cache_start < 0:
            cache_start = 0
        if cache_end > n_total_frames:
            cache_end = n_total_frames

        cache_onsets = np.append(cache_onsets, cache_start)
        cache_offsets = np.append(cache_offsets, cache_end)

    return cache_onsets.astype(int), cache_offsets.astype(int), cache_perch_idx.astype(int)

def get_retrievals_refined(count_data, seed_struct, n_total_frames, dt=0.02):
    '''
    Retrievals are site interactions where a seed is removed

    Returns retrieval onset and offset times, 
    as well as the perch ID for each retrieval
    
    Define a retrieval window as in SC, EM 2024
    - 250 ms before retrieval onset to 250 ms after retrieval offset
    - truncated to avoid other interactions
    - retrievals > 2 sec, only include 1 sec after onset and 1 sec before offset
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_site_idx = count_data['siteNum']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # retrievals = remove a seed
    ret_onsets_raw = all_int_start[all_int_changes < 0]
    ret_offsets_raw = all_int_end[all_int_changes < 0]
    ret_perch_idx = all_site_idx[all_int_changes < 0]

    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    n_perches = all_perch_start.shape[0]

    # +/-240 ms window around retrieval, avoiding other events
    t_window = 0.25/dt
    ret_onsets = np.asarray([])
    ret_offsets = np.asarray([])
    for cs, ce in zip(ret_onsets_raw, ret_offsets_raw):
        # create the time window
        ret_start = cs - t_window
        ret_end = ce + t_window

        # get the perch index for this retrieval event
        temp_perch_start = all_perch_start.copy()
        temp_perch_start[temp_perch_start > cs] = 0
        ret_idx = np.argmin(cs-temp_perch_start)

        # check for overlap with other events
        if ret_idx > 0:
            if all_perch_end[ret_idx-1] >= ret_start:
                ret_start = all_perch_end[ret_idx-1]
        if ret_idx < n_perches-1:
            if all_perch_start[ret_idx+1] <= ret_end:
                ret_end = all_perch_start[ret_idx+1]

        # check session ends
        if ret_start < 0:
            ret_start = 0
        if ret_end > n_total_frames:
            ret_end = n_total_frames

        ret_onsets = np.append(ret_onsets, ret_start)
        ret_offsets = np.append(ret_offsets, ret_end)

    return ret_onsets.astype(int), ret_offsets.astype(int), ret_perch_idx.astype(int)

def get_visits_raw(count_data):
    '''
    Visits are perch interactions without eating or site interaction
    '''
    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    all_perch_idx = count_data['perchNum']
    n_perches = all_perch_start.shape[0]

    # collect non-visit interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    n_interactions = all_int_start.shape[0]
    eat_onsets = count_data['newBeakPerch']
    eat_offsets = count_data['endBeakPerch']
    all_non_visit_start = np.append(eat_onsets, all_int_start)

    # get visits by excluding other interactions
    visits = np.full(n_perches, 1).astype(bool)
    for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
        start_idx = all_non_visit_start > ps
        end_idx = all_non_visit_start < pe
        if any(start_idx & end_idx):
            visits[i] = False        
    visit_onsets = all_perch_start[visits]
    visit_offsets = all_perch_end[visits]

    return visit_onsets, visit_offsets

def get_visits_refined(count_data, n_total_frames, dt=0.02):
    '''
    Visits are perch interactions without eating or site interaction

    Returns visit onset and offset times, 
    as well as the perch ID for each visit

    Define a visit window as in SC, EM 2024
    +/- 500 ms from perch arrival, truncated to avoid other interactions
    '''
    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    all_perch_idx = count_data['perchNum']
    n_perches = all_perch_start.shape[0]

    # collect non-visit interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    n_interactions = all_int_start.shape[0]
    eat_onsets = count_data['newBeakPerch']
    eat_offsets = count_data['endBeakPerch']
    all_non_visit_start = np.append(eat_onsets, all_int_start)

    # get visits by excluding other interactions
    visits = np.full(n_perches, 1).astype(bool)
    for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
        start_idx = all_non_visit_start > ps
        end_idx = all_non_visit_start < pe
        if any(start_idx & end_idx):
            visits[i] = False        

    # 1000ms window around visit onset, avoiding other events
    t_window = 0.5/dt
    visit_onsets = np.asarray([])
    visit_offsets = np.asarray([])
    for i, visit_t in enumerate(all_perch_start):
        if visits[i]:
            visit_start = visit_t - t_window
            visit_end = visit_t + t_window

            # check for overlap with other events
            if i > 0:
                if all_perch_end[i-1] >= visit_start:
                    visit_start = all_perch_end[i-1]
            if i < n_perches-1:
                if all_perch_start[i+1] <= visit_end:
                    visit_end = all_perch_start[i+1]

            # check session ends
            if visit_start < 0:
                visit_start = 0
            if visit_end > n_total_frames:
                visit_end = n_total_frames

            visit_onsets = np.append(visit_onsets, visit_start)
            visit_offsets = np.append(visit_offsets, visit_end)

    return visit_onsets.astype(int), visit_offsets.astype(int), all_perch_idx[visits].astype(int)


def get_n_seeds(seed_struct):
    '''
    Get the number of seeds in the arena (roughly n cached seeds)
    at the time of each interaction
    '''
    count_data = seed_struct['countData']
    n_init = np.sum(seed_struct['initSeedCounts'])
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    all_int_start = count_data['newSite']
    n_seeds_arena = np.cumsum(all_int_changes) + n_init
    if all_int_start[0] > 0:
        n_seeds_arena = np.insert(n_seeds_arena, 0, n_init)
    return n_seeds_arena



def dist_binned_mean_sem(vector_correlations, vector_distances, distance_bin_edges):
    '''
    Given correlations and distances, process the data and
    return distance-binned means and SEMs
    '''
    n_bins = int(distance_bin_edges.shape[0] - 1)

    # remove nans
    keep_idx = np.abs(np.isnan(vector_correlations)-1).astype(bool)
    vector_correlations = vector_correlations[keep_idx]
    vector_distances = vector_distances[keep_idx]

    # compute the average and sem
    dist_bin_idx = np.digitize(vector_distances, distance_bin_edges)-1
    avg_correlations = np.zeros(n_bins)
    sem_correlations = np.zeros(n_bins)
    for b_idx in range(n_bins):
        avg_correlations[b_idx] = np.mean(vector_correlations[dist_bin_idx==b_idx])
        sem_correlations[b_idx] = stats.sem(vector_correlations[dist_bin_idx==b_idx]) 

    return  avg_correlations, sem_correlations



''' Classify feeder interactions '''
def get_feeder_ints(count_data, use_beak=True, frame_rate=50, feeder_perches=np.asarray([84, 85, 86, 87])):
    '''
    Parses count_data to extract feeder interactions.

    Params
    ------
    count_data : dict
        dict of interaction data from get_site_interactions.py
    use_beak : bool
        if True, feeder interactions are defined by the beak near the feeder
        else, feeder interactions are defined by the feet on the feeder perch
    frame_rate : int
        behavioral video frame rate in Hz
    feeder_perches : array of ints
        feeder perch ID numbers (as defined in get_site_interactions.py)

    Returns
    -------
    feeder_int_start/end : array, shape (n_feeder_int,)
        feeder interaction start/end frame numbers
        interaction is classified as the beak getting close to the feeder
        (as defined in get_site_interactions.py)
        or feet on the feeder perch, determined by the use_beak param
    feeder_idx : array of ints, shape (n_feeders,)
        the feeder ID number associated with each interaction
    '''
    if use_beak:
        ''' beak near feeder '''
        # get all feeder interactions
        feeder_int_start = count_data['newFeeder']
        feeder_int_end = count_data['endFeeder']
        feeder_idx = count_data['feederNum']
        
    else:
        ''' feet on feeder perch '''
        # get all perch interactions
        all_perch_start = count_data['newPerch']
        all_perch_end = count_data['endPerch']
        all_perch_idx = count_data['perchNum']
        n_perches = all_perch_start.shape[0]

        # get all feeder interactions
        feeder_int_start = []
        feeder_int_end = []
        feeder_idx = []
        
        for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
            this_perch = all_perch_idx[i]
            if this_perch in feeder_perches:
                feeder_int_start.append(ps)
                feeder_int_end.append(pe)
                feeder_idx.append(np.where(feeder_perches==this_perch)[0][0] + 1)
        feeder_int_start = np.asarray(feeder_int_start)
        feeder_int_end = np.asarray(feeder_int_end)
        feeder_idx = np.asarray(feeder_idx)

    return feeder_int_start, feeder_int_end, feeder_idx

def get_feeder_periods(session_info_file, bird, session_id):
    '''
    Extracts feeder open/close times from the session info spreadsheet
    and converts them into numpy arrays.
    '''
    # get the session IDs
    session_info = pd.read_excel(session_info_file, sheet_name=bird, header=1)
    session_info["id"] = session_info["date"].dt.strftime("%y%m%d")
    
    # get the feeder times for this session
    feeder_times_raw = session_info.loc[session_info["id"] == session_id,
                                                   "feeder open times"].iloc[0]
    
    # convert to numpy arrays of open and close times
    feeder_ranges = feeder_times_raw.split(sep=', ')
    feeder_open_list = []
    feeder_close_list = []
    for fr in feeder_ranges:
        times = fr.split(sep='-')
        feeder_open_list.append(times[0])
        feeder_close_list.append(times[1])
    feeder_open_times = [int(t) for t in feeder_open_list]
    feeder_close_times = [int(t) for t in feeder_close_list]

    return np.asarray(feeder_open_times), np.asarray(feeder_close_times)



def classify_feeder_ints(feeder_int_start, feeder_int_end, 
                            feeder_open_times, feeder_close_times):
    '''
    For each feeder interaction, was the feeder open or closed?

    Params
    ------
    feeder_int_start/end : array, shape (n_feeder_int,)
        feeder interaction start/end frame numbers
    feeder_open/close_times : array, shape (n_feeder_periods,)
        times in minutes that feeders opened/closed

    Returns
    -------
    feeder_status : array of floats, shape (n_feeder_int,)
        feeder status for each interaction
        either 0 (closed the entire interaction), 1 (open the entire interaction),
        or 0.5 (closed or opened during the interaction)
    '''
    n_feeder_int = feeder_int_start.shape[0]

    # convert feeder times to frames
    feeder_open_frames = feeder_open_times*60*50
    feeder_close_frames = feeder_close_times*60*50

    # classify each interaction as open vs. closed
    feeder_status = np.full(n_feeder_int, 0)
    for i, (fs, fe) in enumerate(zip(feeder_int_start, feeder_int_end)):
        start_status = 0
        end_status = 0
        for fo, fc in zip(feeder_open_frames, feeder_close_frames):
            if (fs > fo) & (fs < fc):
                start_status = 1
            if (fe > fo) & (fe < fc):
                end_status = 1
        feeder_status[i] = np.mean((start_status, end_status))

    return feeder_status