import numpy as np
import pandas as pd
import sys
sys.path.append("../utils/")
from load_matlab_data import loadmat_sbx

''' Load and format behavior data '''
def load_behavior_data(data_dir):
    seed_struct = loadmat_sbx(f'{data_dir}annotatedSeeds.mat')['annotatedSeeds']
    count_data = seed_struct['countData']
    return(seed_struct, count_data)


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

    return feeder_open_times, feeder_close_times



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