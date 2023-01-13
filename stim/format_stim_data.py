import numpy as np
import scipy
import sys
sys.path.append("../utils/")

def load_stim(data_folder, bird_id, session_id):
    '''
    Loads a .csv file containing the timestamps and voltage for an
    antidromic stimulation session.

    Returns
    -------
    voltage : ndarray, shape (n_obs,)
    timestamps : ndarray, shape (n_obs,)
        time in seconds of each observation
    '''
    # load the raw data
    raw_timestamps = np.genfromtxt(f"{data_folder}{bird_id}/{session_id}",
                                   delimiter=',',
                                   usecols=(0),
                                   dtype=str)
    voltage = np.genfromtxt(f"{data_folder}{bird_id}/{session_id}",
                                 delimiter=',',
                                 usecols=(1),
                                 dtype=float)
    n_obs = raw_timestamps.shape[0]

    # extract the timestamps, convert to seconds, and set time zero
    offset_timestamps = np.zeros(n_obs)
    for i, raw_t in enumerate(raw_timestamps):
        sec = float(raw_t[-10:])
        min_str = raw_t[-13:-11]
        if min_str[0] == '0':
            mins = float(min_str[1])
        else:
            mins = float(min_str)
        hr_str = raw_t[-16:-14]
        if hr_str[0] == '0':
            hrs = float(hr_str[1])
        else:
            hrs = float(hr_str)
            
        offset_timestamps[i] = sec + mins*60 + hrs*60*60        
    timestamps = offset_timestamps - offset_timestamps[0]

    return voltage, timestamps


def bin_by_time(voltage, timestamps, dt=0.0001):
    # define the timeseries
    t_max = np.max(timestamps)
    t = np.arange(0, t_max + dt, dt)

    # bin by time
    bdx = np.digitize(timestamps, t)
    v = np.empty(t.shape[0])
    v[:] = np.nan
    for b in np.unique(bdx):
        v[b] = np.mean(voltage[bdx==b])
    v_new = nan_interp(v)

    return v_new, t


# helper functions
def nan_interp(y):
    def find(x):
        return x.nonzero()[0]
    nans = np.isnan(y)
    y[nans] = np.interp(find(nans),find(~nans),y[~nans])
    return y