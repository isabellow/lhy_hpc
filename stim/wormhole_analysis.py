'''
Functions to detect antidromic responses in silicon probe data
'''
import numpy as np
from scipy import stats
from scipy.spatial import distance as dist

sampling_rate = 30000

def one_changepoint_vectors(X):
    '''
    Brute force method to find a single change point in a vector.
    
    Given a matrix of vectors X, finds the changepoint time (tau)
    that maximizes the likelihood ratio (LR). The LR compares the
    max likelihood for a model w/ a change at tau to the max likelihood
    for a model w/ no change.
    
    This LR can than be compared to shuffle to determine if the change
    is greater than expected by chance.
    
    Params
    ------
    X : single trial hash of shape (n_trials, n_channels*n_samples) where n_trials >= 2
        
    Returns
    -------
    LR : the likelihood ratio statistic
    tau_hat : the estimated change point
    '''
    # data params
    n = X.shape[0]
    
    # get the means
    tau = np.arange(n-1) + 1
    M1 = np.cumsum(X, axis=0)[:-1] / tau[:, None]
    M2 = np.cumsum(X[::-1], axis=0)[1:] / tau[:, None]
    
    # Euclidean distance between mean vectors
    D = np.sqrt(np.sum((M1 - M2[::-1])**2, axis=1))
    
    # LR statistic at timepoints tau
    LR = ((D**2) * tau * (n - tau)) / n
    tau_hat = np.argmax(LR)
    
    return LR[tau_hat], tau_hat 

def one_change_point(x):
    '''
    Brute force method to find a single change point in a vector.
    
    Given a vector x, finds the changepoint time (tau) that maximizes
    the likelihood ratio (LR). The LR compares the max likelihood
    for a model w/ a change at tau to the max likelihood for a
    model w/ no change.
    
    This LR can than be compared to shuffle to determine if the change
    is greater than expected by chance.
    
    Params
    ------
    x : correlations or distances b/w single trial hash and bulk avg
        shape (n_trials,) where n_trials >= 2
        
    Returns
    -------
    LR : the likelihood ratio statistic
    tau_hat : the estimated change point
    '''
    # data params
    n = x.shape[0]
    S = np.cumsum(x)
    
    # possible change points
    tau = np.arange(n-1) + 1
    
    # difference in means
    D = (S[tau-1] / (tau)) - ((S[n-1] - S[tau-1]) / (n - tau))
    
    # LR statistic at timepoints tau
    LR = ((D**2) * tau * (n - tau)) / n
    tau_hat = np.argmax(LR)
    
    return LR[tau_hat], tau_hat 

def subsample_trials(sorted_lat, # sorted latencies to stim (nearest to furthest) for this cell
                     n_trials=100, # number of stim trials to take
                     min_thresh=50e-3*sampling_rate, # closest latency to stim for baseline trials
                     collision_thresh=15e-3*sampling_rate): # latency past which collisions are possible                
    '''
    Get n_trials with spikes near to the stim, ordered from closest to furthest latency.
    Mix of possible collision trials (defined by collision_thresh) and non-collisions trials 
    (defined by min_thresh).
    '''
    # data params
    all_stim_idx = np.arange(sorted_lat.shape[0])
    all_stim_idx = all_stim_idx[np.isfinite(sorted_lat)]
    finite_lat = sorted_lat[np.isfinite(sorted_lat)]

    # n_trials trials with spikes near to the hash (mix of collision and non-collision)
    short_idx = all_stim_idx[finite_lat <= collision_thresh]
    n_short = short_idx.shape[0]
    if n_short > n_trials//2:
        short_idx = np.random.choice(all_stim_idx[finite_lat <= collision_thresh],
                                     size=n_trials//2, replace=False)
        n_short = short_idx.shape[0]
    long_idx = np.random.choice(all_stim_idx[finite_lat >= min_thresh], 
                                size=n_trials-n_short, replace=False)
    
    return np.append(np.sort(short_idx), np.sort(long_idx)), n_short
    
def subsample_channels(best_ch, A_shank=np.arange(32),
                        B_shank=np.arange(32, 64), n_wf_ch=7):
    '''
    Gets the index of n_wf_ch channels centered on the best channel (best_ch) for a 
    given waveform. Shifts this index as-needed to stay on the same shank as the
    best channel and avoid going past shank ends.

    A_shank and B_shank define the channel indices on each shank.
    '''
    if best_ch - n_wf_ch//2 < 0: # near tip of A
        hash_ch_idx = np.arange(n_wf_ch)
    elif (best_ch <= A_shank[-1]) & (best_ch + n_wf_ch//2 > A_shank[-1]): # near top of A
        hash_ch_idx = np.arange(A_shank[-1] - n_wf_ch, A_shank[-1])
    elif (best_ch > A_shank[-1]) & (best_ch - n_wf_ch//2 <= A_shank[-1]): # near tip of B
        hash_ch_idx = np.arange(B_shank[0], B_shank[0] + n_wf_ch)
    elif best_ch + n_wf_ch//2 > B_shank[-1]: # near top of B
        hash_ch_idx = np.arange(B_shank[-1] - n_wf_ch, B_shank[-1])
    else: # mid-shank
        hash_ch_idx = np.arange(best_ch - n_wf_ch//2, best_ch + n_wf_ch//2 + 1)
        
    return hash_ch_idx

def shuffle_LR(hash_subsamp, hash_baseline, all_stim_idx, n_trials=50):
    '''
    Get the shuffled change points for a given number of trials.

    Params
    ------
    hash_subsamp : ndarray of hash matched to that used to compute the changepoint
                    e.g., for a given set of channels
                    shape (n_stim, n_wf_ch, n_samples)
    hash_baseline : ndarray of average hash also subsampled appropriately
                    shape (n_wf_ch, n_samples)
    all_stim_idx : ndarray of stim trial indices; np.arange(n_stim)
    '''
    shuff_idx = np.random.choice(all_stim_idx, size=n_trials, replace=False)
    hash_unwrapped = np.reshape(hash_subsamp[shuff_idx], (n_trials, -1))
    all_shuff_hash = np.row_stack((hash_unwrapped, hash_baseline))

    # get the distance to the baseline hash
    hash_dist = []
    for i in range(n_trials):
        hash_dist.append(dist.pdist(all_shuff_hash[[i, -1]], 'correlation'))
    hash_dist = np.asarray(hash_dist)

    # get the change point
    shuff_LR, _ = one_change_point(hash_dist)
    
    return shuff_LR


def get_shuffle_dist(stim_hash, avg_hash, all_stim_idx,
                        A_shank, B_shank, best_ch, 
                        n_wf_ch=7, num_shuff=1000):
    '''
    Compute the shuffled change points for a given set of channels,
    defined as in subsample_channels.
    '''
    # get the subsampled hash
    hash_ch_idx = subsample_channels(best_ch,  A_shank, B_shank, n_wf_ch=n_wf_ch)
    hash_subsamp = stim_hash[:, hash_ch_idx]
    hash_baseline = np.ravel(avg_hash[hash_ch_idx])

    # computed the shuffled change points
    shuff_LR = np.full(num_shuff, np.NaN)
    for s in range(num_shuff):
        shuff_LR[s] = shuffle_LR(hash_subsamp, hash_baseline,
                                    all_stim_idx)
    return shuff_LR