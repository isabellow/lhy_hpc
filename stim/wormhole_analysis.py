'''
Functions to detect antidromic responses in silicon probe data
'''
import numpy as np
from scipy import stats
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

def get_pre_stim_spikes(stim_times, spike_id, spike_t, sampling_rate=30000):
    '''
    Get the latency of the last spike before each stim for each cell

    **Note that this function will need to be modified to account for spikes
    between the stim and the hash if blanking procedure is refined**
    '''
    # data params
    good_clusters = np.unique(spike_id)
    n_cells = good_clusters.shape[0]
    n_stim = stim_times.shape[0]

    # remove spike times more than 1 second before first stim/after last stim
    start_idx = stim_times[0] - sampling_rate
    end_idx = stim_times[-1] + sampling_rate
    spike_id = spike_id[(spike_t >= start_idx) & (spike_t <= end_idx)]
    spike_t = spike_t[(spike_t >= start_idx) & (spike_t <= end_idx)]
    
    # re-align s.t. time 0 is 1 second pre first stim
    stim_times = stim_times - start_idx
    spike_t = spike_t - start_idx
    
    # get the closest spike time to each stim for every cell
    pre_stim_spikes = np.full((n_cells, n_stim), np.Inf)
    for i, cell_id in enumerate(good_clusters):
        prev_stim_t = 0
        all_sp_t = spike_t[spike_id==cell_id]
        for j, stim_t in enumerate(stim_times):
            if any(all_sp_t < stim_t):
                before_spikes = all_sp_t[(all_sp_t > prev_stim_t) & (all_sp_t < stim_t)]
                if any(before_spikes):
                    pre_stim_spikes[i, j] = np.min(stim_t - before_spikes)
            prev_stim_t = stim_t
    
    return pre_stim_spikes

def one_changepoint_vectors(X, max_col_trial=-1, show_plots=False):
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
    max_col_trial : max trial index for LR peak
        
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
    tau_hat = np.argmax(LR[:max_col_trial])

    # plot the distance and LR
    if show_plots:
        f, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].plot(D)
        ax[0].set_ylabel('Euclidean distance')
        ax[0].set_xlabel('trial')
        ax[1].plot(LR)
        ylim = ax[1].get_ylim()
        ax[1].vlines(tau_hat, 0, ylim[1],
                      colors='xkcd:scarlet',
                      linestyles='dashed', lw=1,
                      label=f'trial = {tau_hat}')
        ax[1].vlines(max_col_trial, 0, ylim[1],
                      colors='k',
                      linestyles='dotted', lw=1,
                      label=f'max col. lat.')
        ax[1].set_ylabel('LR statistic')
        ax[1].set_xlabel('change point')
        ax[1].legend(loc='upper left',
                        bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
    
    return LR, tau_hat 

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
                     sampling_rate=30000,
                     min_thresh=50e-3, # closest latency to stim for baseline trials (seconds)
                     collision_thresh=15e-3): # latency past which collisions are possible (seconds)             
    '''
    Get n_trials with spikes near to the stim, ordered from closest to furthest latency.
    Mix of possible collision trials (defined by collision_thresh) and non-collisions trials 
    (defined by min_thresh).
    '''
    # convert thresholds from seconds to samples
    min_thresh = min_thresh*sampling_rate
    collision_thresh = collision_thresh*sampling_rate

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
    n_long = all_stim_idx[finite_lat >= min_thresh].shape[0]
    if n_long >= n_trials-n_short:
        long_idx = np.random.choice(all_stim_idx[finite_lat >= min_thresh], 
                                    size=n_trials-n_short, replace=False)
    else:
        print('not enough long latency trials--selecting adjacent trials without spikes')
        long_idx = np.random.choice(all_stim_idx[finite_lat >= min_thresh], 
                                    size=n_long, replace=False)
        extras_idx = np.setdiff1d(np.setdiff1d(long_idx+1, long_idx), short_idx)
        extras_idx = extras_idx[extras_idx < all_stim_idx.shape[0]]
        if extras_idx.shape[0] < n_trials-n_short-n_long:
            extras_idx_1 = np.setdiff1d(np.setdiff1d(long_idx+1, long_idx), short_idx)
            extras_idx_2 = np.setdiff1d(np.setdiff1d(long_idx+2, long_idx), short_idx)
            extras_idx_1 = extras_idx_1[extras_idx_1 < all_stim_idx.shape[0]]
            extras_idx_2 = extras_idx_2[extras_idx_2 < all_stim_idx.shape[0]]
            extras_idx = np.append(extras_idx_1, extras_idx_2)
        if extras_idx.shape[0] < n_trials-n_short-n_long:
            print('not enough long latency or spike-free trials')
        else:
            long_idx_extras = np.random.choice(extras_idx,
                                                    size=n_trials-n_short-n_long, replace=False)
            long_idx = np.append(long_idx, long_idx_extras)
    
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
    shuff_LR, _ = one_changepoint(hash_dist)
    
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


def get_shuffle_by_cell(stim_hash, mask,
                        cell_trial_idx, n_short,
                         A_shank, B_shank, best_ch,
                         n_wf_ch=7, num_shuff=1000):
    '''
    Compute the shuffled change points for a given set of channels,
    defined as in subsample_channels.
    '''
    # subsample the hash around the relevant channel
    shuff_ch_idx = subsample_channels(best_ch, A_shank, B_shank, n_wf_ch=n_wf_ch)
    hash_subsamp = np.abs(stim_hash[:, shuff_ch_idx] * mask[None, shuff_ch_idx])

    # compute the shuffled change points
    n_trials = cell_trial_idx.shape[0]
    shuff_LR = np.full((num_shuff, n_trials-1), np.NaN)
    for s in range(num_shuff):
        shuff_trial_idx = np.random.permutation(cell_trial_idx)
        shuff_unwrapped = np.reshape(hash_subsamp[shuff_trial_idx], (n_trials, -1))
        shuff_LR[s], change_idx = one_changepoint_vectors(shuff_unwrapped,
                                                            max_col_trial=n_short)

    return shuff_LR