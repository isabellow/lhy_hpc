import numpy as np
from scipy.io import loadmat, savemat
from load_matlab_data import loadmat_v73

'''
Load SC data used in Chettih, Mackevicius et al, 2024
Reformat to match IL formatting conventions as needed
'''
def load_behavior_data(data_dir):
    seed_struct = loadmat_v73(f'{data_dir}annotatedSeeds.mat')['annotatedSeeds']
    count_data = seed_struct['countData']
    return(seed_struct, count_data)

def load_neural_data(data_dir, min_rate=0.02):
    aligned_data = loadmat_v73(f'{data_dir}alignedSpikesAndPosture.mat')['alignedData']
    waveform_struct = aligned_data['wvStruct']

    # SC params for good units
    max_contam = 0.2

    # cell type index
    type_idx = waveform_struct['idx']
    excitatory_idx = type_idx == 1
    inhibitory_idx = type_idx == 2

    # "good" unit IDs
    n_cells = type_idx.shape[0]
    all_clusters = np.arange(n_cells).astype(int)
    clean = waveform_struct['contam'] < max_contam
    high_fr = waveform_struct['meanRate'] > min_rate
    classified = (type_idx > 0) & (type_idx < 4)
    good_clusters = all_clusters[clean & high_fr & classified]

    # spikes per frame
    spike_fr = aligned_data['spks'].T # n_cells x n_frames

    # keep only good
    spike_fr = spike_fr[good_clusters]
    excitatory_idx = excitatory_idx[good_clusters].astype(bool)
    inhibitory_idx = inhibitory_idx[good_clusters].astype(bool)

    # get n_frames for the behavioral data, trim the spike matrix
    smooth_pts = aligned_data['smPts']
    n_frames = smooth_pts.shape[0]
    spike_fr = spike_fr[:, :n_frames]

    return spike_fr, excitatory_idx, inhibitory_idx