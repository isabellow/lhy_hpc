import numpy as np
import pandas as pd

import sys
sys.path.append("../utils/")
import load_matlab_data
import helpers

def load_wf_data(session_dir, ks_dir='kilosort4'):
    waveform_struct = load_matlab_data.loadmat_sbx(f"{session_dir}{ks_dir}waveformStruct.mat")
    waveform_struct = waveform_struct['wvStruct']
    print(waveform_struct.keys())
    return waveform_struct

def load_wf_multi_session(wf_file_path):
    waveform_struct = load_matlab_data.loadmat_sbx(wf_file_path)
    waveform_struct = waveform_struct['wvStruct']
    return waveform_struct

def sort_wf_by_channel(session_dir, waveform_struct, data_dir='raw_ephys_output/'):
    '''
    Sorts the waveform data by the channel order specified in the Intan header.

    Returns
    -------
    mean_waveforms_sorted_reordered : shape (n_cells, n_channels, n_timepoints)
    max_site_sorted : channel with the waveform peak
    max_idx : index for the channel with the waveform peak
    '''
    # intan header
    intan_info = load_matlab_data.loadmat_sbx(f"{session_dir}{data_dir}intan_info.mat")
    intan_info = intan_info['header']

    # get the channel IDs
    amp_ch_info_mat = intan_info['amplifier_channels']
    ch_sort_idx = np.asarray([])
    ch_names_unsorted = []
    for amp_ch in amp_ch_info_mat:
        for strg in amp_ch._fieldnames:
            if strg == 'custom_channel_name':
                name = amp_ch.__dict__[strg]
                ch_names_unsorted.append(name)
            elif strg == 'custom_order':
                idx = amp_ch.__dict__[strg]
                ch_sort_idx = np.append(ch_sort_idx, idx)
    ch_sort_idx = ch_sort_idx.astype(int)

    # get the waveform properties
    mean_waveforms = waveform_struct['waveFormsMean']
    max_site = waveform_struct['max_site']
    
    # sort by channel index
    sort_idx = np.argsort(ch_sort_idx)
    mean_waveforms_sorted = mean_waveforms[:, sort_idx]
    mean_waveforms_sorted_reordered = np.transpose(mean_waveforms_sorted, (2, 1, 0))
    max_site_sorted = [ch_names_unsorted[i] for i in max_site-1]

    # get the channel indices for the best channels
    ch_names = [ch_names_unsorted[i] for i in sort_idx]
    max_idx = np.asarray([])
    for ch in max_site_sorted:
        max_idx = np.append(max_idx, ch_names.index(ch))
    max_idx = max_idx.astype(int)

    return mean_waveforms_sorted_reordered, max_site_sorted, max_idx

def get_spike_times(session_dir, ks_dir='kilosort4', only_good=True):
    ''' get the spike times for each (good) unit '''
    phy_info = pd.read_csv(f"{session_dir}{ks_dir}cluster_group.tsv", sep='\t')
    cluster_id = phy_info['cluster_id'].values
    ks_label = phy_info['group'].values
    spike_t_raw = np.load(f'{session_dir}{ks_dir}spike_times.npy')
    spike_id_raw = np.load(f'{session_dir}{ks_dir}spike_clusters.npy')
    
    # only keep good units or include mua
    if only_good:
        good_idx = (ks_label == 'good').astype(bool)
        good_clusters = cluster_id[good_idx]
    else:
        good_clusters = cluster_id
    spike_good_idx = np.isin(spike_id_raw, good_clusters)
    spike_id = spike_id_raw[spike_good_idx]
    spike_t = spike_t_raw[spike_good_idx]
    
    # remove negative spike times
    spike_id = spike_id[spike_t >= 0]
    spike_t = spike_t[spike_t >= 0]
    
    return good_clusters, spike_id, spike_t