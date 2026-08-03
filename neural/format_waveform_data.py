import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import sys
sys.path.append("../utils/")
import load_matlab_data
import helpers

def load_wf_data(session_dir, ks_dir='kilosort4'):
    waveform_struct = load_matlab_data.loadmat_sbx(f"{session_dir}{ks_dir}waveformStruct.mat")
    waveform_struct = waveform_struct['wvStruct']
    return waveform_struct

def load_wf_multi_session(wf_file_path):
    waveform_struct = load_matlab_data.loadmat_sbx(wf_file_path)
    waveform_struct = waveform_struct['wvStruct']
    return waveform_struct

def sort_wf_by_channel(session_dir, waveform_struct,
                            data_dir='raw_ephys_output/',
                            return_ch_names=False):
    '''
    Sorts the waveform data by the channel order specified in the Intan header.

    Returns
    -------
    mean_waveforms_sorted_reordered : shape (n_cells, n_channels, n_timepoints)
    max_site_sorted : channel with the waveform peak
    max_idx : index for the channel with the waveform peak
    ch_names : list of strings, channel names associated with each index (optional)
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

    if return_ch_names:
        return mean_waveforms_sorted_reordered, max_site_sorted, max_idx, ch_names
    else:
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

def pop_normalize(aligned_spikes, dt=0.02, std_reg=1e-2,baseline_window=30):
    ''' Normalize activity for population analysis 
    aligned_spikes : ndarray, shape (n_cells, n_frames)
        number of spikes per behavior bin
    '''
    n_cells = aligned_spikes.shape[0]

    # instantaneous firing rate
    inst_firing_rate = aligned_spikes/dt

    # divide by the standard deviation (regularize by adding std_reg)
    st_dev_fr = stats.tstd(inst_firing_rate, axis=1) + std_reg
    norm_fr = inst_firing_rate.copy()
    for cell in range(n_cells):
        norm_fr[cell] /= st_dev_fr[cell]

    # get the baseline rate for each cell (running 30min avg activity)
    moving_avg_fr = np.zeros_like(inst_firing_rate)
    for cell in range(n_cells):
        moving_avg_fr[cell] = helpers.moving_avg(norm_fr[cell], window=baseline_window)
    norm_fr -= moving_avg_fr

    return norm_fr

def map_contacts_to_intan(probe, map_file_path):
    '''
    Given probe contact positions and a mapping file,
    get the index mapping each probe contact to an Intan channel

    probe_pos : ndarray, shape (n_contacts, 2)
        XY positions of the probe contacts in microns
    '''
    # load the channel map
    ch_map = pd.read_excel(map_file_path)
    
    # get the map channel positions
    xpos = ch_map["xpos"].to_numpy()
    ypos = ch_map["ypos"].to_numpy()
    map_pos = np.column_stack([xpos, ypos])
    if np.max(map_pos) < 1: # units are in mm
        map_pos = map_pos*1000
    n_channels = map_pos.shape[0]

    # get the probe contact positions
    probe_pos = probe.contact_positions
    
    # match using nearest neighbors
    tree = cKDTree(map_pos)
    distances, map_idx = tree.query(probe_pos, k=1)
    assert np.max(distances) < 0.1
    assert len(map_idx) == probe.get_contact_count()

    # map intan channel index to probe contact
    intan_ch_idx = ch_map["Intan Channel"].to_numpy()
    intan_ch_idx = intan_ch_idx[map_idx]
    contact_sort = np.argsort(intan_ch_idx)
    
    # get custom channel names
    ch_names_unsorted = []
    shank_idx_unsorted = np.zeros(n_channels)
    for j, i in enumerate(map_idx):
        shank_id = str(ch_map["Shank Letter"][i])
        shank_row = str(ch_map["Shank Row"][i])
        shank_col = str(ch_map["Shank Column"][i])
        if len(shank_row) == 1:
            ch_names_unsorted.append(f"{shank_id}-0{shank_row}-{shank_col}")
        else:
            ch_names_unsorted.append(f"{shank_id}-{shank_row}-{shank_col}")
        if map_pos[i, 0] > 100:
            shank_idx_unsorted[j] = 1

    # reorder
    ch_names = []
    for i in contact_sort:
        ch_names.append(ch_names_unsorted[i])
    shank_idx = shank_idx_unsorted[contact_sort]

    return contact_sort, ch_names, shank_idx