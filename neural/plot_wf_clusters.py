import numpy as np
import matplotlib.pyplot as plt

import os 
import sys
sys.path.append("..//utils/")
import color_utils
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim
import waveform_plots, waveform_analysis

'''
Simplified version of plot_all_cell_props that just clusters and plots those
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Load the dictionary of waveform data for all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

'''Cluster and plot the waveform properties across all birds '''
# collect all the waveform properties
for i, bird in enumerate(bird_ids):
    if i == 0:
        all_waveform_props = data_dict[bird]['all_waveform_props']
    else:
        waveform_props = data_dict[bird]['all_waveform_props']
        all_waveform_props = np.column_stack([all_waveform_props, waveform_props])

asymm = all_waveform_props[0]
width = all_waveform_props[1]
log_fr = all_waveform_props[2]
n_cells = all_waveform_props.shape[1]

# get rid of miscalculated widths for now TODO
proper_width = width < 0.7
asymm_filt = asymm[proper_width]
width_filt = width[proper_width]
log_fr_filt = log_fr[proper_width]
n_cells_filt = log_fr_filt.shape[0]

# use k-means to assign cluster indices
clu1_idx, clu2_idx = waveform_analysis.clu_waveforms_kmeans(width_filt, asymm_filt, log_fr_filt)
n_excite = np.sum(clu1_idx)
print(f'\nkmeans clustering: {n_excite}/{n_cells_filt} cells are putative excitatory neurons ({np.round((n_excite/n_cells_filt)*100, 1)}%)')
fig, ax = waveform_plots.plot_wf_clusters(asymm_filt, width_filt, log_fr_filt, clu1_idx, clu2_idx)
fig.savefig(f'{save_figs}waveform_props_kmeans.png', 
                      dpi=600, bbox_inches='tight')