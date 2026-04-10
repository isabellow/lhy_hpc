import numpy as np

import os 
import sys
sys.path.append("..//utils/")
import color_utils, make_data_dict
import format_waveform_data, waveform_analysis, waveform_plots
import matplotlib.pyplot as plt

'''
Loads a dictionary containing waveform information for good stim sessions.

Clusters and plots the waveform properties across all sessions as in Payne et al. 2021
Computes and plots the cumulative firing rates across neurons by session and bird

TODO
-----
- other clustering algorithms?
'''

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs = f"../figures/basic_neural_analysis/"
save_data = f"{root_dir}stim_session_data.npy"

''' Load the dictionary of waveform data for all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)
print(f'current birds with saved data: {bird_ids}')

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

# use k-means to assign cluster indices
clu1_idx, clu2_idx = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)
n_excite = np.sum(clu1_idx)
print(f'\nkmeans clustering: {n_excite}/{n_cells} cells are putative excitatory neurons ({np.round((n_excite/n_cells)*100, 1)}%)')
fig, ax = waveform_plots.plot_wf_clusters(asymm, width, log_fr, clu1_idx, clu2_idx)
fig.savefig(f'{save_figs}waveform_props_kmeans.png', 
                      dpi=600, bbox_inches='tight')

# use a GMM to assign cluster indices
prob_thresh = 0.8
clu1_idx, clu2_idx, clu_none_idx = waveform_analysis.clu_waveforms_gmm(width, asymm, log_fr, prob_thresh=prob_thresh)
n_excite = np.sum(clu1_idx)
print(f'GMM clustering: {n_excite}/{n_cells} cells are putative excitatory neurons ({np.round((n_excite/n_cells)*100, 1)}%)')
fig, ax = waveform_plots.plot_wf_clusters(asymm, width, log_fr, clu1_idx, clu2_idx)
# if excluding low probability cells, add those to the plot
if prob_thresh > 0.5:
    ax.scatter(asymm[clu_none_idx],
               width[clu_none_idx],
               log_fr[clu_none_idx],
               c='k', 
               alpha=0.1, lw=0, s=10, zorder=0)
fig.savefig(f'{save_figs}waveform_props_gmm.png', 
                      dpi=600, bbox_inches='tight')

''' Plot the cumulative firing rates by session and bird '''
session_rates = []
session_bins = []
bird_rates = []
bird_bins = []
session_idx_s = 0
bird_idx_s = 0
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if 'waveform_props' in data_dict[bird][session_id].keys():
            waveform_props = data_dict[bird][session_id]['waveform_props']
            log_fr = waveform_props[2]

            # filter out interneurons
            n_cells = log_fr.shape[0]
            session_idx_e = session_idx_s + n_cells
            session_clu1_idx = clu1_idx[session_idx_s:session_idx_e]
            session_idx_s = session_idx_e
            print(f'{bird}_{session_id}: {np.sum(session_clu1_idx)}/{n_cells} cells are excitatory')

            # collect the normalized rates by session
            norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr[session_clu1_idx])
            session_rates.append(norm_rates)
            session_bins.append(bin_vals)
    
    waveform_props = data_dict[bird]['all_waveform_props']
    log_fr = waveform_props[2]

    # filter out interneurons
    n_cells = log_fr.shape[0]
    bird_idx_e = bird_idx_s + n_cells
    bird_clu1_idx = clu1_idx[bird_idx_s:bird_idx_e]
    bird_idx_s = bird_idx_e
    print(f'{bird}: {np.sum(bird_clu1_idx)}/{n_cells} cells are excitatory\n')
    
    # collect the normalized rates by bird
    norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr[bird_clu1_idx])
    bird_rates.append(norm_rates)
    bird_bins.append(bin_vals)

bird_colors = color_utils.get_bird_colors_da(bird_ids)
fig, ax = waveform_plots.plot_cum_fr(session_rates, session_bins,
                                     bird_rates, bird_bins,
                                     colors=bird_colors
                                    )
fig.savefig(f'{save_figs}cumulative_frs.png', 
                      dpi=600, bbox_inches='tight')