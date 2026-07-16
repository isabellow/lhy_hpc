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
data_file = f"{root_dir}stim_session_data.npy"

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

# plot the kmeans clusters distinguishing in/out of nucleus
all_stim_idx = np.asarray([]).astype(bool)
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if 'waveform_props' in data_dict[bird][session_id].keys():
            waveform_props = data_dict[bird][session_id]['waveform_props']
            log_fr = waveform_props[2]
            n_cells = log_fr.shape[0]

            # to filter out interneurons & keep only in-nucleus cells
            if 'stim_resp_idx_ch' in data_dict[bird][session_id].keys():
                stim_idx = idx_cells_by_stim(data_dict, bird, session_id)
            else:
                print(f'no stim data for {bird}_{session_id}, excluding')
                stim_idx = np.zeros(n_cells).astype(bool)
            all_stim_idx = np.append(all_stim_idx, stim_idx)

log_fr = all_waveform_props[2]
excite_nucleus = clu1_idx & all_stim_idx
inhib_nucleus = clu2_idx & all_stim_idx
excite_out = clu1_idx & ~all_stim_idx
inhib_out = clu2_idx & ~all_stim_idx

fig, ax = waveform_plots.plot_wf_clusters(asymm, width, log_fr, excite_nucleus, inhib_nucleus)

# add non-nucleus cells
ax.scatter(asymm[excite_out],
           width[excite_out],
           log_fr[excite_out],
           c='xkcd:scarlet',
               alpha=0.1, lw=0, s=10, zorder=0)
ax.scatter(asymm[inhib_out],
           width[inhib_out],
           log_fr[inhib_out],
           c='xkcd:cobalt blue', 
           alpha=0.1, lw=0, s=10, zorder=0)

fig.savefig(f'{save_figs}waveform_props_kmeans_nuc_highlight.png', 
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
    bird_stim_exc_idx = np.asarray([]).astype(bool)
    n_cells_total = 0
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if 'waveform_props' in data_dict[bird][session_id].keys():
            waveform_props = data_dict[bird][session_id]['waveform_props']
            log_fr = waveform_props[2]
            n_cells = log_fr.shape[0]

            # to filter out interneurons & keep only in-nucleus cells
            exc_idx = data_dict[bird][session_id]['excitatory_idx']
            if 'stim_resp_idx_ch' in data_dict[bird][session_id].keys():
                stim_idx = idx_cells_by_stim(data_dict, bird, session_id)
            else:
                print(f'no stim data for {bird}_{session_id}, excluding')
                stim_idx = np.zeros(n_cells).astype(bool)
            print(f'{bird}_{session_id}: {np.sum(exc_idx)}/{n_cells} cells are excitatory')
            
            # save indices by bird
            bird_stim_exc_idx = np.append(bird_stim_exc_idx, exc_idx&stim_idx)
            n_cells_total += np.sum(stim_idx)

            # collect the normalized rates by session
            norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr[exc_idx&stim_idx])
            session_rates.append(norm_rates)
            session_bins.append(bin_vals)
    
    print(f'{bird}: {np.sum(bird_stim_exc_idx)}/{n_cells_total} projection nucleus cells are excitatory\n')
    
    # collect the normalized rates by bird
    log_fr_all = data_dict[bird]['all_waveform_props'][2]
    norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr_all[bird_stim_exc_idx])
    bird_rates.append(norm_rates)
    bird_bins.append(bin_vals)

bird_colors = color_utils.get_bird_colors_da(bird_ids)
fig, ax = waveform_plots.plot_cum_fr(session_rates, session_bins,
                                     bird_rates, bird_bins,
                                     colors=bird_colors
                                    )
fig.savefig(f'{save_figs}cumulative_frs_nuc_only.png', 
                      dpi=600, bbox_inches='tight')