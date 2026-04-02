import numpy as np

import os 
import sys
sys.path.append("..//utils/")
import color_utils, make_data_dict
import format_waveform_data, waveform_analysis, waveform_plots
import matplotlib.pyplot as plt

'''
Loads a dictionary containing waveform information for good stim sessions
and adds new sessions to it as needed.

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

''' Load/create the dictionary of waveform data for all good stim sessions '''
# load or create data dictionary and get bird list
bird_ids = []
if os.path.isfile(save_data):
    data_dict = np.load(save_data, allow_pickle=True).item()
    for bird in data_dict.keys():
        bird_ids.append(bird)
    print(f'current birds with saved data: {bird_ids}')
    modify_dict = input("modify data dictionary? (y/n)")
    if modify_dict == 'y':
        data_dict = make_data_dict.modify_data_dict(root_dir, save_data)
else:
    data_dict = make_data_dict.modify_data_dict(root_dir, save_data)

# update bird list
for bird in data_dict.keys():
    if bird in bird_ids:
        continue
    else:
        bird_ids.append(bird)


''' Load and organize the waveform properties '''
for bird in bird_ids:
    print(f'\ncollecting waveform data for {bird}')
    bird_dir = f"{root_dir}{bird}/"
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        # only calculate for new data
        if 'waveform_props' in data_dict[bird][session_id].keys():
                continue

        # specify the file paths
        if 'ephys' in data_dict[bird][session_id]['preprocessed_data']:
            session_dir = f'{bird_dir}/{bird}_{session_id}/'
            for folder in os.listdir(session_dir):
                if f'{bird}_{session_id}' in folder:
                    ephys_id = folder[-13:]
            
            for file in os.listdir(f"{session_dir}{bird}_{ephys_id}"):
                if 'kilosort4' in file:
                    ks_dir = f"{bird}_{ephys_id}/{file}/"
                    ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"

                    # load and format the waveform struct
                    waveform_struct = format_waveform_data.load_wf_data(session_dir, ks_dir=ks_dir)
                    wf_ids = waveform_struct['goodIDs']
                    mean_waveforms, wf_channels, _, ch_names = format_waveform_data.sort_wf_by_channel('', waveform_struct,
                                                                                                       data_dir=ephys_dir,
                                                                                                       return_ch_names=True)       
                    n_cells = mean_waveforms.shape[0]
                    wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])

                    # calculate the waveform properties
                    fr = waveform_struct['meanRate']
                    log_fr = np.log10(fr)
                    width = np.zeros(n_cells)
                    asymm = np.zeros(n_cells)
                    for wf_idx in range(n_cells):
                        best_ch = wf_ch_idx[wf_idx]
                        width[wf_idx] = waveform_analysis.calc_spike_width(mean_waveforms[wf_idx, best_ch])
                        asymm[wf_idx] = waveform_analysis.calc_amp_assym(mean_waveforms[wf_idx, best_ch])  

                    # save by session and overall
                    waveform_props = np.row_stack([asymm, width, log_fr])
                    data_dict[bird][session_id]['waveform_props'] = waveform_props
                    if 'all_waveform_props' in data_dict[bird].keys():
                        all_props = data_dict[bird]['all_waveform_props']
                        data_dict[bird]['all_waveform_props'] = np.column_stack([all_props, waveform_props])
                    else:
                        data_dict[bird]['all_waveform_props'] = waveform_props

# save the updated dictionary
np.save(save_data, data_dict)


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