''' Given a list of birds, run analyses on all sessions, get summary data '''
import numpy as np

import os 
import sys
sys.path.append(".//neural/")
import format_waveform_data, waveform_analysis, waveform_plots

import matplotlib.pyplot as plt

root_dir = "Z:/Isabel/data/hpc_implants/"
save_dir = f"./figures/basic_neural_analysis/"

''' Get bird list and create dict for data storage '''
n_birds = int(input("input number of birds: "))
bird_ids = []
for b in range(n_birds):
    bird = input(f"input bird {b} ID: ")
    bird_ids.append(bird)
data_dict = {}
for bird in bird_ids:
    data_dict[bird] = {}
# TODO: load existing data dict and check for/add new data

''' Get session lists '''
for bird in bird_ids:
    bird_dir = f"{root_dir}{bird}/"
    session_dirs = sorted(os.listdir(bird_dir))
    print(f"\n{bird} has {len(session_dirs)} total sessions")
    
    all_sessions = []
    behavior_sessions = []
    ephys_sessions = []
    waveform_sessions = []
    stim_sessions = []
    for session_folder in session_dirs:
        session_ID = session_folder[-7:-1]
        all_sessions.append(session_ID)
        for folder in os.listdir(session_folder):
            # check for a recording and waveform struct
            if f'{bird}_{session_ID}' in folder:
                ephys_sessions.append(folder[-13:-1])
                for file in os.listdir(folder):
                    if 'kilosort4' in file: 
                        if 'waveformStruct' in os.listdir(file):
                            waveform_sessions.append(session_ID)
                    
                    # check for stim data
                    if 'raw_ephys_output' in file:
                        stim_sessions.append(session_ID)

            # check for behavior data
            if 'behavior_data' in folder:
                for file in os.listdir(folder):
                    if 'annotatedSeeds' in file:
                        behavior_sessions.append(session_ID)

    # check for fully preprocessed sessions
    complete_sessions = list(set(behavior_sessions) & set(waveform_sessions) & set(stim_sessions))
    print(f"{len(ephys_sessions)} have ephys ({len(waveform_sessions)} sorated and curated)")
    print(f"{len(stim_sessions)} have stim")
    print(f"{len(behavior_sessions)} have annotated behavior")
    print(f"{len(complete_sessions)} total sessions have complete data and are ready for analysis")
    print(f"complete session IDs: {complete_sessions}")
    data_dict[bird]['complete_sessions'] = complete_sessions
    data_dict[bird]['ephys_session_ids'] = ephys_sessions

    # choose sessions to analyze
    print("all complete sessions will be included in the waveform & cumulative firing rates plots")
    n_sessions = int(input("input number of sessions to analyze/plot further: "))
    full_anlysis = []
    for s in range(n_sessions):
        session = input(f"input session {s} ID: ")
        full_anlysis.append(session)
    data_dict[bird]['sessions_to_analyze'] = full_anlysis


''' TODO? Save complete sessions locally to speed up analysis? '''


''' Get waveform properties and make plots '''
for bird in bird_ids:
    ephys_ids = data_dict[bird]['ephys_session_ids']
    for session_id in data_dict[bird]['complete_sessions']:
        # specify file paths
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        for file in os.listdir(session_dir):
            if 'kilosort4' in file:
                ks_dir = file
        for eid in ephys_ids:
            if session_id in eid:
                ephys_id = eid
        ephys_dir = f"{session_dir}{ephys_id}/raw_ephys_output/"

        # probe channels to ignore
        # TODO load from session csv

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
        data_dict[bird][session_id] = {}
        data_dict[bird][session_id]['waveform_props'] = waveform_props
        if 'all_waveform_props' in data_dict[bird].keys():
            all_props = data_dict[bird]['all_waveform_props']
            data_dict[bird]['all_waveform_props'] = np.column_stack([all_props, waveform_props])
        else:
            data_dict[bird]['all_waveform_props'] = waveform_props

# collect all the waveform properties
# TODO Gaussian mixture model!
for i, bird in enumerate(bird_ids):
    if i == 0:
        all_waveform_props = data_dict[bird]['all_waveform_props']
    else:
        waveform_props = data_dict[bird]['all_waveform_props']
        all_waveform_props = np.column_stack([all_waveform_props, waveform_props])

# get the cluster indices - swap as needed so interneurons are clu2
asymm = all_waveform_props[0]
width = all_waveform_props[1]
log_fr = all_waveform_props[2]
clu1_idx, clu2_idx = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)
if np.sum(clu1_idx) < np.sum(clu2_idx):
    clu1_idx_temp = clu2_idx.copy()
    clu2_idx = clu1_idx
    clu1_idx = clu1_idx_temp
n_excite = np.sum(clu1_idx)
n_cells = all_waveform_props.shape[1]
print(f'\n{n_excite}/{n_cells} cells are putative excitatory neurons ({np.round((n_excite/n_cells)*100, 1)}%)')

# plot the waveform property clusters
fig, ax = waveform_plots.plot_wf_clusters(asymm, width, log_fr, clu1_idx, clu2_idx)
fig.savefig(f'{save_dir}waveform_props.png', 
                      dpi=600, bbox_inches='tight')

# plot the cumulative firing rates by session and by bird
session_rates = []
session_bins = []
bird_rates = []
bird_bins = []
for bird in bird_ids:
    for session_id in data_dict[bird]['complete_sessions']:
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]
        norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr)
        session_rates.append(norm_rates)
        session_bins.append(bin_vals)
    waveform_props = data_dict[bird]['all_waveform_props']
    log_fr = waveform_props[2]
    norm_rates, bin_vals = waveform_analysis.calc_cum_rates(log_fr)
    bird_rates.append(norm_rates)
    bird_bins.append(bin_vals)

# TODO bird_colors
fig, ax = waveform_plots.plot_cum_fr(session_rates, session_bins,
                                        bird_rates, bird_bins,
                                        bird_colors
                                    )
fig.savefig(f'{save_dir}cumulative_frs.png', 
                      dpi=600, bbox_inches='tight')

'''
for each bird...
- collect session IDs : print N total sessions
- check for ephys : print N session w/ recordings
- check for preprocessing:
-- stim data : print N sessions w/ stim
-- waveform struct : print N sessions w/ sorted cells
-- annotated seeds : print N session w/ annotated behavior
- print N session w/ all preprocessing completed
--> print session list

user input: analyze sessions? (input list of IDs)

collect waveform properties
'''