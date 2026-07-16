import numpy as np

import os 
import sys
sys.path.append("..//utils/")
import make_data_dict
import format_waveform_data, waveform_analysis, waveform_plots

'''
Loads a dictionary containing waveform information for good stim sessions
and adds new sessions to it as needed

Saves per session:
ephys_id : string
    folder name for ephys data with sorted cells
waveform_props : array, shape (3, n_cells)
    asymm, width, log_fr for each cell

Saves per bird:
all_waveform_props : array, shape (3, n_cells_all_sessions)
    asymm, width, log_fr for each cell across all sessions
'''

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"

''' Load the data dictionary of all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)
print(f'current birds with saved data: {bird_ids}')

modify_dict = input("add birds or sessions? (y/n)")
if modify_dict == 'y':
    data_dict = make_data_dict.modify_data_dict(root_dir, data_file)

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
    all_props = []
    for session_id in session_list:
        # # only calculate for new data
        # if 'waveform_props' in data_dict[bird][session_id].keys():
        #         continue

        # specify the file paths
        if 'ephys' in data_dict[bird][session_id]['preprocessed_data']:
            session_dir = f'{bird_dir}/{bird}_{session_id}/'
            for folder in sorted(os.listdir(session_dir)):
                if f'{bird}_{session_id}' in folder:
                    ephys_id = folder[-13:]
            
                    for file in sorted(os.listdir(f"{session_dir}{bird}_{ephys_id}")):
                        if 'kilosort4' in file:
                            data_dict[bird][session_id]['ephys_id'] = ephys_id
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
                            if len(all_props) == 0:
                                all_props = waveform_props
                            else:
                                all_props = np.column_stack([all_props, waveform_props])
                            data_dict[bird]['all_waveform_props'] = all_props

# Save inhib/exc clustering indices by session
# gather the waveform properties and n cells
all_waveform_props = []
sess_idx = 0
session_index = np.asarray([]).astype(int)
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if 'waveform_props' in data_dict[bird][session_id].keys():
            waveform_props = data_dict[bird][session_id]['waveform_props']
            n_cells = waveform_props.shape[1]
            if len(all_waveform_props) == 0:
                all_waveform_props = waveform_props
            else:
                all_waveform_props = np.column_stack((all_waveform_props, waveform_props))
            session_index = np.append(session_index, np.full(n_cells, sess_idx))
            sess_idx += 1

# cluster to get the cell type indices
asymm = all_waveform_props[0]
width = all_waveform_props[1]
log_fr = all_waveform_props[2]
exc_idx_all, inhib_idx_all = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)

# save by session
sess_idx = 0
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if 'waveform_props' in data_dict[bird][session_id].keys():
            data_dict[bird][session_id]['excitatory_idx'] = exc_idx_all[session_index==sess_idx]
            data_dict[bird][session_id]['inhibitory_idx'] = inhib_idx_all[session_index==sess_idx]
            sess_idx += 1

''' Save the updated dictionary '''
np.save(data_file, data_dict)