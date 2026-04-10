import numpy as np

import os 
import sys
import format_chronic_stim
sys.path.append("..//utils/")
import make_data_dict

'''
Collect the antidromic response data for all good stim sessions
add this data to the session data dictionary
'''
''' Set file paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"
save_figs = f"../figures/antidromic_hpc_to_lhy/"

''' Params '''
# data params
sampling_rate = 30000
t_pre = 0.02 # seconds collected before stim starts
t_post = 0.03 # seconds collected after stim time
spk_thresh = 25 # minimum spike amplitude in uV

# stim response window
buffer = 6 # samples
start_t = 5e-3 # seconds
end_t = 15e-3 # seconds
start_idx = np.round((t_pre + start_t)*sampling_rate).astype(int) # samples
end_idx = np.round((t_pre + end_t)*sampling_rate).astype(int) # samples

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

''' Identify stim sessions '''
stim_sessions = []
ephys_dirs = []
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        # specify the file paths/session params
        if ('stim' in data_dict[bird][session_id]['preprocessed_data']) & ('ephys' in data_dict[bird][session_id]['preprocessed_data']):
            session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
            for folder in sorted(os.listdir(session_dir)):
                if f'{bird}_{session_id}' in folder:
                    ephys_id = folder[-13:]
                    for file in sorted(os.listdir(f"{session_dir}{bird}_{ephys_id}")):
                        if 'kilosort4' in file:
                            ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"
                            stim_sessions.append(f'{bird}_{session_id}')
                            ephys_dirs.append(ephys_dir)


'''
Load stim data and check for antidromic responses 

For now, collect across all stim params:
- channels that have a response
- collision-verified projection cells 
'''
for bird in bird_ids:
    print(f'\ncollecting stim response data for {bird}')
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if f'{bird}_{session_id}' in stim_sessions:
            ephys_dir = ephys_dirs[stim_sessions.index(f'{bird}_{session_id}')]

            # get the stim params
            stim_params = []
            for file in sorted(os.listdir(ephys_dir)):
                if ('neg' in file) & ('amplifier' in file):
                    file_parts = file.split(sep='_')
                    if 'neg' in file_parts[-1]:
                        stim_pol = file_parts[-1][:-4]
                        stim_params.append(stim_pol)
            
            for idx, stim_pol in enumerate(stim_params):
                print(f'loading stim data for {session_id}_{stim_pol}')

                # load and preprocess the stim responses
                raw_ephys = format_chronic_stim.load_stim(ephys_dir, stim_pol=stim_pol)
                ephys_data, ch_names = format_chronic_stim.sort_stim_by_channel(ephys_dir, raw_ephys)
                [n_channels, n_samples, n_stim] = ephys_data.shape

                # get the stim times
                stim_times = np.load(f'{ephys_dir}stim_t_{stim_pol}.npy')
                stim_times = np.squeeze(stim_times.astype(int))

                # compute the average stim response (hash)
                filt_data = format_chronic_stim.filter_stim_for_spikes(ephys_data)
                stim_hash = np.moveaxis(filt_data[:, start_idx:end_idx], -1, 0)
                avg_hash = np.mean(stim_hash, axis=0)

                # identify channels with an antidromic response (worm)
                if idx == 0:
                    worm_ch_idx = np.zeros(n_channels).astype(bool)
                for i in range(n_channels):
                    if any(np.abs(avg_hash[i]) >= spk_thresh):
                        worm_ch_idx[i] = True
            print(rf'{session_id} has {np.sum(worm_ch_idx)} total channels with stim responses')
            
            # check for collision dict files and collect projection cells
            # TODO add "keep_idx" to session spreadsheet to exclude false positives
            # ...or longer term TODO just improve the collision detection
            all_sig_cells = np.asarray([])
            all_sig_idx = np.asarray([])
            for file in sorted(os.listdir(ephys_dir[:-17])):
                if 'collision_props' in file:
                    collision_dict = np.load(f'{ephys_dir[:-17]}{file}', allow_pickle=True).item()
                    sig_cells = collision_dict['sig_cell_IDs']
                    sig_idx = collision_dict['sig_cell_idx']
                    all_sig_cells = np.append(all_sig_cells, sig_cells)
                    all_sig_idx = np.append(all_sig_idx, sig_idx)

            # remove double counted cells
            all_sig_cells, unique_idx = np.unique(all_sig_cells, return_index=True)
            all_sig_idx = all_sig_idx[unique_idx]
            print(f'{session_id} has {all_sig_cells.shape[0]} total cells with significant collisions (p <= 0.01)\n')

            # store the data
            data_dict[bird][session_id]['worm_ch_idx'] = worm_ch_idx
            if all_sig_cells.shape[0] > 0:
                data_dict[bird][session_id]['proj_cell_IDs'] = all_sig_cells.astype(int)
                data_dict[bird][session_id]['proj_cell_idx'] = all_sig_idx.astype(int)

# save the updated dictionary
np.save(data_file, data_dict)