import numpy as np

import os
import sys
sys.path.append("..//utils/")
import make_data_dict
import neural_analysis

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

''' Save spikes aligned to behavior for each ephys + behavior session '''
for bird in bird_ids:
    print(f'\naligning spikes to behavior for {bird}')
    
    # collect sessions with pose tracking & ephys
    session_list = data_dict[bird]['all_sessions']
    behavior_sessions = []
    for session_id in session_list:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)

    # save the aligned spikes
    for session_id in behavior_sessions:
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        neural_analysis.align_spikes_behavior(session_dir)