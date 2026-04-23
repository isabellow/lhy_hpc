import numpy as np

import os 
import sys
sys.path.append("..//utils/")
import make_data_dict

'''
Loads a dictionary containing neural data/session info for good stim sessions
and adds new sessions to it as needed

Saves per session:
pred_date : string
    prediction date (for file ID) for cells with posture tracking
todo other things?
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

''' Check for posture tracking and save file info '''
for bird in bird_ids:
    print(f'\ncollecting behavior tracking info for {bird}')
    bird_dir = f"{root_dir}{bird}/"
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        # specify the file paths
        if 'behavior' in data_dict[bird][session_id]['preprocessed_data']:
            session_dir = f'{bird_dir}/{bird}_{session_id}/behavior_data/'
            flag = 0
            for file in sorted(os.listdir(f"{session_dir}")):
                if 'posture_2stage_face.npy' in file:
                    if flag == 1:
                        print(f'Warning! 2 pose tracking files found for {bird}_{session_id}')
                    pred_date = file[:6]
                    data_dict[bird][session_id]['pred_date'] = pred_date
                    flag = 1

# save the updated dictionary
np.save(data_file, data_dict)