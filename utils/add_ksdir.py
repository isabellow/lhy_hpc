import numpy as np
import os

'''
Patch function to add ks_dir to the hpc data dict
'''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"

data_dict = np.load(data_file, allow_pickle=True).item()

existing_birds = list(data_dict.keys())
print(f"current birds with saved data: {existing_birds}")

bird_ids = list(data_dict.keys())

# check each bird's session folders for preprocessed data
for bird in bird_ids:
    bird_dir = f"{root_dir}{bird}/"
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        # get file paths for preprocessed data
        session_folder = f"{root_dir}{bird}/{bird}_{session_id}/"
        for folder in os.listdir(session_folder):
            if f'{bird}_{session_id}' in folder:
                # ephys data
                ephys_id = folder[-13:]
                for file in os.listdir(f'{session_folder}/{folder}'):
                    if 'kilosort4' in file:
                        for f in os.listdir(f'{session_folder}/{folder}/{file}'):
                            if 'waveformStruct' in f:
                                data_dict[bird][session_id]['ephys_id'] = ephys_id
                                data_dict[bird][session_id]['ks_folder'] = file

np.save(data_file, data_dict)