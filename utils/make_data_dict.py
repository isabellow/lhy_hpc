import numpy as np
import os 
import sys

def modify_data_dict(root_dir, save_data):
    '''
    Loads or creates a dictionary of sessions for a given set of birds.
    Adds new sessions/birds as needed.
    Stores whether each session has preprocessed stim, ephys, and behavior data.

    root_dir : string
        path to data folders
    save_data : string
        path to saved dict (or where to save the new dict)
    '''
    # load or create data dictionary and get bird list
    bird_ids = []
    if os.path.isfile(save_data):
        data_dict = np.load(save_data, allow_pickle=True).item()
        
        for bird in data_dict.keys():
            bird_ids.append(bird)
        print(f'current birds with stored data: {bird_ids}')
        
        add_birds = input("add more birds? (y/n)")
        if add_birds == 'y':
            n_birds = int(input("input number of new birds: "))
            for b in range(n_birds):
                bird = input(f"input bird {b+1} ID: ")
                bird_ids.append(bird)
    else:
        data_dict = {}

        n_birds = int(input("input number of birds: "))
        for b in range(n_birds):
            bird = input(f"input bird {b+1} ID: ")
            bird_ids.append(bird)

    for bird in bird_ids:
        if bird in data_dict.keys():
            continue
        else:
            data_dict[bird] = {}


    ''' Check for pre-processed data in each session '''
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
            if bird in session_folder:
                parts = session_folder.split('_')
                session_ID = parts[1]
                all_sessions.append(session_ID)
                if session_ID not in data_dict[bird].keys():
                    data_dict[bird][session_ID] = {}
                data_list = []
                for folder in os.listdir(f'{bird_dir}/{session_folder}'):
                    # check for a recording and waveform struct
                    if f'{bird}_{session_ID}' in folder:
                        ephys_sessions.append(folder[-13:])
                        for file in os.listdir(f'{bird_dir}/{session_folder}/{folder}'):
                            if 'kilosort4' in file: 
                                for f in os.listdir(f'{bird_dir}/{session_folder}/{folder}/{file}'):
                                    if 'waveformStruct' in f:
                                        waveform_sessions.append(session_ID)
                                        data_list.append('ephys')

                            # check for stim data
                            if 'raw_ephys_output' in file:
                                stim_sessions.append(session_ID)
                                data_list.append('stim')

                    # check for behavior data
                    if 'behavior_data' in folder:
                        for file in os.listdir(f'{bird_dir}/{session_folder}/{folder}'):
                            if 'annotatedSeeds' in file:
                                behavior_sessions.append(session_ID)
                                data_list.append('behavior')
                data_dict[bird][session_ID]['preprocessed_data'] = data_list
                                
        # check for fully preprocessed sessions
        complete_sessions = list(set(behavior_sessions) & set(waveform_sessions) & set(stim_sessions))
        print(f"{len(ephys_sessions)} have ephys ({len(waveform_sessions)} sorted and curated)")
        print(f"{len(stim_sessions)} have stim")
        print(f"{len(behavior_sessions)} have annotated behavior")
        print(f"{len(complete_sessions)} total sessions have complete data (ephys, stim, and behavior)")
        print(f"complete session IDs: {complete_sessions}")

        # save lists of all session IDs and session IDs with complete data
        data_dict[bird]['all_sessions'] = all_sessions
        data_dict[bird]['complete_sessions'] = complete_sessions

    # save the dict for future use
    np.save(save_data, data_dict)

    return data_dict