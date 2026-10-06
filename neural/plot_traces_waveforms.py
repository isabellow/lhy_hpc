import numpy as np
from plot_cell_ephys import run_session, saved_unit_features

'''
For all good cells, plot example traces and waveforms, for a list of sessions
(or every session) from one bird. Waveforms are colored by the saved GMM
cluster, and the titles report the saved rate / width / spread.
'''
''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'ROS107'
session_ids = ['261004']   # list of session IDs, or None for every session of this bird
data_dict = np.load(data_file, allow_pickle=True).item()
# example_cells = np.asarray([2]) # example cell IDs (KS IDs differ across sessions,
#                                 # so only use this with a single session)
example_cells = None # all good cells

# set save dir for figures
save_figs_dir = f"../figures/basic_neural_analysis/"

if session_ids is None:
    session_ids = data_dict[bird]['all_sessions']

for session_id in session_ids:
    session_data = data_dict[bird][session_id]
    if 'ephys_id' not in session_data:      # no ephys for this session
        print(f'{bird}_{session_id}: no ephys, skipping')
        continue
    print(f'--- {bird}_{session_id} ---')
    save_figs_folder = f"{save_figs_dir}{bird}/{bird}_{session_id}/waveforms/"

    ''' Get the params for this session '''
    # set paths
    session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
    ephys_id = session_data['ephys_id']
    ks_id = session_data['ks_folder']
    ks_dir = f"{session_dir}{bird}_{ephys_id}/{ks_id}/"
    intan_folder = f"{session_dir}{bird}_{ephys_id}/"
    # si_folder = f"{intan_folder}kilosort4_bc/"

    ''' Plot the waveforms '''
    # GMM cluster colors + saved rate / width / spread for each unit
    saved = saved_unit_features(data_dict, bird, session_id)
    # results = run_session(intan_folder=intan_folder, si_folder=si_folder, ks_dir=ks_dir,
    #                         cluster_ids=example_cells, out_dir=save_figs_folder,
    #                         saved_features=saved)
    results = run_session(intan_folder=intan_folder, ks_dir=ks_dir,
                          cluster_ids=example_cells, out_dir=save_figs_folder,
                          saved_features=saved)
