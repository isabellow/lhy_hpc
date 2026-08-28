import numpy as np
from plot_cell_ephys import run_session

'''
For all good cells, plot example traces and waveforms
'''
''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'LMN88'
session_id = '260720'
data_dict = np.load(data_file, allow_pickle=True).item()
# example_cells = np.asarray([2]) # example cell IDs
example_cells = None # all good cells

# set save dir for figures
save_figs_dir = f"../figures/basic_neural_analysis/"
save_figs_folder = f"{save_figs_dir}{bird}/{bird}_{session_id}/waveforms/"

''' Get the params for this session '''
# set paths
session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
ephys_id = data_dict[bird][session_id]['ephys_id']
ks_id = data_dict[bird][session_id]['ks_folder']
ks_dir = f"{session_dir}{bird}_{ephys_id}/{ks_id}/"
intan_folder = f"{session_dir}{bird}_{ephys_id}/"
# si_folder = f"{intan_folder}kilosort4_bc/"

''' Plot the waveforms '''
# results = run_session(intan_folder=intan_folder, si_folder=si_folder, ks_dir=ks_dir, 
#                         cluster_ids=example_cells, out_dir=save_figs_folder)
results = run_session(intan_folder=intan_folder, ks_dir=ks_dir, 
                        cluster_ids=example_cells, out_dir=save_figs_folder)