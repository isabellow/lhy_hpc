import numpy as np
from plot_cell_ephys import make_cell_figure
from format_waveform_data import cluster_ids_for_session

'''
For all good cells, plot example traces and waveforms
'''
''' File paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Data params '''
bird = 'TRQ82'
session_id = '260806'
data_dict = np.load(data_file, allow_pickle=True).item()
example_cells = np.asarray([2])

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

# set example cell list
if example_cells is None:
    example_cells = cluster_ids_for_session(data_dict, bird, session_id, root_dir)

''' Plot the waveforms '''
for cell_id in example_cells:
    fig, data = make_cell_figure(intan_folder, ks_dir, cluster_id=cell_id, out_dir=save_figs_folder)