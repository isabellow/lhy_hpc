import numpy as np

import os 
import sys
sys.path.append("..//utils/")
sys.path.append("..//anatomy/")
import color_utils, make_data_dict
import get_probe_coords
import format_waveform_data, waveform_analysis, waveform_plots
import matplotlib.pyplot as plt

'''
For each session gather the following info for each cell
and add it to the dict.

- log firing rate (should already be there from plot_all_cells.py)
- estimated anatomical location
    - ML relative to DM/DL bound
    - DV using est depth and position on probe
    - AP est using hpc width and shank A/B


todo
- label each bird on the plot
- filter by antidromic resp/in vs out of nucleus
- process amb and rby data
- plot interneuron/excitatory identity vs anatomical position
'''

''' Set file paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"
save_figs = f"../figures/basic_neural_analysis/"

''' Load/create the dictionary of waveform data for all good stim sessions '''
# load or create data dictionary and get bird list
bird_ids = []
if os.path.isfile(data_file):
    data_dict = np.load(data_file, allow_pickle=True).item()
    for bird in data_dict.keys():
        bird_ids.append(bird)
    print(f'current birds with saved data: {bird_ids}')
    modify_dict = input("modify data dictionary? (y/n)")
    if modify_dict == 'y':
        data_dict = make_data_dict.modify_data_dict(root_dir, data_file)
else:
    data_dict = make_data_dict.modify_data_dict(root_dir, data_file)

# update bird list
for bird in data_dict.keys():
    if bird in bird_ids:
        continue
    else:
        bird_ids.append(bird)

''' Get the cell positions for each ephys session '''
update_pos = input("update cell position data? (y/n)")
if update_pos == 'y':
    # ensure the probe info is up to date
    data_dict = get_probe_coords.get_anatomy_info(session_info_file, data_dict)

    # get the cell positions
    data_dict = get_probe_coords.save_cell_positions(data_dict, root_dir)

    # save the updated dictionary
    np.save(data_file, data_dict)

# collect all the cell positions
all_cell_pos = []
for bird in bird_ids:
    for session_id in data_dict[bird]['all_sessions']:
        if 'cell_pos' in data_dict[bird][session_id].keys():
            cell_pos = data_dict[bird][session_id]['cell_pos']
            if all_cell_pos == []:
                all_cell_pos = cell_pos
            else:
                all_cell_pos = np.row_stack([all_cell_pos, cell_pos])

# get the dm/dl boundary points
min_ap = np.min(all_cell_pos[:, 1])
max_ap = np.max(all_cell_pos[:, 1])
ap_lims = np.asarray([min_ap, max_ap+100])
dmdl = get_probe_coords.define_dm_dl(ap_lims, n_pts=100)

''' Get the waveform properties '''
for i, bird in enumerate(bird_ids):
    if i == 0:
        all_waveform_props = data_dict[bird]['all_waveform_props']
    else:
        waveform_props = data_dict[bird]['all_waveform_props']
        all_waveform_props = np.column_stack([all_waveform_props, waveform_props])
asymm = all_waveform_props[0]
width = all_waveform_props[1]
log_fr = all_waveform_props[2]

# cluster to get the excitatory index
exc_idx, _ = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)

''' Plot firing rate by cell location for all cells '''
fig, ax = waveform_plots.plot_fr_by_pos(all_cell_pos[exc_idx], log_fr[exc_idx], dmdl)
fig.savefig(f'{save_figs}fr_by_pos.png', 
                      dpi=600, bbox_inches='tight')

''' Plot all cells colored by inhibitory/excitatory '''
fig, ax = waveform_plots.plot_bool_by_pos(all_cell_pos, exc_idx, dmdl,
                                            labels=['excitatory', 'inhibitory'],
                                            colors=['xkcd:scarlet', 'xkcd:cobalt blue'])
fig.savefig(f'{save_figs}id_by_pos.png', 
                      dpi=600, bbox_inches='tight')