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
For all sessions with stim data:
- cum fr for only channels w/ stim response
- cum fr for all collision-verified cells across all sessions
- stim response by channel plotted against position
- collision cells by position

TODO make a save_all_wf_data script and pull out relevant chunks from plot_all_cell_props
add ephys dir to the dict in this script
'''
''' Set file paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"
save_figs = f"../figures/antidromic_hpc_to_lhy/"


''' Load the dictionary of waveform data for all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)
print(f'current birds with saved data: {bird_ids}')

# check for stim data
stim_sessions = []
collision_sessions = []
for bird in bird_ids:
    if bird == 'RBY94':
        continue
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        # get the list of stim sessions
        if 'worm_ch_idx' in data_dict[bird][session_id].keys():
            stim_sessions.append(f'{bird}_{session_id}')

        # get the list of collision sessions
        if 'proj_cell_IDs' in data_dict[bird][session_id].keys():
            collision_sessions.append(f'{bird}_{session_id}')

# collect the stim response index
stim_resp_idx = np.asarray([])
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if f'{bird}_{session_id}' in stim_sessions:
            stim_idx = data_dict[bird][session_id]['worm_ch_idx']
            stim_resp_idx = np.append(stim_resp_idx, stim_idx)

''' Get the channel positions for all stim sessions '''
update_pos = input("update cell position data? (y/n)")
if update_pos == 'y':
    # ensure the probe info is up to date
    data_dict = get_probe_coords.get_anatomy_info(session_info_file, data_dict)

    # get the cell positions
    data_dict = get_probe_coords.save_cell_positions(data_dict, root_dir)

    # save the updated dictionary
    np.save(data_file, data_dict)

# collect all the channel locations
all_ch_pos = []
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if f'{bird}_{session_id}' in stim_sessions:
            ch_pos = data_dict[bird][session_id]['channel_pos']
            if len(all_ch_pos) == 0:
                all_ch_pos = ch_pos
            else:
                all_ch_pos = np.row_stack([all_ch_pos, ch_pos])

# collect the bird/shank IDs and insertion coordinates
bird_shank_list = []
for i, bird in enumerate(bird_ids):
    if bird == 'RBY94':
        continue
    insert_coords = data_dict[bird]['insert_coords']
    bird_shank_list.append(f'{bird}_A')
    bird_shank_list.append(f'{bird}_B')
    if i == 0:
        all_insert_coords = insert_coords[:, :2]
    else:
        all_insert_coords = np.row_stack((all_insert_coords, insert_coords[:, :2]))

# get the dm/dl boundary points
min_ap = np.min(all_ch_pos[:, 1])
max_ap = np.max(all_ch_pos[:, 1])
ap_lims = np.asarray([min_ap, max_ap+100])
dmdl = get_probe_coords.define_dm_dl(ap_lims, n_pts=100)

''' Plot all channels colored by stim response '''
fig, ax = waveform_plots.plot_bool_by_pos(all_ch_pos, stim_resp_idx.astype(bool), dmdl,
                                            labels=['stim response', 'no response'],
                                            colors=['xkcd:scarlet', 'xkcd:cobalt blue'])
# add the bird/shank labels
fig, ax = waveform_plots.add_bird_labels(fig, ax, bird_shank_list, all_insert_coords)
fig.savefig(f'{save_figs}stim_response_by_pos.png', 
                      dpi=600, bbox_inches='tight')


''' Plot all collision cells in brain space '''
all_proj_idx = np.asarray([])
all_cell_pos = []
for bird in bird_ids:
    session_list = data_dict[bird]['all_sessions']
    for session_id in session_list:
        if f'{bird}_{session_id}' in collision_sessions:
            # get the cell positions
            cell_pos = data_dict[bird][session_id]['cell_pos']
            if len(all_cell_pos) == 0:
                all_cell_pos = cell_pos
            else:
                all_cell_pos = np.row_stack([all_cell_pos, cell_pos])

            # get the indices for the collision-verified cells and convert to bool
            proj_idx = data_dict[bird][session_id]['proj_cell_idx']
            n_cells = cell_pos.shape[0]
            proj_bool = np.zeros(n_cells)
            proj_bool[proj_idx] = 1
            all_proj_idx = np.append(all_proj_idx, proj_bool)
all_proj_idx = all_proj_idx.astype(bool)

# plot the collision cells vs. other cells in those sessions
fig, ax = waveform_plots.plot_bool_by_pos(all_cell_pos, all_proj_idx, dmdl,
                                            labels=['projection cell', 'no collisions'],
                                            colors=['xkcd:scarlet', 'xkcd:light gray'])
fig, ax = waveform_plots.add_bird_labels(fig, ax, bird_shank_list, all_insert_coords)
fig.savefig(f'{save_figs}proj_cell_positions.png', 
                      dpi=600, bbox_inches='tight')
