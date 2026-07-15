import numpy as np

import os 
import sys
sys.path.append("..//utils/")
sys.path.append("..//anatomy/")
import color_utils, make_data_dict
import get_probe_coords
import format_waveform_data, waveform_analysis, waveform_plots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


'''
For each session gather the following info for each cell
and add it to the dict.

- log firing rate (should already be there from plot_all_cells.py)
- estimated anatomical location
    - ML relative to DM/DL bound
    - DV using est depth and position on probe
    - AP est using hpc width and shank A/B


todo plots
- filter by antidromic resp/in vs out of nucleus
- compute average firing rate in smaller time windows to account for drift?

todo data munging
- process amb and rby data
- make sure all stim sessions have collision analysis
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
    # if bird == 'RBY94':
    #     continue
    for session_id in data_dict[bird]['all_sessions']:
        if 'cell_pos' in data_dict[bird][session_id].keys():
            cell_pos = data_dict[bird][session_id]['cell_pos']
            if len(all_cell_pos) == 0:
                all_cell_pos = cell_pos
            else:
                all_cell_pos = np.row_stack([all_cell_pos, cell_pos])

# get the dm/dl boundary points
min_ap = np.nanmin(all_cell_pos[:, 1])
max_ap = np.nanmax(all_cell_pos[:, 1])
ap_lims = np.asarray([min_ap, max_ap+100])
dmdl = get_probe_coords.define_dm_dl(ap_lims, n_pts=100)

''' Get the waveform properties '''
for i, bird in enumerate(bird_ids):
    if i == 0:
        all_waveform_props = data_dict[bird]['all_waveform_props']
    # elif bird == 'RBY94':
    #     continue
    else:
        waveform_props = data_dict[bird]['all_waveform_props']
        all_waveform_props = np.column_stack([all_waveform_props, waveform_props])
asymm = all_waveform_props[0]
width = all_waveform_props[1]
log_fr = all_waveform_props[2]

# cluster to get the excitatory index
exc_idx, _ = waveform_analysis.clu_waveforms_kmeans(width, asymm, log_fr)

# collect the bird/shank IDs and insertion coordinates
bird_shank_list = []
for i, bird in enumerate(bird_ids):
    # if bird == 'RBY94':
    #     continue
    insert_coords = data_dict[bird]['insert_coords']
    bird_shank_list.append(f'{bird}_A')
    bird_shank_list.append(f'{bird}_B')
    if i == 0:
        all_insert_coords = insert_coords[:, :2]
    else:
        all_insert_coords = np.row_stack((all_insert_coords, insert_coords[:, :2]))

''' Plot firing rate by cell location for all cells '''
fig, ax = waveform_plots.plot_fr_by_pos(all_cell_pos[exc_idx], log_fr[exc_idx], dmdl)
fig, ax = waveform_plots.add_bird_labels(fig, ax, bird_shank_list, all_insert_coords)
fig.savefig(f'{save_figs}fr_by_pos.png', 
                      dpi=600, bbox_inches='tight')

''' Plot all cells colored by inhibitory/excitatory '''
fig, ax = waveform_plots.plot_bool_by_pos(all_cell_pos, exc_idx, dmdl,
                                            labels=['excitatory', 'inhibitory'],
                                            colors=['xkcd:scarlet', 'xkcd:cobalt blue'])
fig, ax = waveform_plots.add_bird_labels(fig, ax, bird_shank_list, all_insert_coords)
fig.savefig(f'{save_figs}id_by_pos.png', 
                      dpi=600, bbox_inches='tight')



''' Make session list '''
pos_sessions = []
behavior_sessions = []
for i, bird in enumerate(bird_ids):
    for session_id in data_dict[bird]['all_sessions']:
        if 'cell_pos' in data_dict[bird][session_id].keys():
            pos_sessions.append(f'{bird}_{session_id}')
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        # if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        if ('ephys' in preprocessed_data):
            behavior_sessions.append(f'{bird}_{session_id}')
sessions_to_use = list(set(pos_sessions) & set (behavior_sessions))


''' Collect data across sessions for each bird '''
pos_dict = {}
all_AP = np.asarray([])
for i, bird in enumerate(bird_ids):
    # if (bird == 'LIM63') | (bird == 'RBY94'):
    #     continue
    pos_dict[bird] = {}
    for session_id in data_dict[bird]['all_sessions']:
        if f'{bird}_{session_id}' in sessions_to_use:
            # get the position of each cell (ML, est AP, DV)
            cell_pos = data_dict[bird][session_id]['cell_pos']
            
            # get the waveform props (asymm, width, log_fr)
            waveform_props = data_dict[bird][session_id]['waveform_props']

            # index cells by stim responsive channels
            stim_idx = data_dict[bird][session_id]['stim_resp_idx_ch']
            ch_pos = data_dict[bird][session_id]['channel_pos']
            stim_pos = ch_pos[stim_idx]
            n_cells = cell_pos.shape[0]
            cell_stim_idx = np.zeros(n_cells)
            for cell_idx, this_pos in enumerate(cell_pos):
                cell_stim_idx[cell_idx] = np.any(np.all(stim_pos == this_pos, axis=1))

            # excitatory/inhibitory indices
            exc_idx = data_dict[bird][session_id]['excitatory_idx']
            inhib_idx = data_dict[bird][session_id]['inhibitory_idx']

            # shank index
            shank_idx = data_dict[bird][session_id]['shank_A_idx']

            
            # store for this bird
            if 'cell_pos' in pos_dict[bird].keys():
                pos_dict[bird]['cell_pos'] = np.row_stack((pos_dict[bird]['cell_pos'], cell_pos))
                pos_dict[bird]['shank_A_idx'] = np.append(pos_dict[bird]['shank_A_idx'], shank_idx.astype(bool))
                pos_dict[bird]['waveform_props'] = np.column_stack((pos_dict[bird]['waveform_props'], waveform_props))     
                pos_dict[bird]['cell_stim_idx'] = np.append(pos_dict[bird]['cell_stim_idx'], cell_stim_idx.astype(bool))
                pos_dict[bird]['excitatory_idx'] = np.append(pos_dict[bird]['excitatory_idx'], exc_idx.astype(bool))
                pos_dict[bird]['inhibitory_idx'] = np.append(pos_dict[bird]['inhibitory_idx'], inhib_idx.astype(bool))
            else:
                pos_dict[bird]['cell_pos'] = cell_pos
                pos_dict[bird]['shank_A_idx'] = shank_idx.astype(bool)
                pos_dict[bird]['waveform_props'] = waveform_props
                pos_dict[bird]['cell_stim_idx'] = cell_stim_idx.astype(bool)
                pos_dict[bird]['excitatory_idx'] = exc_idx.astype(bool)
                pos_dict[bird]['inhibitory_idx'] = inhib_idx.astype(bool)
                
                # get the AP position of each shank
                cell_AP = np.unique(cell_pos[:, 1])
                shank_AP = np.zeros(2)
                shank_AP[0] = np.mean(cell_AP[:3])
                shank_AP[1] = np.mean(cell_AP[3:])
                if any(np.isnan(shank_AP)):
                    shank_AP[0] = 10001
                    shank_AP[1] = 10000

                pos_dict[bird]['AP'] = shank_AP
    
    # store the AP values and waveform props for all birds
    all_AP = np.append(all_AP, pos_dict[bird]['AP'])
    if i == 0:
        all_waveform_props = pos_dict[bird]['waveform_props']
    else:
        all_waveform_props = np.column_stack((all_waveform_props, pos_dict[bird]['waveform_props']))


''' Plot against depth/waveform width for each shank, sorted by AP location '''
# data params
n_birds = len(bird_ids)
n_shanks = n_birds*2
ap_sort = np.argsort(all_AP) 
ap_sort_idx = np.argsort(ap_sort)
log_fr = all_waveform_props[2]

excitatory_idx_all = np.asarray([]).astype(bool)
inhibitory_idx_all = np.asarray([]).astype(bool)
all_cell_fr = np.asarray([])
for bird in bird_ids:
    # if (bird == 'LIM63') | (bird == 'RBY94'):
    #     continue
    this_exc_idx = pos_dict[bird]['excitatory_idx']
    this_inhib_idx = pos_dict[bird]['inhibitory_idx']
    excitatory_idx_all = np.append(excitatory_idx_all, this_exc_idx)
    inhibitory_idx_all = np.append(inhibitory_idx_all, this_inhib_idx)
    all_cell_fr = np.append(all_cell_fr, pos_dict[bird]['waveform_props'][2])

exc_vmax = np.nanmax(all_cell_fr[excitatory_idx_all])
exc_vmin = np.nanmin(all_cell_fr[excitatory_idx_all])
inhb_vmax = np.nanmax(all_cell_fr[inhibitory_idx_all])
inhb_vmin = np.nanmin(all_cell_fr[inhibitory_idx_all])

# fig params
gs_kw = dict(hspace=0.1, wspace=0.3)
f, ax = plt.subplots(2, n_shanks, figsize=(14, 10),
                     sharey=True, gridspec_kw=gs_kw)
title_size = 14
axis_label = 12
tick_label = 9
ylims = [610, -10]
alpha_pts = 0.8
size_pts = 6

i = 0
for bird in bird_ids:
    # nucleus boundaries - A min, A max; B min, B max
    nucleus_dvs = data_dict[bird]['nucleus_dvs']
    A_nuc_lims = nucleus_dvs[0]
    B_nuc_lims = nucleus_dvs[1]
    if bird == 'RBY94':
        B_nuc_lims = np.asarray([np.nan, np.nan])

    # waveform properties
    wf_width = pos_dict[bird]['waveform_props'][1]
    cell_fr = pos_dict[bird]['waveform_props'][2]
    n_cells = wf_width.shape[0]

    # positions
    cell_pos = pos_dict[bird]['cell_pos']
    cell_dv = cell_pos[:, -1]
    A_idx = pos_dict[bird]['shank_A_idx']
    B_idx = ~A_idx

    # update positions (cells, nucleus bounds) s.t. min DV is zero
    min_dv_A = np.min(np.append(cell_dv[A_idx], A_nuc_lims))
    min_dv_B = np.min(np.append(cell_dv[B_idx], B_nuc_lims))
    if min_dv_A < 0:
        cell_dv[A_idx] = cell_dv[A_idx] - min_dv_A
        A_nuc_lims = A_nuc_lims - min_dv_A
    if min_dv_B < 0:
        cell_dv[B_idx] = cell_dv[B_idx] - min_dv_B
        B_nuc_lims = B_nuc_lims - min_dv_B

    # cluster indices
    exc_cells = pos_dict[bird]['excitatory_idx']
    inhb_cells = pos_dict[bird]['inhibitory_idx']

    # index by shank and cluster
    B_excite = B_idx & exc_cells
    B_inhib = B_idx & inhb_cells
    A_excite = A_idx & exc_cells
    A_inhib = A_idx & inhb_cells

    # subplot indices
    B_ax = ap_sort_idx[i]
    A_ax = ap_sort_idx[i+1]

    # jitter
    jit_dv = np.random.randn(n_cells)*2
    jit_w = np.random.randn(n_cells)*(2/600)

    # highlight the excitatory cells
    sc_exc = ax[0, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite],
                                 cell_dv[B_excite] + jit_dv[B_excite],
                                 c=cell_fr[B_excite], cmap='jet',
                                 vmin=exc_vmin, vmax=exc_vmax,
                                 s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
    ax[0, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
                        cell_dv[B_inhib] + jit_dv[B_inhib], 
                        c='xkcd:gray', 
                        s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[0, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
    ax[0, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
                        cell_dv[A_excite] + jit_dv[A_excite],
                        c=cell_fr[A_excite], cmap='jet', 
                        vmin=exc_vmin, vmax=exc_vmax,
                        s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
    ax[0, A_ax].scatter(wf_width[A_inhib] + jit_w[A_inhib],
                        cell_dv[A_inhib] + jit_dv[A_inhib], 
                        c='xkcd:gray', 
                        s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[0, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

    # highlight the inhibitory cells
    ax[1, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite], 
                        cell_dv[B_excite] + jit_dv[B_excite],
                        c='xkcd:gray', 
                        s=size_pts, lw=0, zorder=0, alpha=0.2)
    sc_inhb = ax[1, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
                                  cell_dv[B_inhib] + jit_dv[B_inhib],
                                  c=cell_fr[B_inhib], cmap='jet', 
                                  vmin=inhb_vmin, vmax=inhb_vmax,
                                  s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
    ax[1, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
    ax[1, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
                        cell_dv[A_excite] + jit_dv[A_excite],
                        c='xkcd:gray', 
                        s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[1, A_ax].scatter(wf_width[A_inhib] + jit_w[A_inhib],
                        cell_dv[A_inhib] + jit_dv[A_inhib], 
                        c=cell_fr[A_inhib], cmap='jet', 
                        vmin=inhb_vmin, vmax=inhb_vmax,
                        s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
    ax[1, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

    # add rough nucleus boundaries
    ax[0, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
                        colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    ax[0, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
                        colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    ax[1, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
                        colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    ax[1, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
                        colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    
    # titles
    ax[0, A_ax].set_title(f'{bird}', fontsize=axis_label)
    ax[0, B_ax].set_title(f'{bird}', fontsize=axis_label)
    if bird == 'RBY94':
        ax[0, B_ax].set_title(f'{bird}', fontsize=axis_label, fontstyle='italic')

    # format axes
    for j in range(2):
        for sh in [B_ax, A_ax]:
            ax[j, sh].set_xlim(0, 1)
            ax[j, sh].spines['right'].set_visible(False)
            ax[j, sh].spines['top'].set_visible(False)
            ax[j, sh].spines['bottom'].set_bounds(0, 1)
            ax[j, sh].spines['left'].set_bounds(600, 0)
            ax[j, sh].set_xticks([0, 0.5, 1])
            ax[j, sh].set_xticklabels(['0', '0.5', '1'])

    # update indices
    i += 2

# universal formatting
ax[0, 0].set_ylim(ylims) 
ax[0, 0].set_ylabel('excitatory cells\ndepth (um)', fontsize=axis_label)
ax[1, 0].set_ylabel('inhibitory cells\ndepth (um)', fontsize=axis_label)
f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.06)
f.suptitle(r"shanks sorted posterior $\rightarrow$ anterior", fontsize=axis_label, y=0.93)

# colorbars
max_fr = np.round(exc_vmax, 1)
min_fr = np.round(exc_vmin, 1)
cax = f.add_axes([0.93, 0.8, 0.008, 0.1])
cbar = f.colorbar(sc_exc, cax=cax)
cbar.set_label('log firing rate')
cbar.set_ticks([sc_exc.norm.vmin, sc_exc.norm.vmax])
cbar.set_ticklabels([rf'$10^{{{min_fr}}}$', rf'$10^{{{max_fr}}}$'])

max_fr = np.round(inhb_vmax, 1)
min_fr = np.round(inhb_vmin, 1)
cax = f.add_axes([0.93, 0.37, 0.008, 0.1])
cbar = f.colorbar(sc_inhb, cax=cax)
cbar.set_label('log firing rate')
cbar.set_ticks([sc_exc.norm.vmin, sc_exc.norm.vmax])
cbar.set_ticklabels([rf'$10^{{{min_fr}}}$', rf'$10^{{{max_fr}}}$'])

# add note for RBY94 subplots
# subplot positions + padding
box_axes = [
    ax[0, -2], ax[0, -1],
    ax[1, -2], ax[1, -1],
]
bboxes = [a.get_position() for a in box_axes]
x0 = min(bb.x0 for bb in bboxes)
y0 = min(bb.y0 for bb in bboxes)
x1 = max(bb.x1 for bb in bboxes)
y1 = max(bb.y1 for bb in bboxes)
pad_x = 0.008
pad_y = 0.03

# add rectangle in figure coordinates
rect = Rectangle(
    (x0 - pad_x, y0 - pad_y),
    (x1 - x0) + 2 * pad_x,
    (y1 - y0) + 2 * pad_y,
    transform=f.transFigure,
    fill=False,
    edgecolor='red',
    linewidth=1,
    clip_on=False,
)

f.add_artist(rect)

# add label
f.text(
    (x0 + x1) / 2,
    y1 + pad_y + 0.002,
    'no histology',
    color='red',
    ha='center',
    va='bottom',
    fontsize=axis_label
)

f.savefig(f'{save_figs}/width_by_depth_fr.png', dpi=400, bbox_inches='tight')
plt.show()