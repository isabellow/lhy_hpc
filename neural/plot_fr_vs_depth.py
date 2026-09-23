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
Simpler version of plot_fr_vs_cell_pos.py

Just plot cells against their depth/waveform width
colored by firing rate.

TODO: optionally tag cells as in/out of LHy based on their
estimated position in the brain and known LHy boundaries.
--> would be very cool to do this properly from SC/MA anatomy data
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9
ylims = [6510, 4490]
alpha_pts = 0.8
size_pts = 6

''' Load the data dictionary '''
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = list(data_dict.keys())

''' Define/create the save folder '''
save_dir = f"{save_figs_dir}/"
os.makedirs(save_dir, exist_ok=True)

''' Collect data across sessions for each bird '''
pos_dict = {}
all_cell_fr = np.asarray([])
for bird in bird_ids:
    pos_dict[bird] = {}
    for session_id in data_dict[bird]['all_sessions']:
        # skip sessions without ephys
        session_data = data_dict[bird][session_id]
        if 'ephys' not in session_data['preprocessed_data']:
            continue

        # get the position of each cell (ML, est AP, DV)
        cell_pos = session_data['cell_pos']

        # get the waveform props (asymm, width, log_fr)
        waveform_props = session_data['waveform_props']

        # shank index -- int array, 0..n_shanks-1 (arbitrary n_shanks)
        shank_idx = session_data['shank_idx']

        # save everything
        fields = {
            'cell_pos': cell_pos,
            'waveform_props': waveform_props,
            'shank_idx': shank_idx.astype(int),
        }

        # store for this bird, concatenating across sessions
        if 'cell_pos' in pos_dict[bird]:
            pos_dict[bird]['cell_pos'] = np.row_stack((pos_dict[bird]['cell_pos'], fields['cell_pos']))
            pos_dict[bird]['waveform_props'] = np.column_stack((pos_dict[bird]['waveform_props'],
                                                                 fields['waveform_props']))
            pos_dict[bird]['shank_idx'] = np.append(pos_dict[bird]['shank_idx'], fields['shank_idx'])
        else:
            pos_dict[bird].update(fields)

        # save firing rates acorss all birds
        all_cell_fr = np.append(all_cell_fr, waveform_props[2])

    # skip birds with no qualifying sessions (no cell_pos + behavior/ephys overlap),
    if 'cell_pos' not in pos_dict[bird]:
        print(f'no position + behavior sessions found for {bird}, skipping')
        del pos_dict[bird]
        continue

    # per-shank mean AP position for this bird (generalized to n shanks)
    n_shanks_bird = int(pos_dict[bird]['shank_idx'].max()) + 1
    cell_pos = pos_dict[bird]['cell_pos']
    shank_ap = np.full(n_shanks_bird, np.nan)
    for sh in range(n_shanks_bird):
        sh_mask = pos_dict[bird]['shank_idx'] == sh
        if np.any(sh_mask):
            shank_ap[sh] = np.mean(cell_pos[sh_mask, 1])
    pos_dict[bird]['shank_AP'] = shank_ap
    pos_dict[bird]['n_shanks'] = n_shanks_bird

# only birds with usable data remain
bird_ids = list(pos_dict.keys())


''' Build the (bird, shank) column ordering, sorted posterior -> anterior, across all birds '''
bird_shank_list = []  # one (bird, shank) entry per subplot column
ap_list = []
for bird in bird_ids:
    for sh in range(pos_dict[bird]['n_shanks']):
        bird_shank_list.append((bird, sh))
        ap_list.append(pos_dict[bird]['shank_AP'][sh])
ap_list = np.asarray(ap_list)
col_rank = np.argsort(np.argsort(ap_list))  # column index for each (bird, shank) entry; NaNs sort last
col_idx_for = {bird_shank_list[i]: col_rank[i] for i in range(len(bird_shank_list))}
n_cols = len(bird_shank_list)


''' Plot firing rate by width/depth '''
gs_kw = dict(hspace=0.1, wspace=0.3)
f, ax = plt.subplots(1, n_cols, figsize=(0.85*n_cols, 5),
                     sharey=True, gridspec_kw=gs_kw)
vmax = np.nanmax(all_cell_fr)
vmin = np.nanmin(all_cell_fr)
no_hist_cols = [] # flag shanks if AP is NaN

for bird in bird_ids:
    # data params
    n_shanks_bird = pos_dict[bird]['n_shanks']
    cell_pos = pos_dict[bird]['cell_pos']
    cell_dv = cell_pos[:, -1]
    shank_idx = pos_dict[bird]['shank_idx']
    wf_width = pos_dict[bird]['waveform_props'][1]
    cell_fr = pos_dict[bird]['waveform_props'][2]
    n_cells = wf_width.shape[0]

    # plotting params
    jit_dv = np.random.randn(n_cells) * 2
    jit_w = np.random.randn(n_cells) * (2 / 600)

    for sh in range(n_shanks_bird):
        col = col_idx_for[(bird, sh)]
        sh_mask = shank_idx == sh

        if np.isnan(pos_dict[bird]['shank_AP'][sh]):
            no_hist_cols.append(col)

        # shift depths so the min DV on this shank is zero
        min_dv = np.min(cell_dv[sh_mask])
        if min_dv < 0:
            cell_dv[sh_mask] = cell_dv[sh_mask] - min_dv

        # plot the firing rates by cell DV/WF width
        sc = ax[col].scatter(wf_width[sh_mask] + jit_w[sh_mask],
                              cell_dv[sh_mask] + jit_dv[sh_mask],
                              c=cell_fr[sh_mask], cmap='jet', alpha=alpha_pts,
                              vmin=vmin, vmax=vmax,
                              s=size_pts, lw=0, zorder=1)
        ax[col].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
        ax[col].set_title(f'{bird}', fontsize=axis_label)

# universal formatting
for col in range(n_cols):
    ax[col].set_xlim(0, 1)
    ax[col].spines['right'].set_visible(False)
    ax[col].spines['top'].set_visible(False)
    # ax[col].spines['bottom'].set_bounds(0, 1)
    ax[col].spines['left'].set_bounds(ylims[0]-10, ylims[1]+10)
    ax[col].set_xticks([0, 0.5, 1])
    ax[col].set_xticklabels(['0', '0.5', '1'])
ax[0].set_ylim(ylims)
ax[0].set_ylabel('depth from surface (um)', fontsize=axis_label)
f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.02)
f.suptitle(r"shanks sorted posterior $\rightarrow$ anterior", fontsize=axis_label, y=0.98)

# colorbar
max_fr = np.round(vmax, 1)
min_fr = np.round(vmin, 1)
cax = f.add_axes([0.93, 0.8, 0.008, 0.1])
cbar = f.colorbar(sc, cax=cax)
cbar.set_label('log firing rate')
cbar.set_ticks([sc.norm.vmin, sc.norm.vmax])
cbar.set_ticklabels([rf'$10^{{{min_fr}}}$', rf'$10^{{{max_fr}}}$'])

# flag any shank with no histology
for col in sorted(set(no_hist_cols)):
    bbox = ax[col].get_position()
    pad_x, pad_y = 0.006, 0.01
    rect = Rectangle(
        (bbox.x0 - pad_x, bbox.y0 - pad_y),
        bbox.width + 2 * pad_x, bbox.height + 2 * pad_y,
        transform=f.transFigure, fill=False, 
        edgecolor='xkcd:scarlet', linewidth=1, clip_on=False,
    )
    f.add_artist(rect)
    ax[col].text((bbox.width)/2, bbox.height + pad_y + 0.002, 'no histology', 
                    color='xkcd:scarlet', ha='center', va='bottom',
                    fontsize=axis_label, transform=ax[col].transAxes)

f.savefig(f'{save_dir}/width_by_depth_fr.png', dpi=400, bbox_inches='tight')
plt.show()