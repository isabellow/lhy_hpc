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
        if 'lhy_cell_pos' in session_data:
            cell_pos = session_data['lhy_cell_pos']
        else:
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

    # shank bookkeeping, from the dict where available
    # (cells on excluded shanks are dropped when the dict is built, so a
    #  shank can be missing from shank_idx entirely)
    n_shanks_bird = int(data_dict[bird].get('n_shanks',
                                            int(pos_dict[bird]['shank_idx'].max()) + 1))
    exclude = np.asarray(data_dict[bird].get('exclude_shank',
                                             np.zeros(n_shanks_bird, dtype=bool)))

    # shanks marked 'exclude' in the anatomy sheet get no column at all
    plot_shanks = [sh for sh in range(n_shanks_bird)
                   if (not exclude[sh]) and np.any(pos_dict[bird]['shank_idx'] == sh)]
    if len(plot_shanks) == 0:
        print(f'no included shanks with cells for {bird}, skipping')
        del pos_dict[bird]
        continue

    # flag shanks with no probe track in histology -- their positions come
    # from the intended surgery coordinates, not a measured track
    raw_tip = data_dict[bird].get('raw_tip_coords')
    if raw_tip is None:
        no_hist = np.zeros(n_shanks_bird, dtype=bool)
    else:
        no_hist = np.isnan(np.asarray(raw_tip)[:, :2]).any(axis=1)

    # per-shank mean AP position for this bird (generalized to n shanks)
    cell_pos = pos_dict[bird]['cell_pos']
    shank_ap = np.full(n_shanks_bird, np.nan)
    for sh in plot_shanks:
        sh_mask = pos_dict[bird]['shank_idx'] == sh
        shank_ap[sh] = np.mean(cell_pos[sh_mask, 1])
    pos_dict[bird]['shank_AP'] = shank_ap
    pos_dict[bird]['n_shanks'] = n_shanks_bird
    pos_dict[bird]['plot_shanks'] = plot_shanks
    pos_dict[bird]['no_hist'] = no_hist

# only birds with usable data remain
bird_ids = list(pos_dict.keys())


''' Build the (bird, shank) column ordering, sorted posterior -> anterior, across all birds '''
bird_shank_list = []  # one (bird, shank) entry per subplot column
ap_list = []
for bird in bird_ids:
    for sh in pos_dict[bird]['plot_shanks']:
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
ax = np.atleast_1d(ax)  # subplots returns a bare Axes when n_cols == 1
vmax = np.nanmax(all_cell_fr)
vmin = np.nanmin(all_cell_fr)
no_hist_cols = [] # flag shanks with no probe track in histology

for bird in bird_ids:
    # data params
    plot_shanks = pos_dict[bird]['plot_shanks']
    no_hist = pos_dict[bird]['no_hist']
    cell_pos = pos_dict[bird]['cell_pos']
    cell_dv = cell_pos[:, -1].copy()
    shank_idx = pos_dict[bird]['shank_idx']
    wf_width = pos_dict[bird]['waveform_props'][1]
    cell_fr = pos_dict[bird]['waveform_props'][2]
    n_cells = wf_width.shape[0]

    # check for histology annotations
    if 'lhy_dvs' in data_dict[bird]:
        lhy_dvs = data_dict[bird]['lhy_dvs']
    else:
        n_shanks = pos_dict[bird]['n_shanks']
        lhy_dvs = np.full((n_shanks, 2), np.nan)

    # plotting params
    jit_dv = np.random.randn(n_cells) * 2
    jit_w = np.random.randn(n_cells) * (2 / 600)

    for sh in plot_shanks:
        col = col_idx_for[(bird, sh)]
        sh_mask = shank_idx == sh
        if no_hist[sh]:
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

        # plot the LHy boundaries relative to the probe location
        lhy_lims = lhy_dvs[sh]
        ax[col].hlines(lhy_lims, [0, 0], [1, 1], 
                        colors='xkcd:scarlet',
                        linestyles='dashed', lw=0.5)

        # add a reference line and title
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

# flag any shank with no probe track in histology.
# adjacent flagged columns share a single box + label, so the labels don't
# collide when several shanks in a row are missing histology
flag_cols = sorted(set(no_hist_cols))
col_groups = []
for col in flag_cols:
    if col_groups and col == col_groups[-1][-1] + 1:
        col_groups[-1].append(col)
    else:
        col_groups.append([col])

pad_x, pad_y = 0.006, 0.01
for grp in col_groups:
    bbox_first = ax[grp[0]].get_position()
    bbox_last = ax[grp[-1]].get_position()
    rect = Rectangle(
        (bbox_first.x0 - pad_x, bbox_first.y0 - pad_y),
        (bbox_last.x1 - bbox_first.x0) + 2 * pad_x, bbox_first.height + 2 * pad_y,
        transform=f.transFigure, fill=False,
        edgecolor='xkcd:scarlet', linewidth=1, clip_on=False,
    )
    f.add_artist(rect)
    # wrap onto 2 lines over a single column, where one line doesn't fit
    label = 'no histology' if len(grp) > 1 else 'no\nhistology'
    f.text((bbox_first.x0 + bbox_last.x1) / 2, bbox_first.y1 - pad_y,
           label, color='xkcd:scarlet', ha='center', va='top',
           fontsize=axis_label, transform=f.transFigure)

f.savefig(f'{save_dir}/width_by_depth_fr.png', dpi=400, bbox_inches='tight')
plt.show()