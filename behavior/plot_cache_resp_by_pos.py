import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import os
import sys
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

'''
Get the percent of caches that each cell is active.
Plot against depth/waveform width for each shank, sorted by AP location.

Stim-based cell filtering and nucleus-boundary plotting are optional.
'''

''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Stim toggle '''
use_stim_filter = False

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9

''' Load the data dictionary '''
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = list(data_dict.keys())

''' Make session list: sessions with both cell position and behavior + ephys data '''
pos_sessions = []
behavior_sessions = []
for bird in bird_ids:
    for session_id in data_dict[bird]['all_sessions']:
        if 'cell_pos' in data_dict[bird][session_id].keys():
            pos_sessions.append(f'{bird}_{session_id}')
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) and ('ephys' in preprocessed_data):
            behavior_sessions.append(f'{bird}_{session_id}')
sessions_to_use = list(set(pos_sessions) & set(behavior_sessions))

''' Collect data across sessions for each bird '''
pos_dict = {}
for bird in bird_ids:
    pos_dict[bird] = {}
    for session_id in data_dict[bird]['all_sessions']:
        if f'{bird}_{session_id}' not in sessions_to_use:
            continue

        # get the position of each cell (ML, est AP, DV)
        cell_pos = data_dict[bird][session_id]['cell_pos']

        # get the waveform props (asymm, width, log_fr)
        waveform_props = data_dict[bird][session_id]['waveform_props']

        # get the active cache fraction / modulation
        active_cache_frac = data_dict[bird][session_id]['barcode_dict']['active_cache_frac']
        cache_modulated = data_dict[bird][session_id]['barcode_dict']['cache_modulated']

        # excitatory/inhibitory indices - probably not accurate for LHy data
        exc_idx = data_dict[bird][session_id]['excitatory_idx']
        inhib_idx = data_dict[bird][session_id]['inhibitory_idx']

        # shank index -- int array, 0..n_shanks-1 (arbitrary n_shanks)
        shank_idx = data_dict[bird][session_id]['shank_idx']

        # index cells by stim-responsive channels -- optional
        if use_stim_filter and ('stim_resp_idx_ch' in data_dict[bird][session_id].keys()):
            cell_stim_idx = idx_cells_by_stim(data_dict, bird, session_id)
        else:
            if use_stim_filter:
                print(f'warning! use_stim_filter=True but no stim data for {bird}_{session_id}, '
                      f'including all cells')
            cell_stim_idx = np.ones(exc_idx.shape[0]).astype(bool)

        fields = {
            'cell_pos': cell_pos,
            'waveform_props': waveform_props,
            'active_cache_frac': active_cache_frac,
            'cache_modulated': cache_modulated,
            'excitatory_idx': exc_idx.astype(bool),
            'inhibitory_idx': inhib_idx.astype(bool),
            'cell_stim_idx': cell_stim_idx.astype(bool),
            'shank_idx': shank_idx.astype(int),
        }

        # store for this bird, concatenating across sessions
        if 'cell_pos' in pos_dict[bird]:
            pos_dict[bird]['cell_pos'] = np.row_stack((pos_dict[bird]['cell_pos'], fields['cell_pos']))
            pos_dict[bird]['waveform_props'] = np.column_stack((pos_dict[bird]['waveform_props'],
                                                                 fields['waveform_props']))
            for key in ['active_cache_frac', 'cache_modulated', 'excitatory_idx',
                        'inhibitory_idx', 'cell_stim_idx', 'shank_idx']:
                pos_dict[bird][key] = np.append(pos_dict[bird][key], fields[key])
        else:
            pos_dict[bird].update(fields)

    # skip birds with no qualifying sessions (no cell_pos + behavior/ephys overlap),
    # instead of relying on a hardcoded bird-name exclusion list
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

''' Define/create the save folder '''
save_dir = f"{save_figs_dir}/"
os.makedirs(save_dir, exist_ok=True)

''' Histogram of cache responsiveness (optionally restricted to stim-responsive channels) '''
active_cache_frac_all = np.concatenate([pos_dict[b]['active_cache_frac'][pos_dict[b]['cell_stim_idx']]
                                         for b in bird_ids])
excitatory_idx_all = np.concatenate([pos_dict[b]['excitatory_idx'][pos_dict[b]['cell_stim_idx']]
                                      for b in bird_ids])
inhibitory_idx_all = np.concatenate([pos_dict[b]['inhibitory_idx'][pos_dict[b]['cell_stim_idx']]
                                      for b in bird_ids])

f, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
pct_active = active_cache_frac_all * 100
ax[0].hist(pct_active[excitatory_idx_all], bins=30)
ax[1].hist(pct_active[inhibitory_idx_all], bins=30)
ax[1].set_xlabel('% caches active')
ax[0].set_ylabel('N excitatory cells')
ax[1].set_ylabel('N inhibitory cells')
title_suffix = ' (stim-responsive channels only)' if use_stim_filter else ''
f.suptitle(f'cache responsiveness{title_suffix}', fontsize=title_size, y=1.02)
fname = 'caches_active_stim_only.png' if use_stim_filter else 'caches_active_all_cells.png'
f.savefig(f'{save_dir}{fname}', dpi=600, bbox_inches='tight')
plt.show()

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

''' Plot depth x waveform-width, colored by cache modulation, one column per shank '''
gs_kw = dict(hspace=0.1, wspace=0.3)
f, ax = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 10), sharey=True, gridspec_kw=gs_kw, squeeze=False)

ylims = [610, -10] # hippocampus
ylims = [6150, 5450] # lateral hypothalamus

alpha_pts = 0.8
size_pts = 6
e_color = 'xkcd:orange'
s_color = 'xkcd:cerulean'
n_color = 'xkcd:gray'


def _cache_mod_colors(mod_vals):
    colors = np.full(mod_vals.shape[0], n_color, dtype=object)
    colors[mod_vals == 1] = e_color
    colors[mod_vals == -1] = s_color
    return colors.tolist()


no_hist_cols = []  # subplot columns to flag as missing histology (shank AP is NaN)

for bird in bird_ids:
    n_shanks_bird = pos_dict[bird]['n_shanks']
    cell_pos = pos_dict[bird]['cell_pos']
    cell_dv = cell_pos[:, -1]  # view into cell_pos; in-place shifts below also update cell_pos
    shank_idx = pos_dict[bird]['shank_idx']
    wf_width = pos_dict[bird]['waveform_props'][1]
    exc_cells = pos_dict[bird]['excitatory_idx']
    inhb_cells = pos_dict[bird]['inhibitory_idx']
    cache_modulation = pos_dict[bird]['cache_modulated']
    n_cells = wf_width.shape[0]

    # nucleus boundaries, per shank -- only available/plotted once stim collision
    # data has been collected (collect_stim_response_data); shape (n_shanks, 2)
    if use_stim_filter and ('nucleus_dvs' in data_dict[bird]):
        nucleus_dvs = data_dict[bird]['nucleus_dvs']
    else:
        nucleus_dvs = np.full((n_shanks_bird, 2), np.nan)

    jit_dv = np.random.randn(n_cells) * 2
    jit_w = np.random.randn(n_cells) * (2 / 600)

    for sh in range(n_shanks_bird):
        col = col_idx_for[(bird, sh)]
        sh_mask = shank_idx == sh

        if np.isnan(pos_dict[bird]['shank_AP'][sh]):
            no_hist_cols.append(col)

        sh_nuc_lims = nucleus_dvs[sh]

        # shift depths (+ nucleus bounds) so the min DV on this shank is zero
        finite_vals = np.append(cell_dv[sh_mask], sh_nuc_lims[np.isfinite(sh_nuc_lims)])
        if finite_vals.size:
            min_dv = np.nanmin(finite_vals)
            if min_dv < 0:
                cell_dv[sh_mask] = cell_dv[sh_mask] - min_dv
                sh_nuc_lims = sh_nuc_lims - min_dv

        for row, cell_type_mask in enumerate([exc_cells, inhb_cells]):
            this_mask = sh_mask & cell_type_mask
            cell_colors = _cache_mod_colors(cache_modulation[this_mask])
            ax[row, col].scatter(wf_width[this_mask] + jit_w[this_mask],
                                  cell_dv[this_mask] + jit_dv[this_mask],
                                  c=cell_colors, s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
            ax[row, col].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
            if np.any(np.isfinite(sh_nuc_lims)):
                ax[row, col].hlines(sh_nuc_lims, [0, 0], [1, 1],
                                     colors='xkcd:scarlet', linestyles='dashed', lw=0.5)

        ax[0, col].set_title(f'{bird}', fontsize=axis_label)

    # format axes for this bird's columns
    for sh in range(n_shanks_bird):
        col = col_idx_for[(bird, sh)]
        for row in range(2):
            ax[row, col].set_xlim(0, 1)
            ax[row, col].spines['right'].set_visible(False)
            ax[row, col].spines['top'].set_visible(False)
            ax[row, col].spines['bottom'].set_bounds(0, 1)
            ax[row, col].spines['left'].set_bounds(600, 0)
            ax[row, col].set_xticks([0, 0.5, 1])
            ax[row, col].set_xticklabels(['0', '0.5', '1'])

# universal formatting
ax[0, 0].set_ylim(ylims)
ax[0, 0].set_ylabel('excitatory cells\ndepth (um)', fontsize=axis_label)
ax[1, 0].set_ylabel('inhibitory cells\ndepth (um)', fontsize=axis_label)
f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.06)
f.suptitle(r"shanks sorted posterior $\rightarrow$ anterior", fontsize=axis_label, y=0.93)

# legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='enhanced',
           markerfacecolor=e_color, markersize=6),
    Line2D([0], [0], marker='o', color='w', label='suppressed',
           markerfacecolor=s_color, markersize=6),
    Line2D([0], [0], marker='o', color='w', label='not modulated',
           markerfacecolor=n_color, markersize=6),
]
if use_stim_filter:
    legend_elements.append(Line2D([0], [0], color='xkcd:scarlet', lw=1.5, linestyle='dashed',
                                   label='nucleus boundaries'))
ax[0, -1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

# flag any shank with no AP/histology info available (generic -- not tied to a specific bird)
for col in sorted(set(no_hist_cols)):
    for row in range(2):
        bbox = ax[row, col].get_position()
        pad_x, pad_y = 0.006, 0.01
        rect = Rectangle(
            (bbox.x0 - pad_x, bbox.y0 - pad_y),
            bbox.width + 2 * pad_x, bbox.height + 2 * pad_y,
            transform=f.transFigure, fill=False, edgecolor='red', linewidth=1, clip_on=False,
        )
        f.add_artist(rect)
    ax[0, col].text(0.5, 1.15, 'no histology', color='red', ha='center', va='bottom',
                     fontsize=axis_label, transform=ax[0, col].transAxes)

f.savefig(f'{save_dir}/width_by_depth_sig_cache_mod.png', dpi=400, bbox_inches='tight')
plt.show()
