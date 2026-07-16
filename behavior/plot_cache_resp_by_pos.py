import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import os 
import sys
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim


'''
Get the percent of caches that each cell is active
Plot in 3D brain space
Plot against depth/waveform width for each shank, sorted by AP location

TODO:
save approx DV positions for RBY cells and include in this plot
(note AP position not recoverable, but could est. based on damaged slices to situate subplots)
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Load the data dictionary and get bird ids '''
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

# remove birds with no behavior sessions
no_hist = ['LIM63']
bird_ids = np.setdiff1d(bird_ids, no_hist)

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9

''' Make session list '''
pos_sessions = []
behavior_sessions = []
for i, bird in enumerate(bird_ids):
    for session_id in data_dict[bird]['all_sessions']:
        if 'cell_pos' in data_dict[bird][session_id].keys():
            pos_sessions.append(f'{bird}_{session_id}')
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(f'{bird}_{session_id}')
sessions_to_use = list(set(pos_sessions) & set (behavior_sessions))


''' Collect data across sessions for each bird '''
pos_dict = {}
all_AP = np.asarray([])
for i, bird in enumerate(bird_ids):
    pos_dict[bird] = {}
    for session_id in data_dict[bird]['all_sessions']:
        if f'{bird}_{session_id}' in sessions_to_use:
            # get the position of each cell (ML, est AP, DV)
            cell_pos = data_dict[bird][session_id]['cell_pos']
            
            # get the waveform props (asymm, width, log_fr)
            waveform_props = data_dict[bird][session_id]['waveform_props']

            # get the active cache fraction
            active_cache_frac = data_dict[bird][session_id]['barcode_dict']['active_cache_frac']
            cache_modulated = data_dict[bird][session_id]['barcode_dict']['cache_modulated']

            # index cells by stim responsive channels
            cell_stim_idx = idx_cells_by_stim(data_dict, bird, session_id)

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
                pos_dict[bird]['active_cache_frac'] = np.append(pos_dict[bird]['active_cache_frac'], active_cache_frac)
                pos_dict[bird]['cache_modulated'] = np.append(pos_dict[bird]['cache_modulated'], cache_modulated)
                pos_dict[bird]['cell_stim_idx'] = np.append(pos_dict[bird]['cell_stim_idx'], cell_stim_idx.astype(bool))
                pos_dict[bird]['excitatory_idx'] = np.append(pos_dict[bird]['excitatory_idx'], exc_idx.astype(bool))
                pos_dict[bird]['inhibitory_idx'] = np.append(pos_dict[bird]['inhibitory_idx'], inhib_idx.astype(bool))
            else:
                pos_dict[bird]['cell_pos'] = cell_pos
                pos_dict[bird]['shank_A_idx'] = shank_idx.astype(bool)
                pos_dict[bird]['waveform_props'] = waveform_props
                pos_dict[bird]['active_cache_frac'] = active_cache_frac
                pos_dict[bird]['cache_modulated'] = cache_modulated
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

''' Histogram of cache responsiveness for stim channels only '''
active_cache_frac_all = np.asarray([])
excitatory_idx_all = np.asarray([]).astype(bool)
inhibitory_idx_all = np.asarray([]).astype(bool)
cache_modulation_all = np.asarray([])
for bird in bird_ids:
    this_active_cache = pos_dict[bird]['active_cache_frac'][pos_dict[bird]['cell_stim_idx']]
    this_cache_mod = pos_dict[bird]['cache_modulated'][pos_dict[bird]['cell_stim_idx']]
    this_exc_idx = pos_dict[bird]['excitatory_idx'][pos_dict[bird]['cell_stim_idx']]
    this_inhib_idx = pos_dict[bird]['inhibitory_idx'][pos_dict[bird]['cell_stim_idx']]
    active_cache_frac_all = np.append(active_cache_frac_all, this_active_cache)
    excitatory_idx_all = np.append(excitatory_idx_all, this_exc_idx)
    inhibitory_idx_all = np.append(inhibitory_idx_all, this_inhib_idx)
    cache_modulation_all = np.append(cache_modulation_all, this_cache_mod)

# fig params
f, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

# plot the percent of caches that each cell was active for
pct_active = active_cache_frac_all*100
ax[0].hist(pct_active[excitatory_idx_all], bins=30)
ax[1].hist(pct_active[inhibitory_idx_all], bins=30)

# ticks and labels
ax[1].set_xlabel(f'% caches active')
ax[0].set_ylabel('N excitatory cells')
ax[1].set_ylabel('N inhibitory cells')

f.savefig(f'{save_figs_dir}caches_active_stim_only.png', dpi=600, bbox_inches='tight')
plt.show()

''' Plot in 3D brain space '''


# ''' Plot against depth/waveform width for each shank, sorted by AP location '''
# # data params
# n_birds = len(bird_ids)
# n_shanks = n_birds*2
# ap_sort = np.argsort(all_AP) 
# ap_sort_idx = np.argsort(ap_sort)
# log_fr = all_waveform_props[2]

# excitatory_idx_all = np.asarray([])
# inhibitory_idx_all = np.asarray([])
# for bird in bird_ids:
#     this_exc_idx = pos_dict[bird]['excitatory_idx']
#     this_inhib_idx = pos_dict[bird]['inhibitory_idx']
#     excitatory_idx_all = np.append(excitatory_idx_all, this_exc_idx)
#     inhibitory_idx_all = np.append(inhibitory_idx_all, this_inhib_idx)

# exc_vmax = 0.6
# exc_vmin = 0
# inhb_vmax = 1
# inhb_vmin = 0

# # fig params
# gs_kw = dict(hspace=0.1, wspace=0.3)
# f, ax = plt.subplots(2, n_shanks, figsize=(10, 10),
#                      sharey=True, gridspec_kw=gs_kw)
# title_size = 14
# axis_label = 12
# tick_label = 9
# ylims = [610, -10]
# alpha_pts = 0.8
# size_pts = 6

# i = 0
# for bird in bird_ids:
#     # cache activity
#     active_cache_frac = pos_dict[bird]['active_cache_frac']
#     cache_modulation = pos_dict[bird]['cache_modulated']
#     sig_modulation_idx = np.abs(cache_modulation).astype(bool)
    
#     # nucleus boundaries - A min, A max; B min, B max
#     nucleus_dvs = data_dict[bird]['nucleus_dvs']
#     A_nuc_lims = nucleus_dvs[0]
#     B_nuc_lims = nucleus_dvs[1]

#     # waveform properties
#     wf_width = pos_dict[bird]['waveform_props'][1]
#     cell_fr = pos_dict[bird]['waveform_props'][2]
#     n_cells = wf_width.shape[0]

#     # positions
#     cell_pos = pos_dict[bird]['cell_pos']
#     cell_dv = cell_pos[:, -1]
#     _, ap_idx = np.unique(cell_pos[:, 1], return_inverse=True)
#     B_idx = ap_idx < 3
#     A_idx = ap_idx >= 3

#     # update positions (cells, nucleus bounds) s.t. min DV is zero
#     min_dv_A = np.min(np.append(cell_dv[A_idx], A_nuc_lims))
#     min_dv_B = np.min(np.append(cell_dv[B_idx], B_nuc_lims))
#     if min_dv_A < 0:
#         cell_dv[A_idx] = cell_dv[A_idx] - min_dv_A
#         A_nuc_lims = A_nuc_lims - min_dv_A
#     if min_dv_B < 0:
#         cell_dv[B_idx] = cell_dv[B_idx] - min_dv_B
#         B_nuc_lims = B_nuc_lims - min_dv_B

#     # cluster indices
#     exc_cells = pos_dict[bird]['excitatory_idx']
#     inhb_cells = pos_dict[bird]['inhibitory_idx']

#     # index by shank and cluster
#     B_excite = B_idx & exc_cells & sig_modulation_idx
#     B_inhib = B_idx & inhb_cells & sig_modulation_idx
#     A_excite = A_idx & exc_cells & sig_modulation_idx
#     A_inhib = A_idx & inhb_cells & sig_modulation_idx

#     # subplot indices
#     B_ax = ap_sort_idx[i]
#     A_ax = ap_sort_idx[i+1]

#     # jitter
#     jit_dv = np.random.randn(n_cells)*2
#     jit_w = np.random.randn(n_cells)*(2/600)

#     # highlight the excitatory cells
#     sc_exc = ax[0, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite],
#                                  cell_dv[B_excite] + jit_dv[B_excite],
#                                  c=active_cache_frac[B_excite], cmap='viridis',
#                                  vmin=exc_vmin, vmax=exc_vmax,
#                                  s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[0, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
#                         cell_dv[B_inhib] + jit_dv[B_inhib], 
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[0, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
#     ax[0, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
#                         cell_dv[A_excite] + jit_dv[A_excite],
#                         c=active_cache_frac[A_excite], cmap='viridis', 
#                         vmin=exc_vmin, vmax=exc_vmax,
#                         s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[0, A_ax].scatter(wf_width[A_inhib] + jit_w[A_inhib],
#                         cell_dv[A_inhib] + jit_dv[A_inhib], 
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[0, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

#     # highlight the inhibitory cells
#     ax[1, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite], 
#                         cell_dv[B_excite] + jit_dv[B_excite],
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     sc_inhb = ax[1, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
#                                   cell_dv[B_inhib] + jit_dv[B_inhib],
#                                   c=active_cache_frac[B_inhib], cmap='viridis', 
#                                   vmin=inhb_vmin, vmax=inhb_vmax,
#                                   s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[1, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
#     ax[1, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
#                         cell_dv[A_excite] + jit_dv[A_excite],
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[1, A_ax].scatter(wf_width[A_inhib] +  + jit_w[A_inhib],
#                         cell_dv[A_inhib] + jit_dv[A_inhib], 
#                         c=active_cache_frac[A_inhib], cmap='viridis', 
#                         vmin=inhb_vmin, vmax=inhb_vmax,
#                         s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[1, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

#     # add rough nucleus boundaries
#     ax[0, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[0, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[1, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[1, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    
#     # titles
#     ax[0, A_ax].set_title(f'{bird}', fontsize=axis_label)
#     ax[0, B_ax].set_title(f'{bird}', fontsize=axis_label)
    

#     # format axes
#     for j in range(2):
#         for sh in [B_ax, A_ax]:
#             ax[j, sh].set_xlim(0, 1)
#             ax[j, sh].spines['right'].set_visible(False)
#             ax[j, sh].spines['top'].set_visible(False)
#             ax[j, sh].spines['bottom'].set_bounds(0, 1)
#             ax[j, sh].spines['left'].set_bounds(600, 0)
#             ax[j, sh].set_xticks([0, 0.5, 1])
#             ax[j, sh].set_xticklabels(['0', '0.5', '1'])

#     # update indices
#     i += 2

# # universal formatting
# ax[0, 0].set_ylim(ylims) 
# ax[0, 0].set_ylabel('excitatory cells\ndepth (um)', fontsize=axis_label)
# ax[1, 0].set_ylabel('inhibitory cells\ndepth (um)', fontsize=axis_label)
# f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.06)
# f.suptitle(r"shanks sorted posterior $\rightarrow$ anterior", fontsize=axis_label, y=0.93)

# # colorbars
# cax = f.add_axes([0.93, 0.8, 0.008, 0.1])
# cbar = f.colorbar(sc_exc, cax=cax)
# cbar.set_label('active cache %')
# cbar.set_ticks([sc_exc.norm.vmin, sc_exc.norm.vmax])
# cbar.set_ticklabels([0, int(exc_vmax*100)])

# cax = f.add_axes([0.93, 0.37, 0.008, 0.1])
# cbar = f.colorbar(sc_inhb, cax=cax)
# cbar.set_label('active cache %')
# cbar.set_ticks([sc_inhb.norm.vmin, sc_inhb.norm.vmax])
# cbar.set_ticklabels([0, 100])

# f.savefig(f'{save_figs_dir}/width_by_depth_sig_cache_mod.png', dpi=400, bbox_inches='tight')
# plt.show()



''' Color by enhanced or suppressed '''
# data params
n_birds = len(bird_ids)
n_shanks = n_birds*2
ap_sort = np.argsort(all_AP) 
ap_sort_idx = np.argsort(ap_sort)
log_fr = all_waveform_props[2]

excitatory_idx_all = np.asarray([])
inhibitory_idx_all = np.asarray([])
for bird in bird_ids:
    this_exc_idx = pos_dict[bird]['excitatory_idx']
    this_inhib_idx = pos_dict[bird]['inhibitory_idx']
    excitatory_idx_all = np.append(excitatory_idx_all, this_exc_idx)
    inhibitory_idx_all = np.append(inhibitory_idx_all, this_inhib_idx)

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

e_color = 'xkcd:orange'
s_color = 'xkcd:cerulean'
n_color = 'xkcd:gray'

i = 0
for bird in bird_ids:
    # cache activity
    active_cache_frac = pos_dict[bird]['active_cache_frac']
    cache_modulation = pos_dict[bird]['cache_modulated']

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

    # highlight the excitatory cells - B shank
    cell_colors = []
    for mod in cache_modulation[B_excite]:
        if mod == 1:
            cell_colors.append(e_color)
        elif mod == -1:
            cell_colors.append(s_color)
        else:
            cell_colors.append(n_color)
    sc_exc = ax[0, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite],
                                 cell_dv[B_excite] + jit_dv[B_excite],
                                 c=cell_colors, s=size_pts,
                                 lw=0, zorder=1, alpha=alpha_pts)
    # ax[0, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
    #                     cell_dv[B_inhib] + jit_dv[B_inhib], 
    #                     c='xkcd:gray', 
    #                     s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[0, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
    
    # highlight the excitatory cells - A shank
    cell_colors = []
    for mod in cache_modulation[A_excite]:
        if mod == 1:
            cell_colors.append(e_color)
        elif mod == -1:
            cell_colors.append(s_color)
        else:
            cell_colors.append(n_color)
    ax[0, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
                        cell_dv[A_excite] + jit_dv[A_excite],
                        c=cell_colors, s=size_pts, 
                        lw=0, zorder=1, alpha=alpha_pts)
    # ax[0, A_ax].scatter(wf_width[A_inhib] + jit_w[A_inhib],
    #                     cell_dv[A_inhib] + jit_dv[A_inhib], 
    #                     c='xkcd:gray', 
    #                     s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[0, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

    # highlight the inhibitory cells - B shank
    cell_colors = []
    for mod in cache_modulation[B_inhib]:
        if mod == 1:
            cell_colors.append(e_color)
        elif mod == -1:
            cell_colors.append(s_color)
        else:
            cell_colors.append(n_color)
    # ax[1, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite], 
    #                     cell_dv[B_excite] + jit_dv[B_excite],
    #                     c='xkcd:gray', 
    #                     s=size_pts, lw=0, zorder=0, alpha=0.2)
    sc_inhb = ax[1, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
                                  cell_dv[B_inhib] + jit_dv[B_inhib],
                                  c=cell_colors,
                                  s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
    ax[1, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
    
    # highlight the inhibitory cells - A shank
    cell_colors = []
    for mod in cache_modulation[A_inhib]:
        if mod == 1:
            cell_colors.append(e_color)
        elif mod == -1:
            cell_colors.append(s_color)
        else:
            cell_colors.append(n_color)
    # ax[1, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
    #                     cell_dv[A_excite] + jit_dv[A_excite],
    #                     c='xkcd:gray', 
    #                     s=size_pts, lw=0, zorder=0, alpha=0.2)
    ax[1, A_ax].scatter(wf_width[A_inhib] +  + jit_w[A_inhib],
                        cell_dv[A_inhib] + jit_dv[A_inhib], 
                        c=cell_colors,
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

# add a legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='enhanced',
            markerfacecolor=e_color, markersize=6),
    Line2D([0], [0], marker='o', color='w', label='suppressed',
            markerfacecolor=s_color, markersize=6),
    Line2D([0], [0], marker='o', color='w', label='not modulated',
            markerfacecolor=n_color, markersize=6),
    Line2D([0], [0], color='xkcd:scarlet', lw=1.5, linestyle='dashed',
            label='nucleus boundaries')
]
ax[0, -1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(3.5, 1))

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

f.savefig(f'{save_figs_dir}/width_by_depth_sig_cache_mod_alt.png', dpi=400, bbox_inches='tight')
plt.show()



# ''' Normalize DV by relative position in nucleus
# call center 0, shallow end -1, deep end 1
# everything < -1 is DL
# everything > 1 is the rest of the hippocampus...
# '''
# for bird in bird_ids:
#     # nucleus boundaries - A min, A max; B min, B max
#     nucleus_dvs = data_dict[bird]['nucleus_dvs'].copy()
#     A_nuc_lims = nucleus_dvs[0]
#     B_nuc_lims = nucleus_dvs[1]
#     A_nuc_center = np.mean(A_nuc_lims)
#     B_nuc_center = np.mean(B_nuc_lims)

#     # positions
#     cell_pos = pos_dict[bird]['cell_pos']
#     cell_dv = cell_pos[:, -1]
#     _, ap_idx = np.unique(cell_pos[:, 1], return_inverse=True)
#     B_idx = ap_idx < 3
#     A_idx = ap_idx >= 3

#     # separate by shank and normalize
#     A_dv = cell_dv[A_idx]
#     B_dv = cell_dv[B_idx]
#     A_dv_norm = (A_dv - A_nuc_center) / (A_nuc_lims[1] - A_nuc_center)
#     B_dv_norm = (B_dv - B_nuc_center) / (B_nuc_lims[1] - B_nuc_center)
#     cell_dv_norm = np.zeros_like(cell_dv)
#     cell_dv_norm[A_idx] = A_dv_norm
#     cell_dv_norm[B_idx] = B_dv_norm

#     pos_dict[bird]['cell_dv_norm'] = cell_dv_norm


''' Plot against depth/waveform width for each shank, sorted by AP location '''
# this doesn't work great, maybe would be better if I used raw depth on probe, rather than DV
# fig params
# gs_kw = dict(hspace=0.1, wspace=0.3)
# f, ax = plt.subplots(2, n_shanks, figsize=(10, 10),
#                      sharey=True, gridspec_kw=gs_kw)
# title_size = 14
# axis_label = 12
# tick_label = 9
# ylims = [4, -4.5]
# alpha_pts = 0.8
# size_pts = 7

# i = 0
# for bird in bird_ids:
#     # cache activity
#     active_cache_frac = pos_dict[bird]['active_cache_frac']
    
#     # nucleus boundaries - A min, A max; B min, B max
#     A_nuc_lims = [-1, 1]
#     B_nuc_lims = [-1, 1]

#     # waveform properties
#     wf_width = pos_dict[bird]['waveform_props'][1]
#     cell_fr = pos_dict[bird]['waveform_props'][2]
#     n_cells = wf_width.shape[0]

#     # positions
#     cell_pos = pos_dict[bird]['cell_pos']
#     cell_dv = pos_dict[bird]['cell_dv_norm']
#     _, ap_idx = np.unique(cell_pos[:, 1], return_inverse=True)
#     B_idx = ap_idx < 3
#     A_idx = ap_idx >= 3

#     # cluster indices
#     exc_cells = pos_dict[bird]['excitatory_idx']
#     inhb_cells = pos_dict[bird]['inhibitory_idx']

#     # index by shank and cluster
#     B_excite = B_idx & exc_cells
#     B_inhib = B_idx & inhb_cells
#     A_excite = A_idx & exc_cells
#     A_inhib = A_idx & inhb_cells

#     # subplot indices
#     B_ax = ap_sort_idx[i]
#     A_ax = ap_sort_idx[i+1]

#     # jitter
#     jit_dv = np.random.randn(n_cells)*0.02
#     jit_w = np.random.randn(n_cells)*(2/600)

#     # highlight the excitatory cells
#     sc_exc = ax[0, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite],
#                                  cell_dv[B_excite] + jit_dv[B_excite],
#                                  c=active_cache_frac[B_excite], cmap='viridis',
#                                  vmin=exc_vmin, vmax=exc_vmax,
#                                  s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[0, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
#                         cell_dv[B_inhib] + jit_dv[B_inhib], 
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[0, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
#     ax[0, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
#                         cell_dv[A_excite] + jit_dv[A_excite],
#                         c=active_cache_frac[A_excite], cmap='viridis', 
#                         vmin=exc_vmin, vmax=exc_vmax,
#                         s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[0, A_ax].scatter(wf_width[A_inhib] + jit_w[A_inhib],
#                         cell_dv[A_inhib] + jit_dv[A_inhib], 
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[0, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

#     # highlight the inhibitory cells
#     ax[1, B_ax].scatter(wf_width[B_excite] + jit_w[B_excite], 
#                         cell_dv[B_excite] + jit_dv[B_excite],
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     sc_inhb = ax[1, B_ax].scatter(wf_width[B_inhib] + jit_w[B_inhib],
#                                   cell_dv[B_inhib] + jit_dv[B_inhib],
#                                   c=active_cache_frac[B_inhib], cmap='viridis', 
#                                   vmin=inhb_vmin, vmax=inhb_vmax,
#                                   s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[1, B_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
#     ax[1, A_ax].scatter(wf_width[A_excite] + jit_w[A_excite], 
#                         cell_dv[A_excite] + jit_dv[A_excite],
#                         c='xkcd:gray', 
#                         s=size_pts, lw=0, zorder=0, alpha=0.2)
#     ax[1, A_ax].scatter(wf_width[A_inhib] +  + jit_w[A_inhib],
#                         cell_dv[A_inhib] + jit_dv[A_inhib], 
#                         c=active_cache_frac[A_inhib], cmap='viridis', 
#                         vmin=inhb_vmin, vmax=inhb_vmax,
#                         s=size_pts, lw=0, zorder=1, alpha=alpha_pts)
#     ax[1, A_ax].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)

#     # add rough nucleus boundaries
#     ax[0, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[0, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[1, A_ax].hlines(A_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
#     ax[1, B_ax].hlines(B_nuc_lims, [0, 0], [1, 1], 
#                         colors='xkcd:scarlet', linestyles='dashed', lw=0.5)
    
#     # titles
#     ax[0, A_ax].set_title(f'{bird}', fontsize=axis_label)
#     ax[0, B_ax].set_title(f'{bird}', fontsize=axis_label)
    

#     # format axes
#     for j in range(2):
#         for sh in [B_ax, A_ax]:
#             ax[j, sh].set_xlim(0, 1)
#             ax[j, sh].spines['right'].set_visible(False)
#             ax[j, sh].spines['top'].set_visible(False)
#             ax[j, sh].spines['bottom'].set_bounds(0, 1)
#             ax[j, sh].spines['left'].set_bounds(ylims[0], ylims[1])
#             ax[j, sh].set_xticks([0, 0.5, 1])
#             ax[j, sh].set_xticklabels(['0', '0.5', '1'])

#     # update indices
#     i += 2

# # universal formatting
# ax[0, 0].set_ylim(ylims) 
# ax[0, 0].set_ylabel('excitatory cells\n', fontsize=axis_label)
# ax[1, 0].set_ylabel('inhibitory cells\n', fontsize=axis_label)
# f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.06)
# f.suptitle(r"shanks sorted posterior $\rightarrow$ anterior", fontsize=axis_label, y=0.93)
# ax[0, 0].text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
#                     ha='center', va='bottom', fontsize=tick_label)

# # colorbars
# cax = f.add_axes([0.93, 0.8, 0.008, 0.1])
# cbar = f.colorbar(sc_exc, cax=cax)
# cbar.set_label('active cache %')
# cbar.set_ticks([sc_exc.norm.vmin, sc_exc.norm.vmax])
# cbar.set_ticklabels([0, int(exc_vmax*100)])

# cax = f.add_axes([0.93, 0.37, 0.008, 0.1])
# cbar = f.colorbar(sc_inhb, cax=cax)
# cbar.set_label('active cache %')
# cbar.set_ticks([sc_inhb.norm.vmin, sc_inhb.norm.vmax])
# cbar.set_ticklabels([0, 100])

# f.savefig(f'{save_figs_dir}/width_by_nucleus_cache_active.png', dpi=400, bbox_inches='tight')
# plt.show()