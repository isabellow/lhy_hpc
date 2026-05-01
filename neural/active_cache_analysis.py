import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

import os 
import sys
sys.path.append("..//behavior/")
from format_behavior_data import dist_binned_mean_sem, load_behavior_data, get_n_seeds
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim, chunk_cells_by_region

'''
For each session:
Heatmap of activity against caches by neurons
--> sort caches by time in session
--> cluster ecitatory, inhibitory
--> sort neurons by baseline firing rate
--> sort neurons by dv position?

For each bird:
Histogram of pct caches active for excitatory vs. inhibitory units

Across birds:
Histogram of pct caches active for excitatory vs. inhibitory units
"" split by region
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# load the data dictionary and get bird ids
data_dict = np.load(data_file, allow_pickle=True).item()
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9

''' list of behavior sessions '''
all_behavior_sessions = []
for i, bird in enumerate(bird_ids):
    behavior_sessions = []
    for session_id in data_dict[bird]['all_sessions']:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)
    all_behavior_sessions.append(behavior_sessions)

# to store data across birds
all_birds_active_caches = np.asarray([])

# excitatory/inhibitory indices
exc_idx_all = np.asarray([]).astype(bool)
inhib_idx_all = np.asarray([]).astype(bool)

# to sort by stim responsiveness
all_birds_stim_idx = np.asarray([]).astype(bool)
all_birds_cell_loc = np.asarray([])

for bird in bird_ids:
    print(f'collecting cache activity for {bird}')
    bird_idx = bird_ids.index(bird)

    ''' Define/create the save folder'''
    save_dir = f"{save_figs_dir}/{bird}/"
    if os.path.isdir(save_dir):
        print('save directory exists')
    else:
        os.mkdir(save_dir)
    save_folder = f"{save_dir}/barcodes/"
    if os.path.isdir(save_folder):
        print('save folder exists')
    else:
        os.mkdir(save_folder)

    # to collect things across sessions
    active_cache_frac_all = np.asarray([])
    exc_idx_bird = np.asarray([]).astype(bool)
    inhib_idx_bird = np.asarray([]).astype(bool)
    stim_idx_bird = np.asarray([]).astype(bool)

    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions: 
        ''' Grab the data for this session '''
        barcode_dict = data_dict[bird][session_id]['barcode_dict']

        # get the active cache fraction
        active_cache_frac = barcode_dict['active_cache_frac']
        active_cache_frac_all = np.append(active_cache_frac_all, active_cache_frac)
        cache_vectors = barcode_dict['cache_vectors']

        # excitatory/inhibitory indices
        exc_idx = data_dict[bird][session_id]['excitatory_idx']
        exc_idx_bird = np.append(exc_idx_bird, exc_idx)
        inhib_idx = data_dict[bird][session_id]['inhibitory_idx']
        inhib_idx_bird = np.append(inhib_idx_bird, inhib_idx)

        # projection nucleus boundaries, rough anatomical indices
        stim_idx = idx_cells_by_stim(data_dict, bird, session_id)
        cell_loc_idx = chunk_cells_by_region(data_dict, bird, session_id)
        stim_idx_bird = np.append(stim_idx_bird, stim_idx)
        all_birds_cell_loc = np.append(all_birds_cell_loc, cell_loc_idx)

        # firing rate
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]

        ''' For plotting, chunk and sort the data '''
        # chunk the data
        excitatory_in_nucleus = cache_vectors[:, exc_idx & stim_idx]
        inhibitory_in_nucleus = cache_vectors[:, inhib_idx & stim_idx]
        excitatory_outside = cache_vectors[:, exc_idx & ~stim_idx]
        inhibitory_outside = cache_vectors[:, inhib_idx & ~stim_idx]

        # chunk the firing rates
        exc_stim_fr = log_fr[exc_idx & stim_idx]
        inhib_stim_fr = log_fr[inhib_idx & stim_idx]
        exc_no_stim_fr = log_fr[exc_idx & ~stim_idx]
        inhib_no_stim_fr = log_fr[inhib_idx & ~stim_idx]

        # chunk the active caches
        exc_stim_active = active_cache_frac[exc_idx & stim_idx]
        inhib_stim_active = active_cache_frac[inhib_idx & stim_idx]
        exc_no_stim_active = active_cache_frac[exc_idx & ~stim_idx]
        inhib_no_stim_active = active_cache_frac[inhib_idx & ~stim_idx]

        # save this for plotting
        n_cells_per_condition = np.zeros(4)
        n_cells_per_condition[0] = np.sum(exc_idx & stim_idx)
        n_cells_per_condition[1] = np.sum(inhib_idx & stim_idx)
        n_cells_per_condition[2] = np.sum(exc_idx & ~stim_idx)
        n_cells_per_condition[3] = np.sum(inhib_idx & ~stim_idx)

        ''' Plot the cache vectors for this session - sorted by firing rate '''
        # sort by firing rates & stick everything back together
        exc_in_nucleus_sorted = excitatory_in_nucleus[:, np.argsort(exc_stim_fr)]
        inhib_in_nucleus_sorted = inhibitory_in_nucleus[:, np.argsort(inhib_stim_fr)]
        excitatory_outside_sorted = excitatory_outside[:, np.argsort(exc_no_stim_fr)]
        inhibitory_outside_sorted = inhibitory_outside[:, np.argsort(inhib_no_stim_fr)]
        cache_vectors_sorted = np.column_stack([exc_in_nucleus_sorted, 
                                                inhib_in_nucleus_sorted, 
                                                excitatory_outside_sorted, 
                                                inhibitory_outside_sorted])

        # fig params
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        clims = [-3, 3]
        im1 = ax.imshow(cache_vectors_sorted, aspect='auto', 
                        cmap='bwr', clim=clims, 
                        interpolation='none')

        # label excitatory/inhibitory and in/out of nucleus
        ylims = ax.get_ylim()
        chunk_indices = np.cumsum(n_cells_per_condition)
        chunk_indices = np.insert(chunk_indices, 0, 0)
        chunk_labels = ['E in nucleus', 'I in', 'E out', 'I out']
        for i in range(4):
            if n_cells_per_condition[i] == 0:
                continue
            start_idx = chunk_indices[i]
            end_idx = chunk_indices[i+1]
            if i > 0:
                ax.vlines(start_idx-0.5, ylims[0], ylims[1], color='k', lw=1)
            ax.hlines(ylims[1]-0.5, start_idx, end_idx-1, color='k', lw=0.75)
            ax.text(np.mean(chunk_indices[i:i+2])-0.5, ylims[1]-0.6, chunk_labels[i],
                    size=axis_label, ha='center', va='bottom')

        # lims and labels
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(ylims[0], 0)
        ax.set_xlabel('neurons sorted by firing rate', fontsize=axis_label)
        ax.set_ylabel('cache #', fontsize=axis_label)

        # add a colorbar
        cax = f.add_axes([0.93, 0.4, 0.02, 0.32]) # [left, bottom, width, height]
        cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('activity (z-score)', fontsize=tick_label)
        cbar.set_ticks([])
        cbar.ax.text(0.5, -0.05, f'{clims[0]} std', transform=cbar.ax.transAxes,
                        ha='center', va='top', fontsize=tick_label)
        cbar.ax.text(0.5, 1.02, f'{clims[1]} std', transform=cbar.ax.transAxes,
                        ha='center', va='bottom', fontsize=tick_label)

        f.savefig(f'{save_folder}{session_id}_cache_vectors_fr.png', dpi=600, bbox_inches='tight')
        plt.show()

        ''' Sort by quadrant cache occurred in & firing rate '''
        # sort by firing rates & stick everything back together
        exc_in_nucleus_sorted = excitatory_in_nucleus[:, np.argsort(exc_stim_fr)]
        inhib_in_nucleus_sorted = inhibitory_in_nucleus[:, np.argsort(inhib_stim_fr)]
        excitatory_outside_sorted = excitatory_outside[:, np.argsort(exc_no_stim_fr)]
        inhibitory_outside_sorted = inhibitory_outside[:, np.argsort(inhib_no_stim_fr)]
        cache_vectors_sorted = np.column_stack([exc_in_nucleus_sorted, 
                                                inhib_in_nucleus_sorted, 
                                                excitatory_outside_sorted, 
                                                inhibitory_outside_sorted])

        # get the quadrant each cache belongs to
        cache_loc = barcode_dict['cache_loc']
        quad_1 = (cache_loc[:, 0] > 0) & (cache_loc[:, 1] > 0)
        quad_2 = (cache_loc[:, 0] > 0) & (cache_loc[:, 1] < 0)
        quad_3 = (cache_loc[:, 0] < 0) & (cache_loc[:, 1] < 0)
        quad_4 = (cache_loc[:, 0] < 0) & (cache_loc[:, 1] > 0)

        cache_vectors_quad1 = cache_vectors_sorted[quad_1]
        cache_vectors_quad2 = cache_vectors_sorted[quad_2]
        cache_vectors_quad3 = cache_vectors_sorted[quad_3]
        cache_vectors_quad4 = cache_vectors_sorted[quad_4]

        cache_vectors_sorted = np.row_stack([cache_vectors_quad1, 
                                                cache_vectors_quad2, 
                                                cache_vectors_quad3, 
                                                cache_vectors_quad4])
        # save this for plotting
        n_cache_per_condition = np.zeros(4)
        n_cache_per_condition[0] = np.sum(quad_1)
        n_cache_per_condition[1] = np.sum(quad_2)
        n_cache_per_condition[2] = np.sum(quad_3)
        n_cache_per_condition[3] = np.sum(quad_4)

        # fig params
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        clims = [-3, 3]
        im1 = ax.imshow(cache_vectors_sorted, aspect='auto', 
                        cmap='bwr', clim=clims, 
                        interpolation='none')

        # label excitatory/inhibitory and in/out of nucleus
        ylims = ax.get_ylim()
        chunk_indices = np.cumsum(n_cells_per_condition)
        chunk_indices = np.insert(chunk_indices, 0, 0)
        chunk_labels = ['E in nucleus', 'I in', 'E out', 'I out']
        for i in range(4):
            if n_cells_per_condition[i] == 0:
                continue
            start_idx = chunk_indices[i]
            end_idx = chunk_indices[i+1]
            if i > 0:
                ax.vlines(start_idx-0.5, ylims[0], ylims[1], color='k', lw=1)
            ax.hlines(ylims[1]-0.5, start_idx, end_idx-1, color='k', lw=0.75)
            ax.text(np.mean(chunk_indices[i:i+2])-0.5, ylims[1]-0.6, chunk_labels[i],
                    size=axis_label, ha='center', va='bottom')

        # delineate arena quadrants
        xlims = ax.get_xlim()
        chunk_indices = np.cumsum(n_cache_per_condition)
        for i in range(4):
            if n_cache_per_condition[i] == 0:
                continue
            if i > 0:
                start_idx = chunk_indices[i-1]
                ax.hlines(start_idx-0.5, xlims[0], xlims[1],
                            color='k', lw=1, linestyles='dotted')

        # lims and labels
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(ylims[0], 0)
        ax.set_xlabel('neurons sorted by firing rate', fontsize=axis_label)
        ax.set_ylabel('caches sorted by arena quadrant', fontsize=axis_label)

        # add a colorbar
        cax = f.add_axes([0.93, 0.4, 0.02, 0.32]) # [left, bottom, width, height]
        cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('activity (z-score)', fontsize=tick_label)
        cbar.set_ticks([])
        cbar.ax.text(0.5, -0.05, f'{clims[0]} std', transform=cbar.ax.transAxes,
                        ha='center', va='top', fontsize=tick_label)
        cbar.ax.text(0.5, 1.02, f'{clims[1]} std', transform=cbar.ax.transAxes,
                        ha='center', va='bottom', fontsize=tick_label)

        f.savefig(f'{save_folder}{session_id}_cache_vectors_quad.png', dpi=600, bbox_inches='tight')
        plt.show()

        ''' Sort caches by n seeds in arena '''
        # load behavioral data
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
        seed_struct, count_data = load_behavior_data(data_dir)
        n_seeds_arena = get_n_seeds(seed_struct)
        n_seeds_changes = np.diff(n_seeds_arena)
        n_seeds_caches = n_seeds_arena[1:][n_seeds_changes > 0]
        if n_seeds_caches.shape[0] == cache_vectors.shape[0]:
            # sort by firing rates & stick everything back together
            exc_in_nucleus_sorted = excitatory_in_nucleus[:, np.argsort(exc_stim_fr)]
            inhib_in_nucleus_sorted = inhibitory_in_nucleus[:, np.argsort(inhib_stim_fr)]
            excitatory_outside_sorted = excitatory_outside[:, np.argsort(exc_no_stim_fr)]
            inhibitory_outside_sorted = inhibitory_outside[:, np.argsort(inhib_no_stim_fr)]
            cache_vectors_sorted = np.column_stack([exc_in_nucleus_sorted, 
                                                    inhib_in_nucleus_sorted, 
                                                    excitatory_outside_sorted, 
                                                    inhibitory_outside_sorted])

            # sort by n seeds in arena at each cache
            cache_vectors_sorted = cache_vectors_sorted[np.argsort(n_seeds_caches)]

            # fig params
            f, ax = plt.subplots(1, 1, figsize=(6, 4))
            clims = [-3, 3]
            im1 = ax.imshow(cache_vectors_sorted, aspect='auto', 
                            cmap='bwr', clim=clims, 
                            interpolation='none')

            # label excitatory/inhibitory and in/out of nucleus
            ylims = ax.get_ylim()
            chunk_indices = np.cumsum(n_cells_per_condition)
            chunk_indices = np.insert(chunk_indices, 0, 0)
            chunk_labels = ['E in nucleus', 'I in', 'E out', 'I out']
            for i in range(4):
                if n_cells_per_condition[i] == 0:
                    continue
                start_idx = chunk_indices[i]
                end_idx = chunk_indices[i+1]
                if i > 0:
                    ax.vlines(start_idx-0.5, ylims[0], ylims[1], color='k', lw=1)
                ax.hlines(ylims[1]-0.5, start_idx, end_idx-1, color='k', lw=0.75)
                ax.text(np.mean(chunk_indices[i:i+2])-0.5, ylims[1]-0.6, chunk_labels[i],
                        size=axis_label, ha='center', va='bottom')

            # lims and labels
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_bounds(ylims[0], 0)
            ax.set_xlabel('neurons sorted by firing rate', fontsize=axis_label)
            ax.set_ylabel('caches sorted by N seeds in arena', fontsize=axis_label)

            # add a colorbar
            cax = f.add_axes([0.93, 0.4, 0.02, 0.32]) # [left, bottom, width, height]
            cbar = f.colorbar(im1, cax=cax, orientation='vertical')
            cbar.set_label('activity (z-score)', fontsize=tick_label)
            cbar.set_ticks([])
            cbar.ax.text(0.5, -0.05, f'{clims[0]} std', transform=cbar.ax.transAxes,
                            ha='center', va='top', fontsize=tick_label)
            cbar.ax.text(0.5, 1.02, f'{clims[1]} std', transform=cbar.ax.transAxes,
                            ha='center', va='bottom', fontsize=tick_label)

            f.savefig(f'{save_folder}{session_id}_cache_vectors_seeds.png', dpi=600, bbox_inches='tight')
            plt.show()



        ''' Plot the cache vectors for this session - sorted by % caches active '''
        # sort by % caches active & stick everything back together
        exc_in_nucleus_sorted = excitatory_in_nucleus[:, np.argsort(exc_stim_active)]
        inhib_in_nucleus_sorted = inhibitory_in_nucleus[:, np.argsort(inhib_stim_active)]
        excitatory_outside_sorted = excitatory_outside[:, np.argsort(exc_no_stim_active)]
        inhibitory_outside_sorted = inhibitory_outside[:, np.argsort(inhib_no_stim_active)]
        cache_vectors_sorted = np.column_stack([exc_in_nucleus_sorted, 
                                                inhib_in_nucleus_sorted, 
                                                excitatory_outside_sorted, 
                                                inhibitory_outside_sorted])
        
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        clims = [-3, 3]
        im1 = ax.imshow(cache_vectors_sorted, aspect='auto', 
                        cmap='bwr', clim=clims, 
                        interpolation='none')

        # label excitatory/inhibitory and in/out of nucleus
        ylims = ax.get_ylim()
        chunk_indices = np.cumsum(n_cells_per_condition)
        chunk_indices = np.insert(chunk_indices, 0, 0)
        chunk_labels = ['E in nucleus', 'I in', 'E out', 'I out']
        for i in range(4):
            if n_cells_per_condition[i] == 0:
                continue
            start_idx = chunk_indices[i]
            end_idx = chunk_indices[i+1]
            if i > 0:
                ax.vlines(start_idx-0.5, ylims[0], ylims[1], color='k', lw=1)
            ax.hlines(ylims[1]-0.5, start_idx, end_idx-1, color='k', lw=0.75)
            ax.text(np.mean(chunk_indices[i:i+2])-0.5, ylims[1]-0.6, chunk_labels[i],
                    size=axis_label, ha='center', va='bottom')

        # lims and labels
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(ylims[0], 0)
        ax.set_xlabel(f'neurons sorted by % caches active', fontsize=axis_label)
        ax.set_ylabel('cache #', fontsize=axis_label)

        # add a colorbar
        cax = f.add_axes([0.93, 0.4, 0.02, 0.32]) # [left, bottom, width, height]
        cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('activity (z-score)', fontsize=tick_label)
        cbar.set_ticks([])
        cbar.ax.text(0.5, -0.05, f'{clims[0]} std', transform=cbar.ax.transAxes,
                        ha='center', va='top', fontsize=tick_label)
        cbar.ax.text(0.5, 1.02, f'{clims[1]} std', transform=cbar.ax.transAxes,
                        ha='center', va='bottom', fontsize=tick_label)

        f.savefig(f'{save_folder}{session_id}_cache_vectors_active.png', dpi=600, bbox_inches='tight')
        plt.show()

    ''' Save the data across birds '''
    all_birds_active_caches = np.append(all_birds_active_caches, active_cache_frac_all)
    exc_idx_all = np.append(exc_idx_all, exc_idx_bird)
    inhib_idx_all = np.append(inhib_idx_all, inhib_idx_bird)
    all_birds_stim_idx = np.append(all_birds_stim_idx, stim_idx_bird)

    ''' Plot percent active caches split by excitory/inhibitory, in/out '''
    # fig params
    f, ax = plt.subplots(2, 2, figsize=(4, 4), sharex=True)

    # plot the percent of caches that each cell was active for - stim responsive only
    pct_active = active_cache_frac_all*100
    ax[0, 0].hist(pct_active[exc_idx_bird & stim_idx_bird], bins=30)
    ax[1, 0].hist(pct_active[inhib_idx_bird & stim_idx_bird], bins=30)

    # ticks and labels
    ax[1, 0].set_xlabel(f'% caches active')
    ax[0, 0].set_ylabel('N excitatory cells')
    ax[1, 0].set_ylabel('N inhibitory cells')
    ax[0, 0].set_title('cells in nucleus')

    # plot the percent of caches that each cell was active for - stim responsive only
    ax[0, 1].hist(pct_active[exc_idx_bird & ~stim_idx_bird], bins=30)
    ax[1, 1].hist(pct_active[inhib_idx_bird & ~stim_idx_bird], bins=30)

    # ticks and labels
    ax[1, 1].set_xlabel(f'% caches active')
    ax[0, 1].set_title('cells outside nucleus')

    f.savefig(f'{save_folder}pct_caches_active.png', dpi=600, bbox_inches='tight')
    plt.show()


''' Plot percent active caches split by excitory/inhibitory '''
gs_kw = dict(wspace=0.5)
f, ax = plt.subplots(2, 4, figsize=(8, 4), sharex=True, gridspec_kw=gs_kw)

# plot the percent of caches that each cell was active for
pct_active = all_birds_active_caches*100
ax[0, 0].hist(pct_active[exc_idx_all], bins=30)
ax[1, 0].hist(pct_active[inhib_idx_all], bins=30)

# ticks and labels
ax[0, 0].set_ylabel('N excitatory cells')
ax[1, 0].set_ylabel('N inhibitory cells')
ax[0, 0].set_title('all cells')

# putative projection nucleus cells
pct_active = all_birds_active_caches*100
proj_idx = all_birds_cell_loc == 1
ax[0, 1].hist(pct_active[exc_idx_all & proj_idx], bins=30)
ax[1, 1].hist(pct_active[inhib_idx_all & proj_idx], bins=30)
ax[0, 1].set_title('proj. nucleus')

# putative DL cells
pct_active = all_birds_active_caches*100
DL_idx = all_birds_cell_loc == 0
ax[0, 2].hist(pct_active[exc_idx_all & DL_idx], bins=30)
ax[1, 2].hist(pct_active[inhib_idx_all & DL_idx], bins=30)
ax[0, 2].set_title('DL cells')

# putative DMZ cells
pct_active = all_birds_active_caches*100
DMZ_idx = all_birds_cell_loc == 2
ax[0, 3].hist(pct_active[exc_idx_all & DMZ_idx], bins=30)
ax[1, 3].hist(pct_active[inhib_idx_all & DMZ_idx], bins=30)
ax[0, 3].set_title('DMZ/SESN/ETV?')

f.supxlabel(f'% caches active')

f.savefig(f'{save_figs_dir}pct_caches_active.png', dpi=600, bbox_inches='tight')
plt.show()

