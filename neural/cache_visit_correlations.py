import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance

import os 
import sys
from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel
import waveform_analysis
sys.path.append("..//behavior/")
from format_behavior_data import load_behavior_data, get_caches_refined, get_visits_refined
sys.path.append("..//utils/")
import color_utils, helpers
from load_matlab_data import loadmat_sbx
from scipy.io import loadmat

'''
Code for barcode and related analysis, as in Chettih, Mackevicius et al, 2024

Things to plot:
---------------
TODO:
- caches vs retrievals etc (Fig 4) 
- pct caches active vs. position (3D, 2D w/ waveform width)
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# to load the arena objects
# arena_dir = 'C:/Users/ilow1/Documents/code/il_rig_control/arena_alignment/' # laptop
arena_dir = 'C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/' # rig computer
arena_items_file = 'arena_items_2.mat'


''' Data params '''
# for computing/binning distances between perches
convert_norm_to_cm = 13 * 2.54 # conversion factor normalized coordinates to cm
perch_dist_bins = np.asarray([0, 0.01, 0.25, 0.45, 0.62, 0.76, 0.92, 1.12, 1.29, np.inf])
perch_dist_centers = (perch_dist_bins[:-1] + perch_dist_bins[1:]) / 2
perch_dist_centers[-1] = 1.5
perch_dist_centers_cm = perch_dist_centers*convert_norm_to_cm
n_bins = perch_dist_centers.shape[0]

# get the x, y coordinates of each perch center
arena_data = loadmat(f'{arena_dir}{arena_items_file}', squeeze_me=True)
n_sites = arena_data["perch_w_site"].shape[0]
perch_loc = np.zeros((n_sites, 2))
for site in range(n_sites):
    perch_loc[site] = arena_data["perch_w_site"][site]['Centroid']

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

''' Cache analysis across birds '''
# to store data across birds
all_birds_active_caches = np.asarray([])
all_birds_visit_corr = np.asarray([])
all_birds_visit_dist = np.asarray([])
all_birds_cache_corr = np.asarray([])
all_birds_cache_dist = np.asarray([])

# for comparison with RH data
all_birds_visit_corr_raw = np.asarray([])
all_birds_cache_corr_raw = np.asarray([])

# excitatory/inhibitory indices
exc_idx_all = np.asarray([]).astype(bool)
inhib_idx_all = np.asarray([]).astype(bool)

# to filter by stim response
all_birds_stim_resp = np.asarray([]).astype(bool)
all_birds_cell_loc = np.asarray([])

for bird in bird_ids:
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

    ''' Collect the data '''
    # to store results across sessions
    active_cache_frac_all = np.asarray([])
    all_visit_corr = np.asarray([])
    all_visit_dist = np.asarray([])
    all_cache_corr = np.asarray([])
    all_cache_dist = np.asarray([])

    # to filter by stim response
    all_stim_resp = np.asarray([]).astype(bool)

    # to sort by excitory/inhibitory
    exc_idx_bird = np.asarray([]).astype(bool)
    inhib_idx_bird = np.asarray([]).astype(bool)

    bird_idx = bird_ids.index(bird)
    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions:
        print(f'collecting cache-aligned data for {bird}_{session_id}')

        ''' Get the file params '''
        session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
        data_dir = f"{session_dir}/behavior_data/"
        pred_date = data_dict[bird][session_id]['pred_date']
        ephys_id = data_dict[bird][session_id]['ephys_id']
        ephys_folder= f"{root_dir}{bird}/{bird}_{session_id}/{bird}_{ephys_id}/"
        for folder in sorted(os.listdir(ephys_folder)):
            if 'kilosort4' in folder:
                for file in sorted(os.listdir(f"{ephys_folder}{folder}")):
                    if 'waveformStruct' in file:
                        ks_dir = f"{bird}_{ephys_id}/{folder}/"

        ''' Load the frame times '''
        sampling_rate = 30000 # intan
        framet_raw = np.load(f'{data_dir}frame_times.npy')
        framet_raw = np.squeeze(framet_raw)
        dt = np.unique(np.round(np.diff(framet_raw), 2))
        dt = dt[0]

        # align so that 0 is the video start time
        start_t = framet_raw[0]
        frame_t = framet_raw - start_t
        frame_samples = np.append(frame_t, frame_t[-1] + dt)*sampling_rate
        n_frames = frame_t.shape[0]

        ''' Load/format the neural data '''
        # get the cell IDs and raw spike times
        good_clusters, spike_id, spike_samp_raw = get_spike_times(session_dir, ks_dir=ks_dir)
        n_cells = good_clusters.shape[0]
        
        # keep only spikes from within the session
        spike_t = spike_samp_raw - start_t*sampling_rate
        spike_id = spike_id[(spike_t >= 0) & (spike_t <= frame_samples[-1])]
        spike_t = spike_t[(spike_t >= 0) & (spike_t <= frame_samples[-1])]

        # spikes per frame and spike bool
        spike_frame = np.zeros((n_cells, n_frames))
        i = -1
        for c_idx, cell in enumerate(good_clusters):
            i += 1
            spk_times = spike_t[spike_id==cell]       
            spike_frame[i], _ = np.histogram(spk_times, frame_samples)
        spike_bool = spike_frame.astype(bool)

        # instantaneous firing rate
        inst_firing_rate = spike_frame/dt

        # session average firing rate
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]
        avg_firing_rate = 10**log_fr

        # excitatory/inhibitory indices
        exc_idx = data_dict[bird][session_id]['excitatory_idx']
        exc_idx_bird = np.append(exc_idx_bird, exc_idx)
        inhib_idx = data_dict[bird][session_id]['inhibitory_idx']
        inhib_idx_bird = np.append(inhib_idx_bird, inhib_idx)

        ''' Chop up the data by stim response '''
        # index cells by stim responsive channels - binary stim or no stim
        stim_idx_ch = data_dict[bird][session_id]['stim_resp_idx_ch']
        if bird == 'RBY94':
            # get the channel indices for each cell
            ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"
            waveform_struct = load_wf_data(session_dir, ks_dir=ks_dir)
            _, wf_channels, _, ch_names = sort_wf_by_channel('', waveform_struct,
                                                                data_dir=ephys_dir,
                                                                return_ch_names=True)
            wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])
            ch_pos_probe = np.load(f"{session_dir}{ks_dir}channel_positions.npy")       

            # determine if there's a stim response on each cell's channel
            stim_idx_cell = np.zeros(n_cells).astype(bool)
            cell_dv = np.zeros(n_cells)
            cell_ap = np.zeros(n_cells)
            for cell_idx, ch_idx in enumerate(wf_ch_idx):
                stim_idx_cell[cell_idx] = stim_idx_ch[ch_idx]

                # save the depth on probe for each cell
                cell_dv[cell_idx] = ch_pos_probe[ch_idx, -1]

                # get the shank index for each cell
                cell_ap[cell_idx] = ch_pos_probe[ch_idx, 0]
            A_idx = cell_ap < 100
            B_idx = cell_ap >= 100
        else:
            # get the positions of stim responsive channels
            ch_pos = data_dict[bird][session_id]['channel_pos']
            stim_pos = ch_pos[stim_idx_ch]

            # match each cell to its channel position
            cell_pos = data_dict[bird][session_id]['cell_pos']
            n_cells = cell_pos.shape[0]
            stim_idx_cell = np.zeros(n_cells).astype(bool)
            for cell_idx, this_pos in enumerate(cell_pos):
                stim_idx_cell[cell_idx] = np.any(np.all(stim_pos == this_pos, axis=1))
                cell_dv = cell_pos[:, -1]

            # get the shank index for each cell
            _, ap_idx = np.unique(cell_pos[:, 1], return_inverse=True)
            A_idx = ap_idx >= 3
            B_idx = ap_idx < 3
        all_stim_resp = np.append(all_stim_resp, stim_idx_cell)

        # index cells by location in the brain
        # 1 = putative projection nucleus, 0 = put. DL (dorsal/lateral), 2 = put. ventral subiculum/SESN/DMZ (ventral/medial)
        nucleus_dvs = data_dict[bird][session_id]['nucleus_dvs']
        A_nuc_lims = nucleus_dvs[0]
        B_nuc_lims = nucleus_dvs[1]

        DL_idx = np.zeros(n_cells).astype(bool)
        DL_idx[A_idx] = cell_dv[A_idx] < A_nuc_lims[0]
        DL_idx[B_idx] = cell_dv[B_idx] < B_nuc_lims[0]

        DMZ_idx = np.zeros(n_cells).astype(bool)
        DMZ_idx[A_idx] = cell_dv[A_idx] > A_nuc_lims[1]
        DMZ_idx[B_idx] = cell_dv[B_idx] > B_nuc_lims[1]

        if bird == 'RBY94': # likely entire B shank was medial of the nucleus
            DMZ_idx[B_idx] = True
            DL_idx[B_idx] = False

        proj_idx = np.abs((DL_idx + DMZ_idx) - 1).astype(bool)

        cell_loc_idx = np.full(n_cells, np.nan)
        cell_loc_idx[DL_idx] = 0
        cell_loc_idx[proj_idx] = 1
        cell_loc_idx[DMZ_idx] = 2

        all_birds_cell_loc = np.append(all_birds_cell_loc, cell_loc_idx)

        ''' Load and format behavior data '''
        # load behavioral data
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
        seed_struct, count_data = load_behavior_data(data_dir)

        # get cache interactions and visits
        cache_onsets, cache_offsets, cache_ids = get_caches_refined(count_data, seed_struct, n_frames)
        visit_onsets, visit_offsets, visit_ids = get_visits_refined(count_data, n_frames)

        # adjust for pythonic indexing
        cache_ids -= 1
        visit_ids -= 1

        # exclude visits to feeder perches
        visit_onsets = visit_onsets[visit_ids < n_sites]
        visit_offsets = visit_offsets[visit_ids < n_sites]
        visit_ids = visit_ids[visit_ids < n_sites]

        n_caches = cache_onsets.shape[0]
        n_visits = visit_onsets.shape[0]

        # get the x, y coordinates of each cache/visit
        cache_loc = np.zeros((n_caches, 2))
        for cache_idx, cache_id in enumerate(cache_ids):
            cache_loc[cache_idx] = perch_loc[cache_id]
            
        visit_loc = np.zeros((n_visits, 2))
        for visit_idx, visit_id in enumerate(visit_ids):
            visit_loc[visit_idx] = perch_loc[visit_id]


        ''' Average neural activity during each visit and cache '''
        # account for long caches as in SC, EM 2024
        long_thresh = 2 # seconds
        long_window = int(long_thresh/2/dt) # frames

        # get average activity during cache window
        avg_cache = np.zeros((n_cells, n_caches))
        for i, (cache_on, cache_off) in enumerate(zip(cache_onsets, cache_offsets)):
            if cache_off - cache_on < long_thresh:
                spike_count = np.sum(spike_frame[:, cache_on:cache_off], axis=1)
                occupancy = dt*(cache_off-cache_on)
                avg_cache[:, i] = spike_count/occupancy
            else:
                begin_count = np.sum(spike_frame[:, cache_on:cache_on+long_window], axis=1)
                end_count = np.sum(spike_frame[:, cache_off-long_window:cache_off], axis=1)
                spike_count = begin_count + end_count
                avg_cache[:, i] = spike_count/long_thresh

        # get average activity during visit window
        avg_visit = np.zeros((n_cells, n_visits))
        for i, (visit_on, visit_off) in enumerate(zip(visit_onsets, visit_offsets)):
            avg_visit[:, i] = np.nanmean(spike_frame[:, visit_on:visit_off], axis=1)

        # fraction of caches w/ FR > session avg
        active_cache = np.zeros_like(avg_cache)
        for c_idx in range(n_cells):
            active_cache[c_idx] = avg_cache[c_idx] > avg_firing_rate[c_idx]
        active_cache_frac = np.sum(active_cache, axis=1) / n_caches

        # collect across sessions
        active_cache_frac_all = np.append(active_cache_frac_all, active_cache_frac)


        ''' Normalize activity for population analysis '''
        # get the baseline rate for each cell (running 30min avg activity)
        baseline_window = 30 # minutes
        moving_avg_fr = np.zeros_like(inst_firing_rate)
        for cell in range(n_cells):
            moving_avg_fr[cell] = helpers.moving_avg(inst_firing_rate[cell], window=baseline_window)

        # get the standard deviation (regularize by adding 0.6 Hz)
        st_dev_fr = stats.tstd(inst_firing_rate, axis=1) + 0.6
        assert st_dev_fr.shape[0] == n_cells

        # normalize
        norm_fr = inst_firing_rate.copy()
        norm_fr -= moving_avg_fr
        for cell in range(n_cells):
            norm_fr[cell] /= st_dev_fr[cell]


        ''' Compute the population activity vectors '''
        # collect all the population vectors for this session
        visit_vectors_raw = np.zeros((n_visits, n_cells))
        for i, (vs, ve) in enumerate(zip(visit_onsets, visit_offsets)):
            visit_vectors_raw[i] = np.mean(norm_fr[:, vs:ve], axis=1)

        cache_vectors_raw = np.zeros((n_caches, n_cells))    
        for i, (cs, ce) in enumerate(zip(cache_onsets, cache_offsets)):
            cache_vectors_raw[i] = np.mean(norm_fr[:, cs:ce], axis=1)

        # only keep cells in the projection nucleus
        visit_vectors_raw = visit_vectors_raw[:, stim_idx_cell&exc_idx]
        cache_vectors_raw = cache_vectors_raw[:, stim_idx_cell&exc_idx]

        # compute avg population vectors for all visits, caches
        avg_visit_vector = np.mean(visit_vectors_raw, axis=0, keepdims=True)
        avg_cache_vector = np.mean(cache_vectors_raw, axis=0, keepdims=True)

        # subtract the means across all events
        visit_vectors = visit_vectors_raw #- avg_visit_vector
        cache_vectors = cache_vectors_raw #- avg_cache_vector

        ''' Get the physical and neural distance between each pair of events '''
        # visit-visit correlation and distances
        visit_corr = 1 - distance.pdist(visit_vectors, 'correlation')
        visit_dist = distance.pdist(visit_loc)
            
        # cache-cache correlation and distances
        cache_corr = 1 - distance.pdist(cache_vectors, 'correlation')
        cache_dist = distance.pdist(cache_loc)

        # collect across sessions
        all_visit_corr = np.append(all_visit_corr, visit_corr)
        all_visit_dist = np.append(all_visit_dist, visit_dist)
        all_cache_corr = np.append(all_cache_corr, cache_corr)
        all_cache_dist = np.append(all_cache_dist, cache_dist)

        # store the raw correlations for comparison with RH
        all_birds_visit_corr_raw = np.append(all_birds_visit_corr_raw, 1 - distance.pdist(visit_vectors_raw, 'correlation'))
        all_birds_cache_corr_raw = np.append(all_birds_cache_corr_raw, 1 - distance.pdist(cache_vectors_raw, 'correlation'))

        ''' Save data for future use '''
        # barcode_dict = {}
        # barcode_dict['cache_vectors'] = cache_vectors
        # barcode_dict['visit_vectors'] = visit_vectors
        # barcode_dict['cache_loc'] = cache_loc
        # barcode_dict['visit_loc'] = visit_loc
        # barcode_dict['active_cache_frac'] = active_cache_frac
        # data_dict[bird][session_id]['barcode_dict'] = barcode_dict

        ''' Plot the cache vectors for this session '''
        # # reorganize to cluster excitatory and inhibitory cells
        # cache_vectors_sorted = np.column_stack((cache_vectors[:, exc_idx], 
        #                                         cache_vectors[:, inhib_idx]))
        # n_excite = np.sum(exc_idx)
        # n_inhib = np.sum(inhib_idx)

        # # plot firing changes by caches
        # f, ax = plt.subplots(1, 1, figsize=(6, 4))
        # im1 = ax.imshow(cache_vectors, aspect='auto', 
        #                 cmap='bwr', clim=[-5, 5], 
        #                 interpolation='none')

        # # label excitatory/inhibitory
        # ylims = ax.get_ylim()
        # ax.vlines(n_excite, ylims[0], ylims[1], color='k', lw=1)
        # ax.hlines(ylims[1]-0.5, 0.5, n_excite-0.5, color='k', lw=0.75)
        # ax.hlines(ylims[1]-0.5, n_cells-n_inhib+0.5, n_cells-0.5, color='k', lw=0.75)

        # # lims and labels
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_bounds(ylims[0], 0)
        # ax.set_xlabel('neuron #', fontsize=axis_label)
        # ax.set_ylabel('cache #', fontsize=axis_label)
        # ax.text(n_excite//2, ylims[1]-0.6, 'excitatory',
        #         size=axis_label, ha='center', va='bottom')
        # ax.text(n_excite + (n_inhib//2), ylims[1]-0.6, 'inhib.',
        #         size=axis_label, ha='center', va='bottom')

        # # add a colorbar
        # cax = f.add_axes([0.93, 0.5, 0.02, 0.32]) # [left, bottom, width, height]
        # cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        # cbar.set_label('activity (z-score)', fontsize=tick_label)
        # cbar.set_ticks([])
        # cbar.ax.text(0.5, -0.05, '-5 std', transform=cbar.ax.transAxes,
        #                 ha='center', va='top', fontsize=tick_label)
        # cbar.ax.text(0.5, 1.02, '5 std', transform=cbar.ax.transAxes,
        #                 ha='center', va='bottom', fontsize=tick_label)

        # f.savefig(f'{save_folder}{session_id}_cache_vectors.png', dpi=600, bbox_inches='tight')
        # plt.show()



    ''' Store data for comparison across birds '''
    all_birds_active_caches = np.append(all_birds_active_caches, active_cache_frac_all)
    all_birds_visit_corr = np.append(all_birds_visit_corr, all_visit_corr)
    all_birds_visit_dist = np.append(all_birds_visit_dist, all_visit_dist)
    all_birds_cache_corr = np.append(all_birds_cache_corr, all_cache_corr)
    all_birds_cache_dist = np.append(all_birds_cache_dist, all_cache_dist)

    ''' Store indices to filter across birds '''
    exc_idx_all = np.append(exc_idx_all, exc_idx_bird)
    inhib_idx_all = np.append(inhib_idx_all, inhib_idx_bird)
    all_birds_stim_resp = np.append(all_birds_stim_resp, all_stim_resp)


    ''' Plot percent active caches split by excitory/inhibitory '''
    # fig params
    f, ax = plt.subplots(2, 2, figsize=(4, 4), sharex=True)

    # plot the percent of caches that each cell was active for
    pct_active = active_cache_frac_all*100
    ax[0, 0].hist(pct_active[exc_idx_bird], bins=30)
    ax[1, 0].hist(pct_active[inhib_idx_bird], bins=30)

    # ticks and labels
    ax[1, 0].set_xlabel(f'% caches active')
    ax[0, 0].set_ylabel('N excitatory cells')
    ax[1, 0].set_ylabel('N inhibitory cells')
    ax[0, 0].set_title('all cells')

    # plot the percent of caches that each cell was active for - stim responsive only
    ax[0, 1].hist(pct_active[exc_idx_bird & all_stim_resp], bins=30)
    ax[1, 1].hist(pct_active[inhib_idx_bird & all_stim_resp], bins=30)

    # ticks and labels
    ax[1, 1].set_xlabel(f'% caches active')
    ax[0, 1].set_title('cells in nucleus')

    f.savefig(f'{save_folder}pct_caches_active.png', dpi=600, bbox_inches='tight')
    plt.show()


    ''' Bin the correlations by distance and calculate the mean/sem '''
    # remove nans
    keep_idx_vists = np.abs(np.isnan(all_visit_corr)-1).astype(bool)
    all_visit_corr = all_visit_corr[keep_idx_vists]
    all_visit_dist = all_visit_dist[keep_idx_vists]
    keep_idx_caches = np.abs(np.isnan(all_cache_corr)-1).astype(bool)
    all_cache_corr = all_cache_corr[keep_idx_caches]
    all_cache_dist = all_cache_dist[keep_idx_caches]

    # average for visits
    visit_bin_idx = np.digitize(all_visit_dist, perch_dist_bins)-1
    avg_visit_corr = np.zeros(n_bins)
    for b_idx in range(n_bins):
        avg_visit_corr[b_idx] = np.mean(all_visit_corr[visit_bin_idx==b_idx]) 

    # normalize by visit-visit correlation at the same site
    norm_factor = np.max(avg_visit_corr)
    norm_factor = 1
    avg_visit_corr_norm = avg_visit_corr / norm_factor

    # sem for visits
    sem_visit_corr = np.zeros(n_bins)
    for b_idx in range(n_bins):
        sem_visit_corr[b_idx] = stats.sem(all_visit_corr[visit_bin_idx==b_idx] / norm_factor)   

    # average and sem for caches
    cache_bin_idx = np.digitize(all_cache_dist, perch_dist_bins)-1
    avg_cache_corr = np.zeros(n_bins)
    sem_cache_corr = np.zeros(n_bins)
    for b_idx in range(n_bins):
        avg_cache_corr[b_idx] = np.mean(all_cache_corr[cache_bin_idx==b_idx]) 
        sem_cache_corr[b_idx] = stats.sem(all_cache_corr[cache_bin_idx==b_idx] / norm_factor) 
    avg_cache_corr_norm = avg_cache_corr / norm_factor


    ''' Plot cache-cache and visit-visit correlations '''
    f, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    ax[0].scatter(perch_dist_centers_cm, avg_visit_corr_norm, c='xkcd:dark gray')
    ax[0].vlines(perch_dist_centers_cm, 
                 avg_visit_corr_norm-sem_visit_corr,
                 avg_visit_corr_norm+sem_visit_corr,
                 color='xkcd:dark gray', lw=1)

    ax[1].scatter(perch_dist_centers_cm, avg_cache_corr_norm, c='xkcd:orange')
    ax[1].vlines(perch_dist_centers_cm, 
                 avg_cache_corr_norm-sem_cache_corr,
                 avg_cache_corr_norm+sem_cache_corr,
                 color='xkcd:orange', lw=1)

    ax[0].set_ylabel('correlation (norm.)')
    ax[0].set_xlabel('distance (cm)')
    ax[1].set_xlabel('distance (cm)')
    ax[0].set_title('visit vs. visit')
    ax[1].set_title('cache vs. cache')

    f.savefig(f'{save_folder}cache_visit_corr_raw.png', dpi=600, bbox_inches='tight')
    plt.show()


''' Across all birds '''
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


''' Bin the correlations by distance and calculate the mean/sem '''
# remove nans
keep_idx_vists = np.abs(np.isnan(all_birds_visit_corr)-1).astype(bool)
all_birds_visit_corr = all_birds_visit_corr[keep_idx_vists]
all_birds_visit_dist = all_birds_visit_dist[keep_idx_vists]
keep_idx_caches = np.abs(np.isnan(all_birds_cache_corr)-1).astype(bool)
all_birds_cache_corr = all_birds_cache_corr[keep_idx_caches]
all_birds_cache_dist = all_birds_cache_dist[keep_idx_caches]

# average for visits
visit_bin_idx = np.digitize(all_birds_visit_dist, perch_dist_bins)-1
avg_visit_corr = np.zeros(n_bins)
for b_idx in range(n_bins):
    avg_visit_corr[b_idx] = np.mean(all_birds_visit_corr[visit_bin_idx==b_idx]) 

# normalize by visit-visit correlation at the same site
norm_factor = np.max(avg_visit_corr)
norm_factor = 1
avg_visit_corr_norm = avg_visit_corr / norm_factor
print(f'max visit-visit correlation = {np.round(norm_factor, 4)}')

# sem for visits
sem_visit_corr = np.zeros(n_bins)
for b_idx in range(n_bins):
    sem_visit_corr[b_idx] = stats.sem(all_birds_visit_corr[visit_bin_idx==b_idx] / norm_factor)   

# average and sem for caches
cache_bin_idx = np.digitize(all_birds_cache_dist, perch_dist_bins)-1
avg_cache_corr = np.zeros(n_bins)
sem_cache_corr = np.zeros(n_bins)
for b_idx in range(n_bins):
    avg_cache_corr[b_idx] = np.mean(all_birds_cache_corr[cache_bin_idx==b_idx]) 
    sem_cache_corr[b_idx] = stats.sem(all_birds_cache_corr[cache_bin_idx==b_idx] / norm_factor) 
avg_cache_corr_norm = avg_cache_corr / norm_factor


''' Plot cache-cache and visit-visit correlations '''
f, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

ax[0].scatter(perch_dist_centers_cm, avg_visit_corr_norm, c='xkcd:dark gray')
ax[0].vlines(perch_dist_centers_cm, 
             avg_visit_corr_norm-sem_visit_corr,
             avg_visit_corr_norm+sem_visit_corr,
             color='xkcd:dark gray', lw=1)

ax[1].scatter(perch_dist_centers_cm, avg_cache_corr_norm, c='xkcd:orange')
ax[1].vlines(perch_dist_centers_cm, 
             avg_cache_corr_norm-sem_cache_corr,
             avg_cache_corr_norm+sem_cache_corr,
             color='xkcd:orange', lw=1)

ax[0].set_ylabel('correlation (norm.)')
ax[0].set_ylabel('correlation (raw)')
ax[0].set_xlabel('distance (cm)')
ax[1].set_xlabel('distance (cm)')
ax[0].set_title('visit vs. visit')
ax[1].set_title('cache vs. cache')

f.savefig(f'{save_figs_dir}cache_visit_corr_raw.png', dpi=600, bbox_inches='tight')
plt.show()


# ''' More raw data for comparison to RH '''
# # remove nans
# keep_idx_vists = np.abs(np.isnan(all_birds_visit_corr_raw)-1).astype(bool)
# all_birds_visit_corr = all_birds_visit_corr_raw[keep_idx_vists]
# keep_idx_caches = np.abs(np.isnan(all_birds_cache_corr_raw)-1).astype(bool)
# all_birds_cache_corr = all_birds_cache_corr_raw[keep_idx_caches]

# # average for visits
# visit_bin_idx = np.digitize(all_birds_visit_dist, perch_dist_bins)-1
# avg_visit_corr = np.zeros(n_bins)
# for b_idx in range(n_bins):
#     avg_visit_corr[b_idx] = np.mean(all_birds_visit_corr[visit_bin_idx==b_idx]) 

# # normalize by visit-visit correlation at the same site
# norm_factor = np.max(avg_visit_corr)
# avg_visit_corr_norm = avg_visit_corr #/ norm_factor
# print(norm_factor)

# # sem for visits
# sem_visit_corr = np.zeros(n_bins)
# for b_idx in range(n_bins):
#     sem_visit_corr[b_idx] = stats.sem(all_birds_visit_corr[visit_bin_idx==b_idx])# / norm_factor)   

# # average and sem for caches
# cache_bin_idx = np.digitize(all_birds_cache_dist, perch_dist_bins)-1
# avg_cache_corr = np.zeros(n_bins)
# sem_cache_corr = np.zeros(n_bins)
# for b_idx in range(n_bins):
#     avg_cache_corr[b_idx] = np.mean(all_birds_cache_corr[cache_bin_idx==b_idx]) 
#     sem_cache_corr[b_idx] = stats.sem(all_birds_cache_corr[cache_bin_idx==b_idx])# / norm_factor) 
# avg_cache_corr_norm = avg_cache_corr #/ norm_factor

# ''' Plot cache-cache and visit-visit correlations '''
# f, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

# ax[0].scatter(perch_dist_centers_cm, avg_visit_corr_norm, c='xkcd:dark gray')
# ax[0].vlines(perch_dist_centers_cm, 
#              avg_visit_corr_norm-sem_visit_corr,
#              avg_visit_corr_norm+sem_visit_corr,
#              color='xkcd:dark gray', lw=1)

# ax[1].scatter(perch_dist_centers_cm, avg_cache_corr_norm, c='xkcd:orange')
# ax[1].vlines(perch_dist_centers_cm, 
#              avg_cache_corr_norm-sem_cache_corr,
#              avg_cache_corr_norm+sem_cache_corr,
#              color='xkcd:orange', lw=1)

# ax[0].set_ylabel('correlation (norm.)')
# ax[0].set_xlabel('distance (cm)')
# ax[1].set_xlabel('distance (cm)')
# ax[0].set_title('visit vs. visit')
# ax[1].set_title('cache vs. cache')

# f.savefig(f'{save_figs_dir}cache_visit_corr_raw.png', dpi=600, bbox_inches='tight')
# plt.show()


# save the updated dictionary
np.save(data_file, data_dict)