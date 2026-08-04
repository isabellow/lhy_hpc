import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

import os 
import sys
sys.path.append("..//behavior/")
from format_behavior_data import dist_binned_mean_sem
'''
Code for correlating all pairs of population vectors, as in Chettih, Mackevicius et al, 2024

TODO:
- caches vs checks
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

# load the data dictionary and get bird ids
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9

# for computing/binning distances between perches
convert_norm_to_cm = 13 * 2.54 # conversion factor normalized coordinates to cm
perch_dist_bins = np.asarray([0, 0.01, 0.25, 0.45, 0.62, 0.76, 0.92, 1.12, 1.29, np.inf])
perch_dist_centers = (perch_dist_bins[:-1] + perch_dist_bins[1:]) / 2
perch_dist_centers[-1] = 1.5
perch_dist_centers_cm = perch_dist_centers*convert_norm_to_cm

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
all_birds_visit_corr = np.asarray([])
all_birds_visit_dist = np.asarray([])
all_birds_cache_corr = np.asarray([])
all_birds_cache_dist = np.asarray([])

all_birds_cache_retrieve_corr = np.asarray([])
all_birds_cache_retrieve_dist = np.asarray([])
all_birds_cache_visit_corr = np.asarray([])
all_birds_cache_visit_dist = np.asarray([])

# excitatory/inhibitory indices
exc_idx_all = np.asarray([]).astype(bool)
inhib_idx_all = np.asarray([]).astype(bool)

for bird in bird_ids:
    print(f'collecting cache-retrieval correlations for {bird}')
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

    ''' Collect the data '''
    # to store results across sessions
    all_visit_corr = np.asarray([])
    all_visit_dist = np.asarray([])
    all_cache_corr = np.asarray([])
    all_cache_dist = np.asarray([])

    all_cache_retrieve_corr = np.asarray([])
    all_cache_retrieve_dist = np.asarray([])
    all_cache_visit_corr = np.asarray([])
    all_cache_visit_dist = np.asarray([])

    behavior_sessions = all_behavior_sessions[bird_idx]
    for session_id in behavior_sessions:        
        ''' Load and format behavior data '''
        barcode_dict = data_dict[bird][session_id]['barcode_dict']

        # x, y coordinates of each event
        cache_loc = barcode_dict['cache_loc']
        retrieve_loc = barcode_dict['retrieve_loc']
        visit_loc = barcode_dict['visit_loc']

        # population vectors
        visit_vectors = barcode_dict['visit_vectors']
        cache_vectors = barcode_dict['cache_vectors']
        retrieve_vectors = barcode_dict['retrieve_vectors']

        ''' Get the physical and neural distance between each pair of same-type events '''
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

        ''' Get the physical and neural distance between each pair of different-type events '''
        # cache-retrieve correlation and distances
        cache_retrieve_corr = 1 - distance.cdist(cache_vectors, retrieve_vectors, metric='correlation')
        cache_retrieve_dist = distance.cdist(cache_loc, retrieve_loc)

        # cache-visit correlation and distances
        cache_visit_corr = 1 - distance.cdist(cache_vectors, visit_vectors, metric='correlation')
        cache_visit_dist = distance.cdist(cache_loc, visit_loc)

        # collect across sessions
        all_cache_retrieve_corr = np.append(all_cache_retrieve_corr, cache_retrieve_corr)
        all_cache_retrieve_dist = np.append(all_cache_retrieve_dist, cache_retrieve_dist)
        all_cache_visit_corr = np.append(all_cache_visit_corr, cache_visit_corr)
        all_cache_visit_dist = np.append(all_cache_visit_dist, cache_visit_dist)

    ''' Store data for comparison across birds '''
    all_birds_visit_corr = np.append(all_birds_visit_corr, all_visit_corr)
    all_birds_visit_dist = np.append(all_birds_visit_dist, all_visit_dist)
    all_birds_cache_corr = np.append(all_birds_cache_corr, all_cache_corr)
    all_birds_cache_dist = np.append(all_birds_cache_dist, all_cache_dist)

    all_birds_cache_retrieve_corr = np.append(all_birds_cache_retrieve_corr, all_cache_retrieve_corr)
    all_birds_cache_retrieve_dist = np.append(all_birds_cache_retrieve_dist, all_cache_retrieve_dist)
    all_birds_cache_visit_corr = np.append(all_birds_cache_visit_corr, all_cache_visit_corr)
    all_birds_cache_visit_dist = np.append(all_birds_cache_visit_dist, all_cache_visit_dist)


    ''' Same-to-same correlation within bird '''
    # visit-visit
    avg_visit_corr, sem_visit_corr = dist_binned_mean_sem(all_visit_corr, 
                                                            all_visit_dist,
                                                            perch_dist_bins)

    # cache-cache
    avg_cache_corr, sem_cache_corr = dist_binned_mean_sem(all_cache_corr, 
                                                            all_cache_dist,
                                                            perch_dist_bins)

    # normalize so same site visit-visit = 1
    norm_factor = np.max(avg_visit_corr)
    avg_visit_corr_norm = avg_visit_corr / norm_factor
    sem_visit_corr = sem_visit_corr / norm_factor
    avg_cache_corr_norm = avg_cache_corr / norm_factor
    sem_cache_corr = sem_cache_corr / norm_factor

    # plot it
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

    f.savefig(f'{save_folder}cache_visit_corr.png', dpi=600, bbox_inches='tight')
    plt.show()

    ''' Different events correlation within bird '''
    # cache-retrieve
    avg_cache_ret_corr, sem_cache_ret_corr = dist_binned_mean_sem(all_cache_retrieve_corr, 
                                                                    all_cache_retrieve_dist,
                                                                    perch_dist_bins)

    # cache-visit
    avg_cache_visit_corr, sem_cache_visit_corr = dist_binned_mean_sem(all_cache_visit_corr, 
                                                                        all_cache_visit_dist,
                                                                        perch_dist_bins)

    # normalize so same site visit-visit = 1
    avg_cache_ret_corr = avg_cache_ret_corr / norm_factor
    sem_cache_ret_corr = sem_cache_ret_corr / norm_factor
    avg_cache_visit_corr = avg_cache_visit_corr / norm_factor
    sem_cache_visit_corr = sem_cache_visit_corr / norm_factor

    # plot it
    f, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    ax[0].scatter(perch_dist_centers_cm, avg_cache_ret_corr, c='xkcd:purple')
    ax[0].vlines(perch_dist_centers_cm, 
                 avg_cache_ret_corr-sem_cache_ret_corr,
                 avg_cache_ret_corr+sem_cache_ret_corr,
                 color='xkcd:purple', lw=1)

    ax[1].scatter(perch_dist_centers_cm, avg_cache_visit_corr, c='xkcd:dark gray')
    ax[1].vlines(perch_dist_centers_cm, 
                 avg_cache_visit_corr-sem_cache_visit_corr,
                 avg_cache_visit_corr+sem_cache_visit_corr,
                 color='xkcd:dark gray', lw=1)

    ax[0].set_ylabel('correlation (norm.)')
    ax[0].set_xlabel('distance (cm)')
    ax[1].set_xlabel('distance (cm)')
    ax[0].set_title('cache vs. retrieval')
    ax[1].set_title('cache vs. visit')

    f.savefig(f'{save_folder}cache_ret_corr.png', dpi=600, bbox_inches='tight')
    plt.show()


''' Same-to-same correlations across birds '''
# visit-visit
avg_visit_corr, sem_visit_corr = dist_binned_mean_sem(all_birds_visit_corr, 
                                                        all_birds_visit_dist,
                                                        perch_dist_bins)

# cache-cache
avg_cache_corr, sem_cache_corr = dist_binned_mean_sem(all_birds_cache_corr, 
                                                        all_birds_cache_dist,
                                                        perch_dist_bins)

# normalize so same site visit-visit = 1
norm_factor = np.max(avg_visit_corr)
avg_visit_corr_norm = avg_visit_corr / norm_factor
sem_visit_corr = sem_visit_corr / norm_factor
avg_cache_corr_norm = avg_cache_corr / norm_factor
sem_cache_corr = sem_cache_corr / norm_factor

# plot it
f, ax = plt.subplots(1, 2, figsize=(6, 3))

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

# axis ticks and limits
ax[0].set_ylim(-0.5, 4.25)
ax[0].set_yticks(np.arange(5).astype(int))
ax[1].set_ylim(-0.5, 4.25)
ax[1].set_yticks(np.arange(5).astype(int))

# axis labels
ax[0].set_ylabel('correlation (norm.)')
ax[0].set_xlabel('distance (cm)')
ax[1].set_xlabel('distance (cm)')
ax[0].set_title('visit vs. visit')
ax[1].set_title('cache vs. cache')

f.savefig(f'{save_figs_dir}cache_visit_corr.png', dpi=600, bbox_inches='tight')
plt.show()


''' Different events correlation within bird '''
# cache-retrieve
avg_cache_ret_corr, sem_cache_ret_corr = dist_binned_mean_sem(all_birds_cache_retrieve_corr, 
                                                                all_birds_cache_retrieve_dist,
                                                                perch_dist_bins)

# cache-visit
avg_cache_visit_corr, sem_cache_visit_corr = dist_binned_mean_sem(all_birds_cache_visit_corr, 
                                                                    all_birds_cache_visit_dist,
                                                                    perch_dist_bins)

# normalize so same site visit-visit = 1
avg_cache_ret_corr = avg_cache_ret_corr / norm_factor
sem_cache_ret_corr = sem_cache_ret_corr / norm_factor
avg_cache_visit_corr = avg_cache_visit_corr / norm_factor
sem_cache_visit_corr = sem_cache_visit_corr / norm_factor

# plot it
f, ax = plt.subplots(1, 2, figsize=(6, 3))

ax[0].scatter(perch_dist_centers_cm, avg_cache_ret_corr, c='xkcd:purple')
ax[0].vlines(perch_dist_centers_cm, 
             avg_cache_ret_corr-sem_cache_ret_corr,
             avg_cache_ret_corr+sem_cache_ret_corr,
             color='xkcd:purple', lw=1)

ax[1].scatter(perch_dist_centers_cm, avg_cache_visit_corr, c='xkcd:dark gray')
ax[1].vlines(perch_dist_centers_cm, 
             avg_cache_visit_corr-sem_cache_visit_corr,
             avg_cache_visit_corr+sem_cache_visit_corr,
             color='xkcd:dark gray', lw=1)

# axis ticks and limits
ax[0].set_ylim(-0.5, 4.25)
ax[0].set_yticks(np.arange(5).astype(int))
ax[1].set_ylim(-0.5, 4.25)
ax[1].set_yticks(np.arange(5).astype(int))

# axis labels
ax[0].set_ylabel('correlation (norm.)')
ax[0].set_xlabel('distance (cm)')
ax[1].set_xlabel('distance (cm)')
ax[0].set_title('cache vs. retrieval')
ax[1].set_title('cache vs. visit')

f.savefig(f'{save_figs_dir}cache_ret_corr.png', dpi=600, bbox_inches='tight')
plt.show()