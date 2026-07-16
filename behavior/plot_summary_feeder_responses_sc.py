import numpy as np
import pandas as pd
from scipy import stats

import os 
import sys
sys.path.append("..//utils/")
import color_utils, helpers
from load_matlab_data import loadmat_sbx
import process_SC_data
sys.path.append("..//neural/")
from format_waveform_data import pop_normalize
from format_behavior_data import load_behavior_data, get_feeder_ints, get_feeder_periods, classify_feeder_ints, get_feeder_departure_bounds, get_foot_angle
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

''' File Paths '''
root_dir = "Z:/Isabel/data/Grid Caching Data/"
save_figs_dir = f"../figures/basic_neural_analysis_sc/"
if os.path.isdir(save_figs_dir):
    print('save directory exists')
else:
    os.mkdir(save_figs_dir)

''' Bird and session list '''
data_dict = {}

all_dirs = sorted(os.listdir(root_dir))
ignore_files = ['arena_im_1_1.mat', '.DS_Store']
session_dirs = []
for s_dir in all_dirs:
    if s_dir in ignore_files:
        continue
    elif '_' in s_dir:
        session_dirs.append(s_dir)

bird_ids = []
for session_folder in session_dirs:
    parts = session_folder.split('_')
    bird = parts[0]
    session_id = f'{parts[1]}_{parts[2]}'
    if bird in bird_ids:
        data_dict[bird][session_id] = {}
    else:
        data_dict[bird] = {}
        data_dict[bird][session_id] = {}
        bird_ids.append(bird)

''' Plotting params '''
# feeder visit time windows
t_pre_sec = 1
t_begin_sec = 0.5
t_end_sec = 0.5
t_post_sec = 1

# frame windows for average offset activity (start, end) - TODO modify as needed
offset_window = {}
for bird in bird_ids:
    offset_window[bird] = np.asarray([5, 10])
offset_window['all'] = np.asarray([5, 10])

# frame windows for average onset activity (start, end) - TODO modify as needed
onset_window = {}
for bird in bird_ids:
    onset_window[bird] = np.asarray([10, 5])
onset_window['all'] = np.asarray([5, 10])

# font sizes
title_size = 14
axis_label = 12
tick_label = 9

# data params
dt = 1/60 # SC data collected at 60Hz
fr_thresh = 0.05

all_birds_feeder_responses = []
for bird in bird_ids:
    print(f'\nplotting feeder responses for {bird}')
    
    ''' Define/create the save folder'''
    save_dir = f"{save_figs_dir}/{bird}/"
    if os.path.isdir(save_dir):
        print('save directory exists')
    else:
        os.mkdir(save_dir)
    save_figs = f"{save_dir}/feeder_responses/"
    if os.path.isdir(save_figs):
        print('save folder exists')
    else:
        os.mkdir(save_figs)

    ''' Collect data across sessions '''
    n_cells_total = 0
    # n_cells_upstart = 0 # todo maybe add back in an automated way
    # n_cells_upend = 0 # todo maybe add back in an automated way
    bird_feeder_responses = np.asarray([])    
    behavior_sessions = data_dict[bird].keys()
    for session_id in behavior_sessions:
        data_dir = f"{root_dir}{bird}_{session_id}/"

        ''' Load the aligned neural data'''
        spike_fr, excitatory_idx, inhibitory_idx = process_SC_data.load_neural_data(data_dir, min_rate=fr_thresh)

        # keep only excitatory units
        spike_fr = spike_fr[excitatory_idx]
        n_cells, n_frames = spike_fr.shape

        # normalize for population analysis
        norm_fr = pop_normalize(spike_fr, dt=dt)

        ''' Load and format behavior data '''
        seed_struct, count_data = load_behavior_data(data_dir)

        # get the feeder interactions + classify as open/closed
        feeder_int_start, feeder_int_end, feeder_idx = get_feeder_ints(count_data, use_beak=False, 
                                                                        feeder_perches=np.asarray([-4, -3, -2, -1]))
        # feeder_open_times, feeder_close_times = get_feeder_periods(session_info_file, bird, session_id) # TODO! for my data too!!
        # feeder_status = classify_feeder_ints(feeder_int_start, feeder_int_end, feeder_open_times, feeder_close_times, frame_rate=fps)
        n_feeder_int = feeder_int_start.shape[0]

        print(f'{bird}_{session_id}: {n_cells} cells, {n_feeder_int} feeder interactions')

        # convert feeder time windows to frames
        t_pre = int(t_pre_sec/dt)
        t_begin = int(t_begin_sec/dt)
        t_end = int(t_end_sec/dt)
        t_post = int(t_post_sec/dt)
        n_timepoints = t_pre + t_begin + t_end + t_post

        ''' Activity of all cells relative to onset/ offset of feeder interactions '''
        # responses for each interaction
        all_feeder_responses = np.zeros((n_feeder_int, n_cells, n_timepoints))
        for i, (feeder_on, feeder_off) in enumerate(zip(feeder_int_start, feeder_int_end)):
            if (feeder_on-t_pre < 0) | (feeder_off+t_post > n_frames): # incomplete interaction
                continue
            if feeder_off-feeder_on < t_begin+t_end: # visit too short
                continue
            this_feeder_start = norm_fr[:, feeder_on-t_pre:feeder_on+t_begin]
            this_feeder_end = norm_fr[:, feeder_off-t_end:feeder_off+t_post]
            all_feeder_responses[i] = np.column_stack([this_feeder_start, this_feeder_end])
        rows_to_keep = np.abs(np.sum(all_feeder_responses, axis=(1, 2))).astype(bool)
        all_feeder_responses = all_feeder_responses[rows_to_keep]

        # average response across interactions
        avg_feeder_responses = np.mean(all_feeder_responses, axis=0)
        smooth_feeder_responses = gaussian_filter1d(avg_feeder_responses, 5, axis=1, mode='nearest')  

        ''' Save across sessions '''
        if bird_feeder_responses.shape[0] > 0:
            bird_feeder_responses = np.row_stack((bird_feeder_responses, avg_feeder_responses))
        else:
            bird_feeder_responses = avg_feeder_responses

        n_cells_total += n_cells
        # n_cells_upstart += feeder_coding_dict[bird][session_id]['up start'].shape[0]
        # n_cells_upend += feeder_coding_dict[bird][session_id]['up end'].shape[0]

        ''' Plot the changes in activity across the feeder interaction '''
        # average response binned by arrival/during/departure
        arrive_pad = onset_window[bird]
        arrive_t = t_pre
        depart_pad = offset_window[bird]
        depart_t = t_pre + t_begin + t_end
        arrive_response = np.sum(avg_feeder_responses[:, arrive_t-arrive_pad[0]:arrive_t+arrive_pad[1]], axis=1)/np.sum(arrive_pad)
        during_response = np.sum(avg_feeder_responses[:, arrive_t+arrive_pad[1]:depart_t-depart_pad[0]],  axis=1)/((t_begin+t_end) - (arrive_pad[1]+depart_pad[0]))
        depart_response = np.sum(avg_feeder_responses[:, depart_t-depart_pad[0]:depart_t+depart_pad[1]],  axis=1)/np.sum(depart_pad)

        # fig params
        gs_kw = dict(hspace=0.5)
        f, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True, gridspec_kw=gs_kw)

        ax[0].hist(arrive_response, color='k', bins=30)
        ax[0].set_ylabel('N cells')
        ax[0].set_title('responses to feeder arrival')

        ax[1].hist(during_response, color='k', bins=50)
        ax[1].set_ylabel('N cells')
        ax[1].set_title('responses during feeder visits')

        ax[2].hist(depart_response, color='k', bins=20)
        ax[2].set_ylabel('N cells')
        ax[2].set_xlabel('normalized activity (std)')
        ax[2].set_title('responses to feeder departure')

        f.savefig(f'{save_figs}/{bird}_{session_id}_feeder_responses.png', dpi=400, bbox_inches='tight')
        plt.close()

        ''' Plot average responses for all cells '''
        sort_idx = np.argsort(during_response)  
        assert sort_idx.shape[0] == avg_feeder_responses.shape[0]

        # plot responses relative to onset/offset
        gs_kw = dict(wspace=0.1)
        f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
        im0 = ax[0].imshow(avg_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
                            aspect='auto', cmap='bwr', interpolation='none')
        im1 = ax[1].imshow(avg_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                            aspect='auto', cmap='bwr', interpolation='none')
        ylims = ax[0].get_ylim()

        # plot arrival/departure
        ax[0].vlines(t_pre, 0, n_cells, colors='k', linestyles='dashed', lw=0.5)
        ax[1].vlines(t_end, 0, n_cells, colors='k', linestyles='dashed', lw=0.5)
        ax[0].set_ylim(ylims)
        ax[1].set_ylim(ylims)

        # axis labels
        ax[0].set_xlabel('time from arrival (sec)')
        ax[1].set_xlabel('time from departure (sec)')
        ax[0].set_ylabel('cells sorted by activity during feeder visit')
        ax[1].set_yticks([])
        ax[1].tick_params(labelleft=False)
        f.suptitle(f'feeder-aligned responses for {bird}_{session_id}', y=0.91)

        # ticks
        ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
        ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
        ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
        ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
        ax[0].set_xticklabels(ax0_labels)
        ax[1].set_xticklabels(ax1_labels)

        # add a colorbar
        cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
        cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('activity (z-score)', fontsize=tick_label)
        cbar.set_ticks([])
        cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                        ha='center', va='top', fontsize=tick_label)
        cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                        ha='center', va='bottom', fontsize=tick_label)
        f.savefig(f'{save_figs}/{bird}_{session_id}_feeder_tuning.png', dpi=400, bbox_inches='tight')
        plt.close()

    ''' Plot for bird across all sessions - sorted by activity at start '''
    # get the onset-aligned average activity for a given window
    start_start_fr = onset_window[bird][0]
    start_end_fr = onset_window[bird][1]
    start_start_idx = t_pre-start_start_fr
    start_end_idx = t_pre+start_end_fr
    n_frames_offset = start_end_idx-start_start_idx
    start_response = np.sum(bird_feeder_responses[:, start_start_idx:start_end_idx],  axis=1)/(n_frames_offset)

    sort_idx = np.argsort(start_response)
    # pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    # pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

    # plot responses relative to onset/offset
    gs_kw = dict(wspace=0.1)
    f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
    ax[0].imshow(bird_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
                 aspect='auto', cmap='bwr', interpolation='none')
    im1 = ax[1].imshow(bird_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                        aspect='auto', cmap='bwr', interpolation='none')
    ylims = ax[0].get_ylim()

    # plot arrival/departure
    ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[0].set_ylim(ylims)
    ax[1].set_ylim(ylims)

    # axis labels
    ax[0].set_xlabel('time from arrival (sec)')
    ax[1].set_xlabel('time from departure (sec)')
    ax[0].set_ylabel('cells sorted by activity around arrival')
    ax[1].set_yticks([])
    ax[1].tick_params(labelleft=False)
    # ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    # ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
    ax[0].set_xticklabels(ax0_labels)
    ax[1].set_xticklabels(ax1_labels)

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning_startsort.png', dpi=400, bbox_inches='tight')
    plt.close()

    ''' Plot for bird across all sessions - sorted by activity during '''
    during_response = np.sum(bird_feeder_responses[:, t_pre:t_end+t_begin+t_pre],  axis=1)/(t_begin+t_end)
    sort_idx = np.argsort(during_response)
    # pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    # pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

    # plot responses relative to onset/offset
    gs_kw = dict(wspace=0.1)
    f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
    ax[0].imshow(bird_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
                 aspect='auto', cmap='bwr', interpolation='none')
    im1 = ax[1].imshow(bird_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                        aspect='auto', cmap='bwr', interpolation='none')
    ylims = ax[0].get_ylim()

    # plot arrival/departure
    ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[0].set_ylim(ylims)
    ax[1].set_ylim(ylims)

    # axis labels
    ax[0].set_xlabel('time from arrival (sec)')
    ax[1].set_xlabel('time from departure (sec)')
    ax[0].set_ylabel('cells sorted by activity during feeder visit')
    ax[1].set_yticks([])
    ax[1].tick_params(labelleft=False)
    # ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    # ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
    ax[0].set_xticklabels(ax0_labels)
    ax[1].set_xticklabels(ax1_labels)

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning_dursort.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ''' Plot for bird across all sessions - sorted by activity at end '''
    # get the offset-aligned average activity for a given window
    end_start_fr = offset_window[bird][0]
    end_end_fr = offset_window[bird][1]
    end_start_idx = t_begin+t_pre+t_end-end_start_fr
    end_end_idx = t_begin+t_pre+t_end+end_end_fr
    n_frames_offset = end_end_idx-end_start_idx
    end_response = np.sum(bird_feeder_responses[:, end_start_idx:end_end_idx],  axis=1)/(n_frames_offset)

    sort_idx = np.argsort(end_response)
    # pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    # pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

    # plot responses relative to onset/offset
    gs_kw = dict(wspace=0.1)
    f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
    ax[0].imshow(bird_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
                 aspect='auto', cmap='bwr', interpolation='none')
    im1 = ax[1].imshow(bird_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                        aspect='auto', cmap='bwr', interpolation='none')
    ylims = ax[0].get_ylim()

    # plot arrival/departure
    ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[0].set_ylim(ylims)
    ax[1].set_ylim(ylims)

    # axis labels
    ax[0].set_xlabel('time from arrival (sec)')
    ax[1].set_xlabel('time from departure (sec)')
    ax[0].set_ylabel('cells sorted by activity around departure')
    ax[1].set_yticks([])
    ax[1].tick_params(labelleft=False)
    # ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    # ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
    ax[0].set_xticklabels(ax0_labels)
    ax[1].set_xticklabels(ax1_labels)

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning_endsort.png', dpi=400, bbox_inches='tight')
    plt.close()

    ''' Plot for bird across all sessions - sorted by activity at start / activity at end '''
    sort_idx = np.argsort(start_response / end_response)
    # pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    # pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

    # plot responses relative to onset/offset
    gs_kw = dict(wspace=0.1)
    f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
    ax[0].imshow(bird_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
                 aspect='auto', cmap='bwr', interpolation='none')
    im1 = ax[1].imshow(bird_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                        aspect='auto', cmap='bwr', interpolation='none')
    ylims = ax[0].get_ylim()

    # plot arrival/departure
    ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    ax[0].set_ylim(ylims)
    ax[1].set_ylim(ylims)

    # axis labels
    ax[0].set_xlabel('time from arrival (sec)')
    ax[1].set_xlabel('time from departure (sec)')
    ax[0].set_ylabel('cells sorted by activity around arrivals & departures')
    ax[1].set_yticks([])
    ax[1].tick_params(labelleft=False)
    # ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    # ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
    ax[0].set_xticklabels(ax0_labels)
    ax[1].set_xticklabels(ax1_labels)

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning_ratsort.png', dpi=400, bbox_inches='tight')
    plt.close()

    ''' Save across all birds '''
    if len(all_birds_feeder_responses) == 0:
        all_birds_feeder_responses = bird_feeder_responses
    else:
        all_birds_feeder_responses = np.row_stack([all_birds_feeder_responses, bird_feeder_responses])


''' All cells all birds '''
n_cells_total = all_birds_feeder_responses.shape[0]

''' Plot for bird across all sessions - sorted by activity at start '''
# get the onset-aligned average activity for a given window
start_start_fr = onset_window['all'][0]
start_end_fr = onset_window['all'][1]
start_start_idx = t_pre-start_start_fr
start_end_idx = t_pre+start_end_fr
n_frames_offset = start_end_idx-start_start_idx
start_response = np.sum(all_birds_feeder_responses[:, start_start_idx:start_end_idx],  axis=1)/(n_frames_offset)

sort_idx = np.argsort(start_response)
# pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
# pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

# plot responses relative to onset/offset
gs_kw = dict(wspace=0.05)
f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
ax[0].imshow(all_birds_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
             aspect='auto', cmap='bwr', interpolation='none')
im1 = ax[1].imshow(all_birds_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                    aspect='auto', cmap='bwr', interpolation='none')
ylims = ax[0].get_ylim()

# plot arrival/departure
ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
ax[0].set_ylim(ylims)
ax[1].set_ylim(ylims)

# axis labels
ax[0].set_xlabel('time from arrival (sec)')
ax[1].set_xlabel('time from departure (sec)')
ax[0].set_ylabel('cells sorted by activity around arrival')
ax[1].set_yticks([])
ax[1].tick_params(labelleft=False)
# ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
# ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
f.suptitle(f'all feeder-aligned responses', y=0.95, fontsize=title_size)

# ticks
ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
ax[0].set_xticklabels(ax0_labels)
ax[1].set_xticklabels(ax1_labels)

# add a colorbar
cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
cbar = f.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('activity (z-score)', fontsize=tick_label)
cbar.set_ticks([])
cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                ha='center', va='top', fontsize=tick_label)
cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                ha='center', va='bottom', fontsize=tick_label)

f.savefig(f'{save_figs_dir}/feeder_tuning_startsort.png', dpi=400, bbox_inches='tight')

''' Plot for bird across all sessions - sorted by activity at end '''
# get the offset-aligned average activity for a given window
end_start_fr = offset_window['all'][0]
end_end_fr = offset_window['all'][1]
end_start_idx = t_begin+t_pre+t_end-end_start_fr
end_end_idx = t_begin+t_pre+t_end+end_end_fr
n_frames_offset = end_end_idx-end_start_idx
end_response = np.sum(all_birds_feeder_responses[:, end_start_idx:end_end_idx],  axis=1)/(n_frames_offset)

sort_idx = np.argsort(end_response)
# pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
# pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

# plot responses relative to onset/offset
gs_kw = dict(wspace=0.05)
f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
ax[0].imshow(all_birds_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
             aspect='auto', cmap='bwr', interpolation='none')
im1 = ax[1].imshow(all_birds_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
                    aspect='auto', cmap='bwr', interpolation='none')
ylims = ax[0].get_ylim()

# plot arrival/departure
ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
ax[0].set_ylim(ylims)
ax[1].set_ylim(ylims)

# axis labels
ax[0].set_xlabel('time from arrival (sec)')
ax[1].set_xlabel('time from departure (sec)')
ax[0].set_ylabel('cells sorted by activity around departure')
ax[1].set_yticks([])
ax[1].tick_params(labelleft=False)
# ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
# ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
f.suptitle(f'all feeder-aligned responses', y=0.95, fontsize=title_size)

# ticks
ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
ax[0].set_xticklabels(ax0_labels)
ax[1].set_xticklabels(ax1_labels)

# add a colorbar
cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
cbar = f.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('activity (z-score)', fontsize=tick_label)
cbar.set_ticks([])
cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                ha='center', va='top', fontsize=tick_label)
cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                ha='center', va='bottom', fontsize=tick_label)

f.savefig(f'{save_figs_dir}/feeder_tuning_endsort.png', dpi=400, bbox_inches='tight')