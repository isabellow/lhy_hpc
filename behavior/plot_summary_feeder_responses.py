import numpy as np
import pandas as pd
from scipy import stats

import os 
import sys
sys.path.append("..//utils/")
import color_utils, helpers
from load_matlab_data import loadmat_sbx
sys.path.append("..//neural/")
from format_waveform_data import get_spike_times, load_wf_data, sort_wf_by_channel, pop_normalize
from format_behavior_data import load_behavior_data, get_feeder_ints, get_feeder_periods, classify_feeder_ints, get_feeder_departure_bounds, get_foot_angle
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

'''
TODO:
- try different time windows for offset avg for each bird
--> could use vertical lines on the plot to show this window
- try ratio of this avg to during (non-overlapping window)
- montage of feeder departures per bird
- aligning to ankles bending before/at turn away from feeder
--> capture departure prep for both turn-first and hop-away birds

- exclude low firing cells
- exclude cells not on stim channels
'''

''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"
posture_file = 'posture_pos_smooth.npy'

''' Load the data dictionary for all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

''' Data params '''
fps = 50 # Hz
dt = 1/fps

''' Plotting params '''
# for filtering out cells
fr_thresh = 0.05 # Hz, threshold for excluding low firing cells

# angle between foot vectors for leaving the feeder
align_to_feet = False
angle_thresh = 20 # degrees

# feeder visit time windows (seconds)
t_pre_sec = 1
t_begin_sec = 0.5
t_end_sec = 0.5
t_post_sec = 1

# convert feeder time windows to frames
t_pre = int(t_pre_sec/dt)
t_begin = int(t_begin_sec/dt)
t_end = int(t_end_sec/dt)
t_post = int(t_post_sec/dt)
n_timepoints = t_pre + t_begin + t_end + t_post

# frame windows for average offset activity (start, end)
offset_window = {}
offset_window['RBY94'] = np.asarray([14, 15])
offset_window['AMB154'] = np.asarray([5, 15])
offset_window['SLV132'] = np.asarray([5, 15])
offset_window['LMN146'] = np.asarray([12, 15])
offset_window['IND67'] = np.asarray([10, 15])
offset_window['all'] = np.asarray([5, 10])

# frame windows for average onset activity (start, end)
onset_window = {}
onset_window['RBY94'] = np.asarray([14, 15])
onset_window['AMB154'] = np.asarray([5, 15])
onset_window['SLV132'] = np.asarray([5, 15])
onset_window['LMN146'] = np.asarray([12, 15])
onset_window['IND67'] = np.asarray([10, 15])
onset_window['all'] = np.asarray([5, 10])

# font sizes
title_size = 14
axis_label = 12
tick_label = 9

''' Get the hand-annotated feeder response info '''
# feeder_info = pd.read_excel(session_info_file, sheet_name='Feeder Coding', header=0)
# feeder_coding_dict = {}
# for i, row in feeder_info.iterrows():
#     # get the bird and session IDs
#     bird = row['bird']
#     session_id = str(row['session'])
#     if bird not in feeder_coding_dict.keys():
#         feeder_coding_dict[bird] = {}
#     feeder_coding_dict[bird][session_id] = {}
    
#     # get the cell IDs associated with each feeder response
#     for col in feeder_info.columns:
#         if col in ['bird', 'session']:
#             continue
#         feeder_cell_string = row[col]

#         if pd.isna(feeder_cell_string): # handle empty cells
#             feeder_coding_dict[bird][session_id][col] = np.array([])
#         else:
#             feeder_cells = np.array([int(x.strip()) for x in str(feeder_cell_string).split(',')])
#             feeder_coding_dict[bird][session_id][col] = feeder_cells

all_birds_feeder_responses = []
for bird in bird_ids:
    if bird == 'LIM63':
        continue
    print(f'\nplotting feeder responses for {bird}')
    
    # to save figures
    save_figs = f"{save_figs_dir}{bird}/feeder_responses/"
    if os.path.isdir(save_figs):
        print('save directory exists')
    else:
        os.mkdir(save_figs)

    # to collect data across sessions
    n_cells_total = 0
    n_cells_upstart = 0
    n_cells_upend = 0
    bird_feeder_responses = np.asarray([])
    
    # collect sessions with pose tracking & ephys
    session_list = data_dict[bird]['all_sessions']
    behavior_sessions = []
    for session_id in session_list:
        preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
        if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
            behavior_sessions.append(session_id)

    for session_id in behavior_sessions:
        # skip sessions with too few feeder visits
        if (bird == 'AMB154') & (session_id == '241114'):
            continue
        
        ''' Get the file params '''
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"

        ''' Load and format the neural data '''
        # spikes per video frame
        spike_fr = np.load(f"{data_dir}aligned_spikes.npy") # cells x video frames

        # session average firing rate
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]
        avg_firing_rate = 10**log_fr

        # filter out low-firing cells and cells not in the nucleus
        high_fr = avg_firing_rate > fr_thresh
        if 'stim_resp_idx_ch' in data_dict[bird][session_id].keys():
            stim_idx_cell = idx_cells_by_stim(data_dict, bird, session_id)
        else:
            print(f'warning! no stim data for {bird}_{session}')
            stim_idx_cell = np.ones(n_cells).astype(bool)
        excitatory_idx = data_dict[bird][session_id]['excitatory_idx']
        cell_filt_idx = high_fr & stim_idx_cell & excitatory_idx
        spike_fr = spike_fr[cell_filt_idx]
        n_cells, n_frames = spike_fr.shape

        # normalize activity for population analysis 
        norm_fr = pop_normalize(spike_fr, dt=dt)

        ''' Load and format the behavior data '''
        seed_struct, count_data = load_behavior_data(data_dir)

        # get the feeder interactions + classify as open/closed
        feeder_int_start, feeder_int_end, feeder_idx = get_feeder_ints(count_data, use_beak=False)
        if align_to_feet:
            feet_angle = get_foot_angle(data_dir, posture_file)
            feeder_depart_start, feeder_depart_end, feeder_idx = get_feeder_departure_bounds(count_data)
            assert feeder_int_end.shape[0] == feeder_depart_end.shape[0]
            for f_idx, (start_t, end_t) in enumerate(zip(feeder_depart_start, feeder_depart_end)):
                these_angles = np.degrees(feet_angle[start_t:end_t])
                leave_idx = np.argmax(these_angles >= angle_thresh)
                if np.sum(these_angles >= angle_thresh):
                    feeder_int_end[f_idx] = np.min([start_t+leave_idx, end_t])
        feeder_open_times, feeder_close_times = get_feeder_periods(session_info_file, bird, session_id)
        feeder_status = classify_feeder_ints(feeder_int_start, feeder_int_end, feeder_open_times, feeder_close_times)
        n_feeder_int = feeder_int_start.shape[0]

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

        # # report fractions of responsive cells
        # if session_id in feeder_coding_dict[bird].keys():
        #     these_feeder_cells = feeder_coding_dict[bird][session_id]
        # else:
        #     continue
        # print('hand annotated feeder response proportions:')
        # for key in these_feeder_cells.keys():
        #     n_feeder_cells = these_feeder_cells[key].shape[0]
        #     frac_feeder_cells = np.round(n_feeder_cells/n_cells*100, 2)
        #     print(f"{frac_feeder_cells}% had {key} feeder responses ({n_feeder_cells}/{n_cells})")

        ''' Save across sessions '''
        if bird_feeder_responses.shape[0] > 0:
            bird_feeder_responses = np.row_stack((bird_feeder_responses, avg_feeder_responses))
        else:
            bird_feeder_responses = avg_feeder_responses

        n_cells_total += n_cells
        # n_cells_upstart += feeder_coding_dict[bird][session_id]['up start'].shape[0]
        # n_cells_upend += feeder_coding_dict[bird][session_id]['up end'].shape[0]

        ''' Plot the changes in activity across the feeder interaction '''
        # # average response binned by arrival/during/departure
        # arrive_pad = onset_window[bird]
        # arrive_t = t_pre
        # depart_pad = offset_window[bird]
        # depart_t = t_pre + t_begin + t_end
        # arrive_response = np.sum(avg_feeder_responses[:, arrive_t-arrive_pad[0]:arrive_t+arrive_pad[1]], axis=1)/np.sum(arrive_pad)
        # during_response = np.sum(avg_feeder_responses[:, arrive_t+arrive_pad[1]:depart_t-depart_pad[0]],  axis=1)/((t_begin+t_end) - (arrive_pad[1]+depart_pad[0]))
        # depart_response = np.sum(avg_feeder_responses[:, depart_t-depart_pad[0]:depart_t+depart_pad[1]],  axis=1)/np.sum(depart_pad)

        # # fig params
        # gs_kw = dict(hspace=0.5)
        # f, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True, gridspec_kw=gs_kw)

        # ax[0].hist(arrive_response, color='k', bins=30)
        # ax[0].set_ylabel('N cells')
        # ax[0].set_title('responses to feeder approach')

        # ax[1].hist(during_response, color='k', bins=50)
        # ax[1].set_ylabel('N cells')
        # ax[1].set_title('responses during feeder visits')

        # ax[2].hist(depart_response, color='k', bins=20)
        # ax[2].set_ylabel('N cells')
        # ax[2].set_xlabel('normalized activity (std)')
        # ax[2].set_title('responses to feeder departure')

        # f.savefig(f'{save_figs}/{bird}_{session_id}_feeder_responses.png', dpi=400, bbox_inches='tight')

        # ''' Plot average responses for all cells '''
        # sort_idx = np.argsort(during_response)

        # # plot responses relative to onset/offset
        # gs_kw = dict(wspace=0.05)
        # f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
        # im0 = ax[0].imshow(avg_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
        #                     aspect='auto', cmap='bwr', interpolation='none')
        # im1 = ax[1].imshow(avg_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
        #                     aspect='auto', cmap='bwr', interpolation='none')
        # ylims = ax[0].get_ylim()

        # # plot arrival/departure
        # ax[0].vlines(t_pre, 0, n_cells, colors='k', linestyles='dashed', lw=0.5)
        # ax[1].vlines(t_end, 0, n_cells, colors='k', linestyles='dashed', lw=0.5)
        # ax[0].set_ylim(ylims)
        # ax[1].set_ylim(ylims)

        # # axis labels
        # ax[0].set_xlabel('time from arrival (sec)')
        # ax[1].set_xlabel('time from departure (sec)')
        # ax[0].set_ylabel('cells sorted by activity during feeder visit')
        # ax[1].set_yticks([])
        # ax[1].tick_params(labelleft=False)
        # f.suptitle(f'feeder-aligned responses for {bird}_{session_id}', y=0.91)

        # # ticks
        # ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
        # ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
        # ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
        # ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
        # ax[0].set_xticklabels(ax0_labels)
        # ax[1].set_xticklabels(ax1_labels)

        # # add a colorbar
        # cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
        # cbar = f.colorbar(im1, cax=cax, orientation='vertical')
        # cbar.set_label('activity (z-score)', fontsize=tick_label)
        # cbar.set_ticks([])
        # cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
        #                 ha='center', va='top', fontsize=tick_label)
        # cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
        #                 ha='center', va='bottom', fontsize=tick_label)
        # f.savefig(f'{save_figs}/{bird}_{session_id}_feeder_tuning.png', dpi=400, bbox_inches='tight')

    ''' Save across all birds '''
    if len(all_birds_feeder_responses) == 0:
        all_birds_feeder_responses = bird_feeder_responses
    else:
        all_birds_feeder_responses = np.row_stack([all_birds_feeder_responses, bird_feeder_responses])
    

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
    gs_kw = dict(wspace=0.05)
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

    ''' Plot for bird across all sessions - sorted by activity during '''
    # during_response = np.sum(bird_feeder_responses[:, t_pre:t_end+t_begin+t_pre],  axis=1)/(t_begin+t_end)
    # sort_idx = np.argsort(during_response)
    # # pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    # # pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

    # # plot responses relative to onset/offset
    # gs_kw = dict(wspace=0.05)
    # f, ax = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw=gs_kw)
    # ax[0].imshow(bird_feeder_responses[sort_idx, :n_timepoints//2], clim=[-1, 1], 
    #              aspect='auto', cmap='bwr', interpolation='none')
    # im1 = ax[1].imshow(bird_feeder_responses[sort_idx, n_timepoints//2:], clim=[-1, 1], 
    #                     aspect='auto', cmap='bwr', interpolation='none')
    # ylims = ax[0].get_ylim()

    # # plot arrival/departure
    # ax[0].vlines(t_pre, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    # ax[1].vlines(t_end, 0, n_cells_total, colors='k', linestyles='dashed', lw=0.5)
    # ax[0].set_ylim(ylims)
    # ax[1].set_ylim(ylims)

    # # axis labels
    # ax[0].set_xlabel('time from arrival (sec)')
    # ax[1].set_xlabel('time from departure (sec)')
    # ax[0].set_ylabel('cells sorted by activity during feeder visit')
    # ax[1].set_yticks([])
    # ax[1].tick_params(labelleft=False)
    # # ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    # # ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    # f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # # ticks
    # ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt/2, 1/dt/2))
    # ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt/2, 1/dt/2))
    # ax0_labels = (np.arange(-t_pre, t_begin+1/dt/2, 1/dt/2))*dt
    # ax1_labels = (np.arange(-t_end, t_post+1/dt/2, 1/dt/2))*dt
    # ax[0].set_xticklabels(ax0_labels)
    # ax[1].set_xticklabels(ax1_labels)

    # # add a colorbar
    # cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    # cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    # cbar.set_label('activity (z-score)', fontsize=tick_label)
    # cbar.set_ticks([])
    # cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
    #                 ha='center', va='top', fontsize=tick_label)
    # cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
    #                 ha='center', va='bottom', fontsize=tick_label)

    # f.savefig(f'{save_figs}/{bird}_feeder_tuning.png', dpi=400, bbox_inches='tight')


    ''' Plot for bird across all sessions - sorted by activity at end '''
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
    gs_kw = dict(wspace=0.05)
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

    ''' Plot for bird across all sessions - sorted by activity at start / activity at end '''
    sort_idx = np.argsort(np.abs(start_response / end_response))
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