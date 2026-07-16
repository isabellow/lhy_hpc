import numpy as np
import pandas as pd

import os 
import sys
sys.path.append("..//utils/")
import color_utils
from load_matlab_data import loadmat_sbx
sys.path.append("..//neural/")
from format_waveform_data import get_spike_times
from format_behavior_data import load_behavior_data, get_feeder_ints, get_feeder_periods, classify_feeder_ints, get_feeder_departure_bounds, get_foot_angle
sys.path.append("..//stim/")
from format_chronic_stim import idx_cells_by_stim

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

'''
For all sessions for a given bird:
- get all feeder interactions
- label whether the feeder was open or closed for each interaction
- identify cells with altered firing rates associated with each interaction
--> count and report % feeder-modulated, N up, N down, N other
- plot the feeder-aligned responses for those cells 

Plot:
Raster aligned to feeder interactions sorted by:
- open vs. closed
- feeder location
- interaction duration
Tuning cruves aligned to interaction onset/offset, split by feeder

TODO:
Add summary plot of feeder activation/suppression magnitude
Print stats for N feeder responsive cells
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"
posture_file = 'posture_pos_smooth.npy'

''' Thresholding params '''
# multiplicative/divisive factors
firing_up = 2 # >= considered elevated firing rate
firing_down = 2 # <= considered reduced firing rate

# for filtering out cells
fr_thresh = 0.05 # Hz, threshold for excluding low firing cells

# angle between foot vectors for leaving the feeder
align_to_feet = False
angle_thresh = 20 # degrees

''' Data params '''
bird = 'RBY94' # update as needed
data_dict = np.load(data_file, allow_pickle=True).item()
session_list = data_dict[bird]['all_sessions']
fps = 50 # Hz
dt = 1/fps

# collect sessions with pose tracking & ephys
behavior_sessions = []
for session_id in session_list:
    preprocessed_data = data_dict[bird][session_id]['preprocessed_data']
    if ('behavior' in preprocessed_data) & ('ephys' in preprocessed_data):
        behavior_sessions.append(session_id)

''' Plotting params '''
# feeder visit time windows (seconds)
t_on_start = -1
t_on_end = 0.5
timepoints_on = np.arange(t_on_start, t_on_end, dt)
n_t_on = timepoints_on.shape[0]

t_off_start = -0.5
t_off_end = 1
timepoints_off = np.arange(t_off_start, t_off_end, dt)
n_t_off = timepoints_off.shape[0]

# convert feeder time windows to frames
fr_on_start = int(t_on_start/dt)
fr_on_end = int(t_on_end/dt)
fr_off_start = int(t_off_start/dt)
fr_off_end = int(t_off_end/dt)

# fig params
feeder_colors = ['xkcd:saffron', 'green', 'xkcd:scarlet', 'blue']
title_size = 14
axis_label = 12
tick_label = 10

''' Define/create the save folder'''
save_dir = f"{save_figs_dir}/{bird}/"
if os.path.isdir(save_dir):
    print('save directory exists')
else:
    os.mkdir(save_dir)
save_folder = f"{save_dir}/feeder_responses_test/"
if os.path.isdir(save_folder):
    print('save folder exists')
else:
    os.mkdir(save_folder)

''' Plot feeder responses for each session '''
for session_id in behavior_sessions:
    print(f'plotting feeder responsive cells for {bird}_{session_id}')
    feeder_exclude = 0

    ''' Get the file params '''
    session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir = f"{session_dir}/behavior_data/"

    ''' Load and format the neural data '''
    # spikes per video frame
    spike_fr = np.load(f"{data_dir}aligned_spikes.npy") # cells x video frames
    spike_bool = spike_fr.astype(bool)

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
    spike_bool = spike_bool[cell_filt_idx]
    n_cells, n_frames = spike_fr.shape

    ''' Load and format behavior data '''
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

    ''' Raster aligned to feeder interaction offset '''
    # time window
    feeder_t_window = 20 # seconds
    feeder_t_points = np.arange(-feeder_t_window//2, feeder_t_window//2 + dt, dt)
    n_t_pts = feeder_t_points.shape[0]
    fr_halfwidth = n_t_pts//2

    # divide by open vs closed (currently excluding partial feeder open interactions)
    open_feeder_onsets = feeder_int_start[feeder_status==1]
    open_feeder_offsets = feeder_int_end[feeder_status==1]
    open_feeder_idx = feeder_idx[feeder_status==1]
    n_open_feeder = open_feeder_onsets.shape[0]

    closed_feeder_onsets = feeder_int_start[feeder_status==0]
    closed_feeder_offsets = feeder_int_end[feeder_status==0]
    closed_feeder_idx = feeder_idx[feeder_status==0]
    n_closed_feeder = closed_feeder_onsets.shape[0]

    # matrix of spike times by feeder interactions
    feeder_off_mat = np.zeros((n_cells, n_feeder_int, n_t_pts), dtype=bool)
    feeder_ids = np.unique(feeder_idx)
    sorted_feeder_ons = np.zeros(n_feeder_int) # for plotting feeder onset points
    feeder_switch_pts = np.asarray([]) # for indexing by feeder ID
    for c_idx in range(n_cells):
        spk_times = spike_bool[c_idx]
        
        # open feeders sorted by feeder location + duration
        f_idx = 0
        for f in feeder_ids:
            if any(open_feeder_idx==f):
                f_on = open_feeder_onsets[open_feeder_idx==f]
                f_off = open_feeder_offsets[open_feeder_idx==f]
                f_duration = f_off - f_on
                duration_idx = np.argsort(f_duration)
                for i, dur_idx in enumerate(duration_idx):
                    sort_idx = i + f_idx
                    start_idx = (f_off[dur_idx] - fr_halfwidth).astype(int)
                    end_idx = (f_off[dur_idx] + fr_halfwidth + 1).astype(int) 
                    if f_off[dur_idx] - fr_halfwidth < 0:
                        n_t = spk_times[:end_idx].shape[0]
                        feeder_off_mat[c_idx, sort_idx, -n_t:] = spk_times[:end_idx]
                    elif f_off[dur_idx] + fr_halfwidth >= n_frames:
                        n_t = spk_times[start_idx:].shape[0]
                        feeder_off_mat[c_idx, sort_idx, :n_t] = spk_times[start_idx:]
                    else:
                        feeder_off_mat[c_idx, sort_idx] = spk_times[start_idx:end_idx]
                    if c_idx == 0:
                        sorted_feeder_ons[sort_idx] = np.max((-f_duration[dur_idx], -fr_halfwidth))
                f_idx += f_on.shape[0]
                if c_idx == 0:
                    feeder_switch_pts = np.append(feeder_switch_pts, f_idx)
        
        # closed feeders sorted by feeder location + duration
        f_idx = n_open_feeder
        for f in feeder_ids:
            if any(closed_feeder_idx==f):
                f_on = closed_feeder_onsets[closed_feeder_idx==f]
                f_off = closed_feeder_offsets[closed_feeder_idx==f]
                f_duration = f_off - f_on
                duration_idx = np.argsort(f_duration)
                for i, dur_idx in enumerate(duration_idx):
                    sort_idx = i + f_idx
                    start_idx = (f_off[dur_idx] - fr_halfwidth).astype(int)
                    end_idx = (f_off[dur_idx] + fr_halfwidth + 1).astype(int) 
                    if f_off[dur_idx] - fr_halfwidth < 0:
                        n_t = spk_times[:end_idx].shape[0]
                        feeder_off_mat[c_idx, sort_idx, -n_t:] = spk_times[:end_idx]
                    elif f_off[dur_idx] + fr_halfwidth >= n_frames:
                        n_t = spk_times[start_idx:].shape[0]
                        feeder_off_mat[c_idx, sort_idx, :n_t] = spk_times[start_idx:]
                    else:
                        feeder_off_mat[c_idx, sort_idx] = spk_times[start_idx:end_idx]
                    
                    # save onset times
                    if c_idx == 0:
                        sorted_feeder_ons[sort_idx] = np.max((-f_duration[dur_idx], -fr_halfwidth))
                
                # save switch points
                f_idx += f_on.shape[0]
                if c_idx == 0:
                    feeder_switch_pts = np.append(feeder_switch_pts, f_idx)                    
    feeder_switch_pts = feeder_switch_pts[:-1].astype(int)

    ''' Compute the psth aligned to onset and offset '''
    # responses and states for each interaction
    feeder_onset_spikes = np.zeros((n_open_feeder, n_cells, n_t_on))
    feeder_offset_spikes = np.zeros((n_open_feeder, n_cells, n_t_off))
    for i, (feeder_on, feeder_off) in enumerate(zip(open_feeder_onsets, open_feeder_offsets)):
        if (feeder_on+fr_on_start < 0) | (feeder_off+fr_off_end > n_frames): # incomplete interaction
            continue
        if feeder_off-feeder_on < fr_on_end-fr_off_start: # visit too short
            continue
        feeder_onset_spikes[i] = spike_fr[:, feeder_on+fr_on_start:feeder_on+fr_on_end]
        feeder_offset_spikes[i] = spike_fr[:, feeder_off+fr_off_start:feeder_off+fr_off_end]
    
    # filter out short/incomplete interactions
    rows_to_keep = np.abs(np.sum(feeder_onset_spikes, axis=(1, 2))).astype(bool)
    feeder_onset_spikes = feeder_onset_spikes[rows_to_keep]
    feeder_offset_spikes = feeder_offset_spikes[rows_to_keep]
    feeder_idx_filt = open_feeder_idx[rows_to_keep]

    # average response across open feeder interactions, chunked by feeder ID and state
    open_feeder_ids, n_visits = np.unique(feeder_idx_filt, return_counts=True)
    n_open_ids = open_feeder_ids.shape[0]
    feeder_onset_psth = np.zeros((n_cells, n_open_ids, n_t_on))
    feeder_offset_psth = np.zeros((n_cells, n_open_ids, n_t_off))
    for f_idx, f_id in enumerate(open_feeder_ids):
        feeder_onset_psth[:, f_idx] = np.mean(feeder_onset_spikes[feeder_idx_filt == f_id]/dt, axis=0)
        feeder_offset_psth[:, f_idx] = np.mean(feeder_offset_spikes[feeder_idx_filt == f_id]/dt, axis=0)

    # smooth the firing rates
    feeder_onset_psth_smooth = gaussian_filter1d(feeder_onset_psth, fps//10, axis=2, mode='nearest')
    feeder_offset_psth_smooth = gaussian_filter1d(feeder_offset_psth, fps//10, axis=2, mode='nearest')

    ''' Check for cells with firing modulations near feeder interactions '''
    n_open_int = np.sum(feeder_status==1)
    feeder_tuned_idx = np.asarray([])
    for c_idx in range(n_cells):
        baseline = avg_firing_rate[c_idx]
        feeder_psth = np.column_stack([feeder_onset_psth_smooth[c_idx], feeder_offset_psth_smooth[c_idx]])
        up_thresh = baseline*firing_up
        down_thresh = baseline/firing_down
        if (feeder_psth >= up_thresh).any() | (feeder_psth <= down_thresh).any():
            if np.sum(feeder_off_mat[c_idx, :n_open_int]) > n_open_int:
                feeder_tuned_idx = np.append(feeder_tuned_idx, c_idx)
    feeder_tuned_idx = feeder_tuned_idx.astype(int)


    ''' Plot feeder tuning for cells with firing rate modulations '''
    # fig params
    psth_lw = 3
    on_lw = 1
    off_lw = 1
    time_int = 5

    # data params
    avg_fr_session = np.round(avg_firing_rate, 2)

    # determine spike size from N events
    spk_s = 18510/(n_open_int**2)
    on_s = spk_s/2

    # plt.ion()
    f, ax = plt.subplots(1, 4, figsize=(10, 2.5),
                             gridspec_kw=dict(width_ratios=[1, 0.2, 1, 1], wspace=0.1))
    for c_idx in feeder_tuned_idx:
        cell_id = c_idx # good_clusters[c_idx]

        # clear last plot
        for j in range(4):
            ax[j].clear()
        
        # remove extraneous tick labels and axes
        for j in range(2):
            ax[j+2].spines['top'].set_visible(False)
            ax[j+2].spines['right'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[3].tick_params(labelleft=False)
        
        # hide the spacer subplot
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_facecolor('None')
        
        # rasters aligned to feeder offsets
        color_idx = 0
        for feeder_idx in range(n_open_int):
            if feeder_idx in feeder_switch_pts:
                color_idx += 1
            spk_idx_on = feeder_off_mat[c_idx, feeder_idx]
            spk_t_on = feeder_t_points[spk_idx_on]
            ax[0].scatter(spk_t_on, np.full(spk_t_on.shape[0], feeder_idx),
                             color=feeder_colors[color_idx], marker='|',
                             lw=0.6, s=spk_s, alpha=1)
        
        # psth for each feeder with sufficient open visits
        for f_idx, f_id in enumerate(open_feeder_ids-1):
            if (n_visits[f_idx] < 5):
                if (feeder_exclude==0):
                    print(f'excluding feeder {f_id+1} from PSTH (too few trials)')
            else:
                ax[2].plot(timepoints_on, feeder_onset_psth_smooth[c_idx, f_idx], 
                           lw=psth_lw, color=feeder_colors[f_id])
                ax[3].plot(timepoints_off, feeder_offset_psth_smooth[c_idx, f_idx],
                           lw=psth_lw, color=feeder_colors[f_id])
            if f_id+1 == n_open_ids:
                feeder_exclude = 1
        
            # label onsets and offsets for the rasters        
            ax[0].vlines(0, 0, n_open_int, colors='k', linestyles='dashed', lw=off_lw)
            sub_ons = sorted_feeder_ons[:n_open_int]
            ax[0].scatter(sub_ons/50, np.arange(n_open_int), color='k', marker='|',
                          lw=on_lw, s=on_s, zorder=2)
                       
        # label onset and offset, baseline FR for the psth
        max_on = np.ceil(np.max(feeder_onset_psth_smooth[c_idx, n_visits>=5]))
        max_off = np.ceil(np.max(feeder_offset_psth_smooth[c_idx, n_visits>=5]))
        max_all = np.max([max_on, max_off])
        ax[2].vlines(0, 0, max_all, colors='k', linestyles='dashed', lw=off_lw)
        ax[3].vlines(0, 0, max_all, colors='k', linestyles='dashed', lw=off_lw)
        ax[2].hlines(avg_fr_session[c_idx], t_on_start, t_on_end, 
                     colors='xkcd:gray', linestyles='dashed', lw=off_lw)
        ax[3].hlines(avg_fr_session[c_idx], t_off_start, t_off_end, 
                     colors='xkcd:gray', linestyles='dashed', lw=off_lw)

        # limits and ticks
        ax[0].set_xlim(feeder_t_points[0], feeder_t_points[-1])
        ax[0].set_ylim(-0.5, n_open_int-0.5)
        ax[0].set_xticks(np.arange(-feeder_t_window//2, feeder_t_window//2 + 0.5, time_int))
    #     ax[0].set_yticks(np.arange(0, n_feeder_int, 5))

        ax[2].set_xlim([t_on_start, t_on_end])
        ax[3].set_xlim([t_off_start, t_off_end])
        ax[2].set_ylim([0, max_all])
        ax[2].set_yticks([0, max_all])
        ax[3].set_ylim([0, max_all])
        ax[3].set_yticks([])

        # axis labels
        ax[2].set_title(f'{bird} {session_id}\ncell {cell_id} (baseline {avg_fr_session[c_idx]} Hz)',
                        fontsize=title_size, pad=10)
        ax[0].set_xlabel('time from departure (s)', fontsize=axis_label)
        ax[2].set_xlabel('time from arrival (s)', fontsize=axis_label)
        ax[3].set_xlabel('time from departure (s)', fontsize=axis_label)
        ax[2].set_ylabel('average firing rate (Hz)', fontsize=axis_label)
        ax[0].set_ylabel('open feeder visits', fontsize=axis_label)
                         
        f.savefig(f'{save_folder}/{session_id}_feeder_tuning_cell{cell_id}.png', dpi=400, bbox_inches='tight')

        # show the plot until the viewer chooses to advance
        # f.canvas.draw_idle()
        # plt.show()
        # input('press enter for next plot')