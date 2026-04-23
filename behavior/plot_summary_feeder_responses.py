import numpy as np
import pandas as pd
from scipy import stats

import os 
import sys
sys.path.append("..//utils/")
import color_utils, helpers
from load_matlab_data import loadmat_sbx
sys.path.append("..//neural/")
from format_waveform_data import get_spike_times
from format_behavior_data import load_behavior_data, get_feeder_ints, get_feeder_periods, classify_feeder_ints

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


''' File Paths '''
root_dir = "Z:/Isabel/data/hpc_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}stim_session_data.npy"
session_info_file = f"{root_dir}good_sessions.xlsx"

''' Load the data dictionary for all good stim sessions '''
bird_ids = []
data_dict = np.load(data_file, allow_pickle=True).item()
for bird in data_dict.keys():
    bird_ids.append(bird)

''' Plotting params '''
# feeder time windows
t_pre_sec = 2
t_begin_sec = 1
t_end_sec = 1
t_post_sec = 2

# font sizes
title_size = 14
axis_label = 12
tick_label = 9

''' Get the hand-annotated feeder response info '''
feeder_info = pd.read_excel(session_info_file, sheet_name='Feeder Coding', header=0)
feeder_coding_dict = {}
for i, row in feeder_info.iterrows():
    # get the bird and session IDs
    bird = row['bird']
    session_id = str(row['session'])
    if bird not in feeder_coding_dict.keys():
        feeder_coding_dict[bird] = {}
    feeder_coding_dict[bird][session_id] = {}
    
    # get the cell IDs associated with each feeder response
    for col in feeder_info.columns:
        if col in ['bird', 'session']:
            continue
        feeder_cell_string = row[col]

        if pd.isna(feeder_cell_string): # handle empty cells
            feeder_coding_dict[bird][session_id][col] = np.array([])
        else:
            feeder_cells = np.array([int(x.strip()) for x in str(feeder_cell_string).split(',')])
            feeder_coding_dict[bird][session_id][col] = feeder_cells


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
        elif (bird == 'IND67') & (session_id == '251014'):
            continue
        
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
        spike_fr = np.zeros((n_cells, n_frames))
        i = -1
        for c_idx, cell in enumerate(good_clusters):
            i += 1
            spk_times = spike_t[spike_id==cell]       
            spike_fr[i], _ = np.histogram(spk_times, frame_samples)
        spike_bool = spike_fr.astype(bool)

        # session average firing rate
        waveform_props = data_dict[bird][session_id]['waveform_props']
        log_fr = waveform_props[2]
        avg_firing_rate = 10**log_fr

        ''' Normalize activity for population analysis '''
        # instantaneous firing rate
        inst_firing_rate = spike_fr/dt

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

        ''' Load and format behavior data '''
        data_dir = f"{root_dir}{bird}/{bird}_{session_id}/behavior_data/"
        seed_struct, count_data = load_behavior_data(data_dir)

        # get the feeder interactions + classify as open/closed
        feeder_int_start, feeder_int_end, feeder_idx = get_feeder_ints(count_data, use_beak=False)
        feeder_open_times, feeder_close_times = get_feeder_periods(session_info_file, bird, session_id)
        feeder_status = classify_feeder_ints(feeder_int_start, feeder_int_end, feeder_open_times, feeder_close_times)
        n_feeder_int = feeder_int_start.shape[0]

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
            if (feeder_on-t_pre < 0) | (feeder_off+t_post > n_frames):
                continue
            this_feeder_start = norm_fr[:, feeder_on-t_pre:feeder_on+t_begin]
            this_feeder_end = norm_fr[:, feeder_off-t_end:feeder_off+t_post]
            all_feeder_responses[i] = np.column_stack([this_feeder_start, this_feeder_end])
        rows_to_keep = np.abs(np.sum(all_feeder_responses, axis=(1, 2))).astype(bool)
        all_feeder_responses = all_feeder_responses[rows_to_keep]

        # average response across interactions
        avg_feeder_responses = np.mean(all_feeder_responses, axis=0)
        smooth_feeder_responses = gaussian_filter1d(avg_feeder_responses, 5, axis=1, mode='nearest')  

        # average response binned by pre/during/post interaction
        pre_response = np.sum(avg_feeder_responses[:, :t_pre], axis=1)/t_pre
        during_response = np.sum(avg_feeder_responses[:, t_pre:t_end+t_begin+t_pre],  axis=1)/(t_begin+t_end)
        post_response = np.sum(avg_feeder_responses[:, -t_post:],  axis=1)/(t_post)
        sort_idx = np.argsort(during_response)  
        assert sort_idx.shape[0] == avg_feeder_responses.shape[0]

        # report fractions of responsive cells
        if session_id in feeder_coding_dict[bird].keys():
            these_feeder_cells = feeder_coding_dict[bird][session_id]
        else:
            continue
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
        n_cells_upstart += feeder_coding_dict[bird][session_id]['up start'].shape[0]
        n_cells_upend += feeder_coding_dict[bird][session_id]['up end'].shape[0]

        ''' Plot the changes in activity across the feeder interaction '''
        gs_kw = dict(hspace=0.5)
        f, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True, gridspec_kw=gs_kw)

        ax[0].hist(pre_response, color='k', bins=30)
        ax[0].set_ylabel('N cells')
        ax[0].set_title('responses to feeder approach')

        ax[1].hist(during_response, color='k', bins=50)
        ax[1].set_ylabel('N cells')
        ax[1].set_title('responses during feeder visits')

        ax[2].hist(post_response, color='k', bins=20)
        ax[2].set_ylabel('N cells')
        ax[2].set_xlabel('normalized activity (std)')
        ax[2].set_title('responses to feeder departure')

        f.savefig(f'{save_figs}/{bird}_{session_id}_feeder_responses.png', dpi=400, bbox_inches='tight')

        ''' Plot average responses for all cells '''
        # plot responses relative to onset/offset
        gs_kw = dict(wspace=0.05)
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
        ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt, 1/dt))
        ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt, 1/dt))
        ax0_labels = (np.arange(-t_pre, t_begin+1/dt, 1/dt))*dt
        ax1_labels = (np.arange(-t_end, t_post+1/dt, 1/dt))*dt
        ax[0].set_xticklabels(ax0_labels.astype(int))
        ax[1].set_xticklabels(ax1_labels.astype(int))

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


    ''' Plot for bird across all sessions - sorted by activity during '''
    during_response = np.sum(bird_feeder_responses[:, t_pre:t_end+t_begin+t_pre],  axis=1)/(t_begin+t_end)
    sort_idx = np.argsort(during_response)
    pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

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
    ax[0].set_ylabel('cells sorted by activity during feeder visit')
    ax[1].set_yticks([])
    ax[1].tick_params(labelleft=False)
    ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt, 1/dt))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt, 1/dt))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt, 1/dt))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt, 1/dt))*dt
    ax[0].set_xticklabels(ax0_labels.astype(int))
    ax[1].set_xticklabels(ax1_labels.astype(int))

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning.png', dpi=400, bbox_inches='tight')


    ''' Plot for bird across all sessions - sorted by activity at end '''
    end_start_fr = 5
    end_end_fr = 15
    end_start_idx = t_begin+t_pre+t_end-end_start_fr
    end_end_idx = t_begin+t_pre+t_end+end_end_fr
    end_response = np.sum(bird_feeder_responses[:, end_start_idx:end_end_idx],  axis=1)/(t_begin+t_end)
    sort_idx = np.argsort(end_response)
    pct_upstart = np.round(n_cells_upstart/n_cells_total*100, 2)
    pct_upend = np.round(n_cells_upend/n_cells_total*100, 2)

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
    ax[0].set_title(fr"{n_cells_upstart}/{n_cells_total} $\uparrow$ approach", fontsize=axis_label)
    ax[1].set_title(fr"{n_cells_upend}/{n_cells_total} $\uparrow$ departure", fontsize=axis_label)
    f.suptitle(f'all feeder-aligned responses for {bird}', y=0.95, fontsize=title_size)

    # ticks
    ax[0].set_xticks(np.arange(0, t_pre+t_begin+1/dt, 1/dt))
    ax[1].set_xticks(np.arange(0, t_post+t_end+1/dt, 1/dt))
    ax0_labels = (np.arange(-t_pre, t_begin+1/dt, 1/dt))*dt
    ax1_labels = (np.arange(-t_end, t_post+1/dt, 1/dt))*dt
    ax[0].set_xticklabels(ax0_labels.astype(int))
    ax[1].set_xticklabels(ax1_labels.astype(int))

    # add a colorbar
    cax = f.add_axes([0.93, 0.62, 0.02, 0.22]) # [left, bottom, width, height]
    cbar = f.colorbar(im1, cax=cax, orientation='vertical')
    cbar.set_label('activity (z-score)', fontsize=tick_label)
    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.05, '-1 std', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=tick_label)
    cbar.ax.text(0.5, 1.02, '1 std', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=tick_label)

    f.savefig(f'{save_figs}/{bird}_feeder_tuning_altsort.png', dpi=400, bbox_inches='tight')

