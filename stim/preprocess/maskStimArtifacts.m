%% Blank out stim artifact and antidromic response from raw data
% Load (a copy of!) the raw data binary file (amplifier_copy.dat)
% Identify the stim times
% Blank out the stim artifact and antidromic hash from the file

% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
root_dir = 'C:\Users\Isabel\Documents\data_temp\SLV132_250310\SLV132_250310_122522\';
amp_file_name = 'amplifier_stim_blanked_short.dat';

save_dir = fullfile(fileparts(root_dir),'raw_ephys_output'); 
save_stimt_file = 'stim_t_all.npy';

% Define the stim period (err on a little extra time)
t_start = 1290; % start t in seconds
t_duration = 1400; % duration in seconds

% Define the mask window
buffer_samples = 12; % samples before the stim to include
mask_buffer_samples = 6; % raw data to smooth over on the interpolation edges
% t_mask_duration = 4e-3; % total time (seconds) to blank (including stim + buffer)
t_mask_duration = 20.5e-3; % if blanking hash

%% Identify the stim start times
% Get the digital input (stim times on dig in ch2)
[dig_data, h] = readIntanDig(root_dir, t_duration, t_start);
stim_in = dig_data(2, :);
n_samples = length(stim_in);

% Get the indices defining the stim start/end times
stim_start_idx = find(diff(stim_in) == 1) + 1;
stim_end_idx = find(diff(stim_in) == -1) + 1;

% Remove non-stim events (e.g. M8 turning on)
stim_dur = stim_end_idx - stim_start_idx;
stim_start_idx = stim_start_idx(stim_dur < 8);
n_stim = length(stim_start_idx);

% Account for buffer and convert to seconds
mask_start_idx = round(stim_start_idx - buffer_samples);
mask_start_times = (mask_start_idx/h.sample_rate) + t_start;

%% Save all stim times for data processing
% get the absolute stim sample indices
stim_t = (stim_start_idx + (t_start*h.sample_rate))-1;
writeNPY(stim_t, fullfile(save_dir, save_stimt_file));

%% Overwrite the Intan data file to mask the stim periods
fprintf('\n processing %d stim events \n', n_stim)
for i = 1:n_stim
    % blank out the stim and antidromic response
    t_mask_start = mask_start_times(i);
    [~] = blankIntanAmp(root_dir, amp_file_name,...
                        h, t_mask_duration, t_mask_start, mask_buffer_samples);
                                
    if round(i / 100) == (i / 100)
        disp(['processed ' int2str(i) ' events out of ' int2str(n_stim)]);
    end
end



