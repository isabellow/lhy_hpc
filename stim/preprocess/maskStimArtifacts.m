%% Blank out stim artifact and antidromic response from raw data
% Load (a copy of!) the raw data binary file (amplifier_copy.dat)
% Identify the stim times
% Blank out the stim artifact and antidromic hash from the file

% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
root_dir = 'Z:\Isabel\data\hpc_implants\ROS105\ROS105_250124\ROS105_250124_112602\';
amp_file_name = 'amplifier_stim_blanked.dat';

% Define the stim period (err on a little extra time)
t_start = 7590; % start t in seconds
t_duration = 675; % duration in seconds

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

% Get the indices defining the stim start times and account for the buffer
stim_start_idx = find(diff(stim_in) == 1) + 1;
n_stim = length(stim_start_idx);
mask_start_idx = round(stim_start_idx - buffer_samples);

% Convert to seconds
mask_start_times = (mask_start_idx/h.sample_rate) + t_start;

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



