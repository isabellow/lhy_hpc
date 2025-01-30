%% Get unsorted ephys data
% Grab a subset of data from the raw data binary file (amplifier.dat)

% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
root_dir = 'Z:\Isabel\data\hpc_implants\ROS105\ROS105_250124\ROS105_250124_112602\';
save_dir = fullfile(fileparts(root_dir),'raw_ephys_output'); 
save_file_name = 'amplifier_data_by_stim_neg.npy';
save_stimt_file = 'stim_t_neg.npy';
mkdir(save_dir)

% Define desired time range to save
t_start = 7596; % start t in seconds
t_duration = 665; % duration in seconds

% Define time window around stim events to save
pre_stim_t = 0.02; % time before stim in seconds
save_duration_t = 0.05; % total time in seconds around each stim

%% Save only around stim times
% Get the digital input (stim times on dig in ch2)
[dig_data, h] = readIntanDig(root_dir, t_duration, t_start);
stim_in = dig_data(2, :);
n_samples = length(stim_in);

% Get the indices defining the stim times
stim_start_idx = find(diff(stim_in) == 1) + 1;
stim_end_idx = find(diff(stim_in) == -1);
if length(stim_end_idx) < length(stim_start_idx)
    stim_start_idx = stim_start_idx(1:end-1);
elseif length(stim_start_idx) < length(stim_end_idx)
    stim_end_idx = stim_end_idx(2:end);
end
n_stim = length(stim_start_idx);

% Convert time window to samples
pre_samples = round(pre_stim_t*h.sample_rate);
total_samples = round(save_duration_t*h.sample_rate);

% Save index
save_start_idx = round(stim_start_idx - pre_samples);
save_end_idx = round(save_start_idx + total_samples);
if save_start_idx(1) < 0
    save_start_idx = save_start_idx(2:end);
    save_end_idx = save_end_idx(2:end);
    stim_start_idx = stim_start_idx(2:end);
end
if save_end_idx(end) > n_samples
    save_start_idx = save_start_idx(1:end-1);
    save_end_idx = save_end_idx(1:end-1);
    stim_start_idx = stim_start_idx(1:end-1);
end
n_stim = length(save_start_idx);

%% Read the Intan data file
[data, header] = readIntanAmp(root_dir, t_duration, t_start);
[n_channels, n_samples] = size(data);
data_by_stim = zeros(n_channels, total_samples, n_stim);
for i = 1:n_stim
    s = save_start_idx(i);
    e = save_end_idx(i);
    data_by_stim(:, :, i) = data(:, s:e-1);
end

% get the absolute stim sample indices
stim_t = (stim_start_idx + (t_start*h.sample_rate))-1;

% Save as a numpy array and struct
writeNPY(data_by_stim, fullfile(save_dir, save_file_name));
writeNPY(stim_t, fullfile(save_dir, save_stimt_file));
save(fullfile(save_dir, 'intan_info.mat'), 'header')