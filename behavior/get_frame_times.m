%% Get ephys + position by frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
bird_id = 'SLV132';
session_date = '250303';
session_root = ['Z:\Isabel\data\hpc_implants\', bird_id, '\'];
session_dir = [bird_id, '_', session_date '\'];
ephys_dir = 'SLV132_250303_100940\';

data_dir = fullfile(fileparts(session_root), [session_dir, ephys_dir]);

% for saving
save_dir = fullfile(fileparts(session_root), session_dir, 'behavior_data'); 
save_file_name = 'frame_times.npy';
mkdir(save_dir)

% % time range to look for frame inputs - optional
% start_t = 0; % start t in seconds
% duration_t = ((3*60 + 7)*60 + 0); % duration in seconds

% frames to keep
frame_rate = 50;
start_frame_idx = 0*frame_rate + 1; % first frame to keep
end_frame_idx = ((2*60 + 49)*60 + 11)*50; % number of frames to keep
end_frame_idx = 468092;

%% load the frame times from Intan
% Get the digital input (frame times on dig in ch1)
% [dig_data, h] = readIntanDig(data_dir, duration_t, start_t); % read in select frame times
[dig_data, h] = readIntanDig(data_dir); % read in all the frame times
frame_in = dig_data(1, :);
n_samples = length(frame_in);

% Get the indices defining the frame times
frame_starts = find(diff(frame_in) == 1) + 1;
frame_ends = find(diff(frame_in) == -1);
if length(frame_ends) < length(frame_starts)
    disp("warning! Intan ends before video")
elseif length(frame_starts) < length(frame_ends)
    disp("warning! Intan starts before video")
end
n_total_frames = length(frame_starts);

% check dt
frame_dt = (diff(frame_starts)) / h.sample_rate;
disp(unique(round(frame_dt, 2)))

%% Trim the frame times and convert to seconds
frame_starts = frame_starts(start_frame_idx:end_frame_idx);
frame_ends = frame_ends(start_frame_idx:end_frame_idx);
framet = frame_starts / h.sample_rate;

%% save to a numpy array
writeNPY(framet, fullfile(save_dir, save_file_name));


