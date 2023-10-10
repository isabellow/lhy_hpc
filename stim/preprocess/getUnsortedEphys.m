%% Get unsorted ephys data
% Grab a subset of data from the raw data binary file (amplifier.dat)

% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
root_dir = 'Z:\Isabel\data\hpc_implants\SPP47\SPP47_231009\SPP47_231009_100756\';
save_dir = fullfile(fileparts(root_dir),'raw_ephys_output'); 
mkdir(save_dir)

% Define desired time range to save
t_start = 7260; % start t in seconds
t_duration = 300; % duration in seconds

% Read the Intan data file
[data, header] = readIntanAmp(root_dir, t_duration, t_start);

% Save as a numpy array and struct
writeNPY(data, fullfile(save_dir, 'amplifier_data.npy'));
save(fullfile(save_dir, 'intan_info.mat'), 'header')