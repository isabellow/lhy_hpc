%% Get ephys + position by frame -- concatenating two sessions split by a noise event
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set file paths
addpath(genpath('C:\Users\ilow1\Documents\code\lhy_hpc\utils\')) % functions to read Intan files
bird_id = 'LMN88';
session_date = '260722';
session_root = ['Z:\Isabel\data\lhy_implants\', bird_id, '\'];
% session_root = 'C:\Users\Isabel\Documents\data_temp\';
session_dir = [bird_id, '_', session_date '\'];

% the two sessions to concatenate, in chronological order
ephys_dirs = {'rec_1\LMN88_260722_110459\', 'rec_2\LMN88_260722_115951\'};

% frames to keep, per session
frame_rate = 50; % Hz
start_frame_idx = [0*frame_rate + 1, 0*frame_rate + 1]; % first frame to keep, per session
end_frame_idx   = [NaN, NaN]; % last frame to keep, per session (NaN = keep all frames found)

% for saving
save_dir = fullfile(fileparts(session_root), session_dir, 'behavior_data');
save_file_name = 'frame_times.npy';
mkdir(save_dir)

%% load + trim frame times from each session separately
n_sessions = length(ephys_dirs);
framet_all = cell(n_sessions, 1);
n_samples_all = zeros(n_sessions, 1);
sample_rates = zeros(n_sessions, 1);
vid_start_sample = zeros(n_sessions, 1);  % first sample to keep, this session's own (raw) clock, 1-indexed
vid_end_sample = zeros(n_sessions, 1);    % last sample to keep

for i_sess = 1:n_sessions
    data_dir = fullfile(fileparts(session_root), [session_dir, ephys_dirs{i_sess}]);

    % Get the digital input (frame times on dig in ch1)
    [dig_data, header_sess] = readIntanDig(data_dir); % read in all the frame times
    frame_in = dig_data(1, :);
    n_samples_all(i_sess) = length(frame_in);
    sample_rates(i_sess) = header_sess.sample_rate;

    % Get the indices defining the frame times
    frame_starts = find(diff(frame_in) == 1) + 1;
    frame_ends = find(diff(frame_in) == -1);
    if length(frame_ends) < length(frame_starts)
        disp(['warning! session ', num2str(i_sess), ': Intan ends before video'])
    elseif length(frame_starts) < length(frame_ends)
        disp(['warning! session ', num2str(i_sess), ': Intan starts before video'])
    end

    % check dt
    frame_dt = diff(frame_starts) / header_sess.sample_rate;
    disp(['session ', num2str(i_sess), ' frame dt (s): ', num2str(unique(round(frame_dt, 2))')])

    % trim this session's frames
    this_end_idx = end_frame_idx(i_sess);
    if isnan(this_end_idx)
        this_end_idx = length(frame_starts);
    end
    frame_starts = frame_starts(start_frame_idx(i_sess):this_end_idx);

    % save the sample numbers for cropping to video length
    pad_samples = round(header_sess.sample_rate / frame_rate);
    vid_start_sample(i_sess) = frame_starts(1);
    vid_end_sample(i_sess) = frame_starts(end) + pad_samples;
    if vid_end_sample(i_sess) > n_samples_all(i_sess)
        disp('warning: ephys ends before final vid frame completes')
        disp('cropping to n video frames - 1')
        disp('note: check video alignment to account for dropped frame(s)')
        vid_end_sample(i_sess) = vid_end_sample(i_sess) - pad_samples;
    end

    assert(vid_end_sample(i_sess) < n_samples_all(i_sess), 'ephys ends before final vid frame completes!')

    % save the frame time relative to the behavior start time
    framet_all{i_sess} = (frame_starts(:) - vid_start_sample(i_sess)) / header_sess.sample_rate;

    % save the header
    if i_sess == 1
        header = header_sess;
    end
end

% sample rate must match across sessions
assert(numel(unique(sample_rates)) == 1, 'sample rates differ between sessions!')

%% offset each session's frame times by the (ephys) duration of the preceding session(s)
n_samples_cropped = vid_end_sample - vid_start_sample + 1;
session_boundary_samples = cumsum(n_samples_cropped);
session_offset_sec = [0; session_boundary_samples(1:end-1) / sample_rates(1)];

framet = [];
for i_sess = 1:n_sessions
    framet = [framet; framet_all{i_sess} + session_offset_sec(i_sess)];
end

% confirm dt
frame_dt = diff(framet);
disp(['concatenated frame time dt (s): ', num2str(unique(round(frame_dt, 2))')])

%% save to a numpy array
writeNPY(framet, fullfile(save_dir, save_file_name));
writeNPY(vid_start_sample, fullfile(save_dir, 'vid_starts.npy'));
writeNPY(vid_end_sample, fullfile(save_dir, 'vid_ends.npy'));
save(fullfile(save_dir, 'intan_info.mat'), 'header');
% TODO save these as desired/needed, 'n_samples_all', 'sample_rates', 'session_boundary_samples'
