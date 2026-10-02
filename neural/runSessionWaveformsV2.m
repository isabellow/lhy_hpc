function T = runSessionWaveformsV2(sessionCsv, overwrite)
% Run addSessionWaveformsV2 on every session listed by
% export_waveform_v2_sessions.py. One session failing does not stop the
% rest; per-session status is written next to the csv as *_status.csv.
%
%   runSessionWaveformsV2('Z:/Isabel/data/lhy_implants/waveform_v2_sessions.csv')
%   runSessionWaveformsV2(csvFile, true)    % recompute sessions that have v2

if nargin < 2, overwrite = false; end
addpath(genpath('C:\Users\Isabel\Documents\code\npy-matlab\'))   % readNPY

opts = detectImportOptions(sessionCsv, 'Delimiter', ',');
opts = setvartype(opts, opts.VariableNames, 'string');   % keep session IDs as text
T = readtable(sessionCsv, opts);
T.status = strings(height(T), 1);
for i = 1:height(T)
    fprintf('\n[%d/%d] %s_%s\n', i, height(T), T.bird(i), T.session_id(i));
    try
        T.status(i) = addSessionWaveformsV2(char(T.dat_file(i)), char(T.ks_dir(i)), ...
            str2double(T.fs(i)), overwrite);
    catch err
        T.status(i) = "error: " + string(err.message);
        fprintf(2, '  %s\n', T.status(i));
    end
end
[p, n] = fileparts(sessionCsv);
writetable(T, fullfile(p, n + "_status.csv"));
disp(T(:, {'bird', 'session_id', 'status'}))
end
