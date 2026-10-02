function status = addSessionWaveformsV2(datFile, ksDir, fs, overwrite)
% Add two fields to ksDir/waveformStruct.mat, computed for the same units in
% the same order as the existing struct (wvStruct.goodIDs), so every row
% still lines up with goodIDs, keep_cells and aligned_spikes:
%
%   waveFormsMean_v2 : (n_t x nCh x nUnits) like waveFormsMean, but
%       high-pass only and with each spike re-aligned to the unit's template
%       (getSpikeWaveformV2). Same window, same channels (all .dat rows).
%   spike_shift : (nWF x nUnits) shift applied to each averaged spike, in
%       samples at fs (+ = re-centred later than the KS spike time). NaN-padded
%       for units with fewer than nWF usable spikes; all NaN for units with
%       none. Units with < 50 spikes are not re-aligned (all zeros).
%
% Spike selection matches getSessionWaveforms (same edge exclusions, up to
% 1000 random spikes per unit), seeded so reruns are identical. The original
% subset was not seeded, so v1 and v2 average different random spikes.
%
% The original file is copied once to waveformStruct_v1_backup.mat.

if nargin < 4, overwrite = false; end
wsFile = fullfile(ksDir, 'waveformStruct.mat');
S = load(wsFile, 'wvStruct');
wvStruct = S.wvStruct;
if isfield(wvStruct, 'waveFormsMean_v2') && ~overwrite
    status = "skipped: v2 already present";
    fprintf('  %s\n', status);
    return
end

% window and channel count straight from the existing struct, so v2 matches v1
nWF = 1e3;
spkOffset = wvStruct.spkOffset;
spkDur = wvStruct.spkDur;
nCh = size(wvStruct.waveFormsMean, 2);
spkDurSamples = round(spkDur*fs);
assert(size(wvStruct.waveFormsMean, 1) == spkDurSamples, ...
    'waveFormsMean has %d samples, expected %d at fs = %g', ...
    size(wvStruct.waveFormsMean, 1), spkDurSamples, fs);

f = dir(datFile);
assert(~isempty(f), 'data file not found: %s', datFile);
assert(mod(f.bytes, nCh*2) == 0, ...
    '%s is not a whole number of %d-channel int16 samples', datFile, nCh);
nSamp = f.bytes/(nCh*2);
mmf = memmapfile(datFile, 'Format', {'int16', [nCh nSamp], 'x'});

spkSamp = readNPY(fullfile(ksDir, 'spike_times.npy'));
sID = double(readNPY(fullfile(ksDir, 'spike_clusters.npy')));   % convert once, not per unit
chanMap = double(readNPY(fullfile(ksDir, 'channel_map.npy'))) + 1;   % 1-based .dat rows
assert(max(chanMap) <= nCh, 'channel_map refers to row %d of a %d-channel file', max(chanMap), nCh);
goodIDs = double(wvStruct.goodIDs(:));
numUnits = numel(goodIDs);

waveFormsMean_v2 = nan(spkDurSamples, nCh, numUnits);
spike_shift = nan(nWF, numUnits);
rng(0);
for u = 1:numUnits
    t = double(spkSamp(sID == goodIDs(u)))/fs;
    t((t + spkDur*10)*fs > nSamp) = [];          % as getSessionWaveforms
    t((t - spkOffset*10)*fs < 1) = [];
    if isempty(t), continue; end                 % unit stays NaN, as in v1
    t = sort(t(randperm(numel(t), min(nWF, numel(t)))));
    [wf, lags] = getSpikeWaveformV2(mmf, t, fs, spkOffset, spkDur, chanMap);
    waveFormsMean_v2(:, :, u) = wf;
    spike_shift(1:numel(lags), u) = lags;
    if mod(u, 25) == 0 || u == numUnits
        fprintf('  %d/%d units\n', u, numUnits);
    end
end

backup = fullfile(ksDir, 'waveformStruct_v1_backup.mat');
if ~isfile(backup)
    copyfile(wsFile, backup);
end
wvStruct.waveFormsMean_v2 = waveFormsMean_v2;
wvStruct.spike_shift = spike_shift;
save(wsFile, 'wvStruct');
status = sprintf("ok: %d units", numUnits);
fprintf('  %s\n', status);
end
