function [mean_wave_filt, lags, ref_ch] = getSpikeWaveformV2(memMap, tspike, Fs, ...
    offset_t_final, duration_t_final, chanMap, max_lag_t, n_iter, min_align_spikes)
% Mean spike waveform, high-pass only, after re-aligning each spike to the
% unit's own template. Same window and channel layout as getSpikeWaveform,
% so the output lines up with waveFormsMean.
%
% Differences from getSpikeWaveform
% - No 5 kHz low-pass. That FIR rings with a side lobe at +/-0.1 ms (19% of
%   the peak), which sharp spikes with a weak opposite lobe get measured to.
%   The high-pass is unchanged.
% - Each spike is shifted by the lag (within +/- max_lag_t) that maximises
%   its dot product with the current mean on the reference channel, and the
%   mean is recomputed, n_iter times. Lags are re-centred on their median so
%   the waveform stays aligned to the KS spike time of the majority. This
%   undoes Kilosort placing the spike time on different extrema of a unit
%   with two similar-sized phases.
%
% Inputs
%   memMap : memmapfile of the .dat ({'int16', [nCh nSamp], 'x'})
%   tspike : spike times (s); same index convention as getSpikeWaveform
%   chanMap : 1-based .dat rows Kilosort sorted (channel_map.npy + 1); the
%       reference channel is picked among these
%   max_lag_t : s (default 0.4e-3);  n_iter : default 2
%   min_align_spikes : units with fewer spikes are not re-aligned (lags = 0):
%       a template from a few noisy spikes mostly aligns noise (default 50)
%
% Outputs
%   mean_wave_filt : (n_t x nCh)
%   lags : (nSpk x 1) samples, + = spike re-centred later than its KS time.
%       One per spike used (spikes too close to the file edges are dropped).
%   ref_ch : .dat row used for alignment (1-based)

if nargin < 7 || isempty(max_lag_t), max_lag_t = 0.4e-3; end
if nargin < 8 || isempty(n_iter), n_iter = 2; end
if nargin < 9 || isempty(min_align_spikes), min_align_spikes = 50; end
nCh = memMap.Format{2}(1);
nSampTotal = memMap.Format{2}(2);

b_highpass = firpm(100, [0 10 800 Fs/2]/(Fs/2), [0 0 1 1]);   % as getSpikeWaveform
max_lag = round(max_lag_t*Fs);
duration_ind_final = round(Fs*duration_t_final);
offset_ind_final = round(Fs*offset_t_final);
duration_ind_initial = duration_ind_final*2 + length(b_highpass)*3;   % filter padding
offset_ind_initial = round(duration_ind_initial/2);
wave_mask = (offset_ind_initial - offset_ind_final) + (1:duration_ind_final);
L = duration_ind_initial + 2*max_lag;

start_ind = round(Fs*tspike(:));
ok = (start_ind - offset_ind_initial - max_lag >= 0) & ...
     (start_ind - offset_ind_initial - max_lag + L <= nSampTotal);
start_ind = start_ind(ok);
nSpk = numel(start_ind);
mean_wave_filt = nan(duration_ind_final, nCh);
lags = zeros(nSpk, 1);
ref_ch = NaN;
if nSpk == 0, return; end

% read every spike on every channel once, with max_lag of margin each side
inds = (start_ind' - offset_ind_initial - max_lag) + (1:L)';      % L x nSpk
W = reshape(memMap.Data.x(1:nCh, inds(:)), [nCh, L, nSpk]);       % int16
center = max_lag + (1:duration_ind_initial);                       % columns of W at zero shift

% reference channel: largest peak-to-peak of the unaligned mean within
% +/- 1 ms of the spike, among sorted channels (excludes bad channels, and
% the narrower window is less exposed to time-locked partner units)
m0 = filtfilt(b_highpass, 1, double(mean(W(:, center, :), 3))');  % dur_init x nCh
m0 = m0(wave_mask, :);
s0 = offset_ind_final + 1;                                         % spike sample in the output window
win = max(s0 - round(1e-3*Fs), 1) : min(s0 + round(1e-3*Fs), duration_ind_final);
ptp = max(m0(win, chanMap), [], 1) - min(m0(win, chanMap), [], 1);
[~, k] = max(ptp);
ref_ch = chanMap(k);

% align on the reference channel
if nSpk >= min_align_spikes
    ref = filtfilt(b_highpass, 1, double(reshape(W(ref_ch, :, :), L, nSpk)));
    core = max_lag + wave_mask;            % rows of ref holding the output window at zero shift
    for it = 1:n_iter
        shifted = zeros(duration_ind_final, nSpk);
        for s = 1:nSpk
            shifted(:, s) = ref(core + lags(s), s);
        end
        tmpl = mean(shifted, 2);
        tmpl = tmpl - mean(tmpl);
        score = zeros(2*max_lag + 1, nSpk);
        for lag = -max_lag:max_lag
            seg = ref(core + lag, :);
            score(lag + max_lag + 1, :) = tmpl' * (seg - mean(seg, 1));
        end
        [~, best] = max(score, [], 1);
        lags = best(:) - max_lag - 1;
        lags = lags - round(median(lags));
        lags = min(max(lags, -max_lag), max_lag);
    end
end

% mean of the shifted spikes, then filter (filtering is linear)
acc = zeros(nCh, duration_ind_initial);
for s = 1:nSpk
    acc = acc + double(W(:, center + lags(s), s));
end
mean_wave_filt = filtfilt(b_highpass, 1, (acc / nSpk)');
mean_wave_filt = mean_wave_filt(wave_mask, :);
end
