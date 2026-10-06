import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats


''' Clustering analysis based on Payne et al. 2021 - for each cell '''
def get_waveform_params(waveform_struct):
    # get avg waveforms and reshape to (n_cells, n_channels, n_timpoints)
    mean_waveforms = np.transpose(waveform_struct['waveFormsMean'], (2, 1, 0)) 
    n_cells = mean_waveforms.shape[0]

    # best channel for each waveform, pythonic indexing
    max_site = waveform_struct['max_site'] - 1

    return  mean_waveforms, max_site

def clu_waveforms_gmm(width, asymm, log_fr, prob_thresh=0.5):
    '''
    use a Gaussian mixture model to identify putative
    inhibitory vs. excitatory neurons based on waveform properties

    prob_thresh : float
        probability cut-off for including a cell in each cluster
    '''
    # consolidate the params for clustering
    wf_params = np.column_stack([asymm, width])
    wf_params = np.column_stack([wf_params, log_fr])

    # fit a Gaussian mixture model
    wf_model = GaussianMixture(n_components=2)
    wf_model.fit(wf_params)

    # find the probability that each cell belongs to each cluster
    probs = wf_model.predict_proba(wf_params)

    # label each cell by its best cluster (excitatory is clu1)
    labels = probs > prob_thresh
    n_cells_clu = np.sum(labels, axis=0)
    clu_idx = np.argsort(n_cells_clu)
    sorted_labels = labels[:, clu_idx]

    clu1_idx = sorted_labels[:, -1]
    clu2_idx = sorted_labels[:, 0]
    clu_none_idx = np.abs((clu1_idx + clu2_idx) - 1).astype(bool)

    return clu1_idx, clu2_idx, clu_none_idx


def clu_waveforms_kmeans(width, asymm, log_fr):
    '''
    use k-means clustering to identify putative
    inhibitory vs. excitatory neurons based on waveform properties
    '''
    # consolidate the params for clustering
    wf_params = np.column_stack([asymm, width])
    wf_params = np.column_stack([wf_params, log_fr])

    # cells with any non-finite property (no waveform, or 0 Hz -> log -inf)
    # are left out of the fit and returned as False in both clusters
    valid = np.all(np.isfinite(wf_params), axis=1)

    # zscore for kmeans
    zscore_params = stats.zscore(wf_params[valid], axis=0)

    # kmeans cluster
    kmeans = KMeans(n_clusters=2, n_init=100, random_state=1234)
    kmeans.fit(zscore_params)
    H = kmeans.cluster_centers_
    W_raw = np.full(valid.size, -1)
    W_raw[valid] = kmeans.labels_

    # get the cluster indices
    clu1_idx = W_raw == 1
    clu2_idx = W_raw == 0

    # swap as needed so interneurons are clu2
    if np.sum(clu1_idx) < np.sum(clu2_idx):
        clu1_idx_temp = clu2_idx.copy()
        clu2_idx = clu1_idx
        clu1_idx = clu1_idx_temp

    return clu1_idx, clu2_idx

''' Waveform parameters '''
def calc_spike_width(wf, sampling_rate=30000):
    '''
    Calculate the time (ms) from the trough to the subsequent peak
    of an average waveform

    Params
    ------
    wf : average waveform for the channel with the waveform peak; shape (n_timepoints,)
    '''
    trough_idx = np.argmin(wf)
    spk_w_samples = np.argmax(wf[trough_idx:])
    spk_w = (spk_w_samples / sampling_rate)*1000
    return spk_w

def calc_spike_width_polarity(wf, sampling_rate=30000):
    """
    Calculate the waveform width, acounting for polarity.

    For negative-going spikes, time (ms) from the trough to 
    the subsequent peak of an average waveform.

    For positive-going, the mirror image--time from the peak
    to the subsequent trough. A cell is called 'positive' when
    the largest excursion from baseline is upward.

    LEGACY -- kept so existing results are reproducible; use waveform_shape.
    Known failure modes: (1) the search runs to the end of the 4 ms window
    with argmax/argmin, so a unit with no clear opposite lobe gets the last
    sample or a late noise/overlap bump (widths up to ~2.5 ms); (2) polarity
    is a global max vs |min| test, so near-symmetric positive-first units
    flip to 'negative' on noise and are measured from their trough to the
    filter side-lobe that follows it (~0.45-0.55 ms); (3) 33 us quantisation.
    """
    # no waveform (unit skipped in getSessionWaveforms.m): nan, not a width of 0
    wf = np.asarray(wf, dtype=float)
    if wf.size == 0 or not np.all(np.isfinite(wf)):
        return np.nan

    # determine polarity
    positive = wf.max() > abs(wf.min())

    # index of the first peak (pos or neg)
    first_idx = int(np.argmax(wf)) if positive else int(np.argmin(wf))

    # index of the next deviation
    rest = wf[first_idx:]
    spk_w_samples = int(np.argmin(rest)) if positive else int(np.argmax(rest))

    # width
    spk_w = spk_w_samples / sampling_rate * 1e3

    return spk_w

# def half_width_ms(wf, fs=30000):
#     """Full width at half maximum of the dominant peak, in ms.

#     Polarity-agnostic, so it can be compared across up- and down-going spikes
#     """
#     i = int(np.argmax(np.abs(wf)))
#     half = wf[i] / 2.0
#     sgn = np.sign(wf[i])

#     def cross(step):
#         j = i
#         while 0 <= j + step < wf.size and sgn * (wf[j + step] - half) > 0:
#             j += step
#         k = j + step
#         if not (0 <= k < wf.size):
#             return float(j)
#         denom = wf[k] - wf[j]
#         return float(j) if denom == 0 else j + (half - wf[j]) / denom * step

#     return (cross(1) - cross(-1)) / fs * 1e3

def calc_amp_assym(wf):
    '''
    Calculate the relative height of the two positive peaks flanking the trough
    of an average waveform
    
    (b-a)/(b+a) where a is the first peak, b is the second
    −1 when 1st peak is there and 2nd is not
    0 when 1st = 2nd
    +1 when 1st is not there and 2nd is there

    Params
    ------
    wf : average waveform for the channel with the waveform peak; shape (n_timepoints,)

    LEGACY -- use waveform_shape()['asymmetry']. Uses the global minimum even
    for positive units, searches the whole window on both sides, has no
    baseline subtraction, and can leave [-1, 1] when a or b is negative.
    '''
    # nan when undefined: no waveform, or the minimum at the first sample
    # (nothing before the trough). All-NaN waveforms land here too, since
    # np.argmin returns 0 for them.
    wf = np.asarray(wf, dtype=float)
    if wf.size == 0 or not np.all(np.isfinite(wf)):
        return np.nan
    trough_idx = np.argmin(wf)
    if trough_idx == 0:
        return np.nan
    a = np.max(wf[:trough_idx])
    b = np.max(wf[trough_idx:])
    return (b - a) / (b + a)

def calc_cum_rates(log_fr):
    # Calculate the cumulative probability of the firing rates
    rates, bin_vals = np.histogram(log_fr, bins=50)
    cumulative_rates = np.cumsum(rates)
    norm_rates = (cumulative_rates - np.min(cumulative_rates)) / (np.max(cumulative_rates) - np.min(cumulative_rates))
    return norm_rates, bin_vals[:-1]



''' ------------------------------------------------------------------------
Waveform features, v2: polarity-aware shape, spatial footprint, stability

Replaces calc_spike_width_polarity / calc_amp_assym for new analyses (the
legacy functions are left untouched above so old results stay reproducible).

What changed, and why
- Polarity is decided near the KS spike time with an explicit 'ambiguous'
  band, instead of a global max > |min| test that flips on noise for
  near-symmetric units (flipped units get measured to a filter side-lobe
  and pile up at ~0.45-0.55 ms).
- The opposite-sign lobe is searched for in a bounded window and must be a
  real local extremum (scipy find_peaks, with prominence). No lobe -> nan,
  instead of the end of the 4 ms window (the source of >1 ms widths).
- Waveforms are baseline-subtracted and cubic-spline upsampled (10x, as in
  getWaveformProps.m), so widths are not quantised to 33 us.
- Asymmetry uses the same polarity-aligned waveform, bounded windows, and
  lobes clipped at zero, so it stays in [-1, 1].
- Spatial footprint is computed on KS channel order with channel_positions,
  on the peak channel's shank only, with a noise floor (so spread is not
  driven by summed noise on distant channels).
- Stability: per-chunk rate and KS amplitude, built on spike_amplitudes.py,
  so a rate can be measured only while the unit is well recorded.

Conventions
- waveFormsMean from getSessionWaveforms.m: spkOffset = 1.5 ms at 30 kHz, so
  the KS spike time sits at sample 45 (0-based) of each 120-sample waveform.
- Amplitudes are in the units of waveFormsMean (Intan int16 bits;
  0.195 uV/bit for RHD amplifiers).
------------------------------------------------------------------------ '''
from scipy.interpolate import CubicSpline

FS = 30000
SPIKE_IDX = 45

SHAPE_KEYS = ['amp', 'pk_trough_ratio', 'polarity', 'width_ms', 'half_width_ms',
              'asymmetry', 'pre_ratio', 'post_ratio', 'main_offset_ms',
              'deriv_ratio', 'asymmetry_raw', 'trough_peak_lag_ms', 'deriv_ratio_raw']
SPATIAL_KEYS = ['n_ch_above', 'extent_um', 'weighted_dist_um', 'decay_um',
                'latency_ms_per_100um', 'latency_r2']


def ks_channel_waveforms(waveform_struct, ks_path, key='waveFormsMean'):
    '''
    Mean waveforms re-ordered to Kilosort channel order, with probe positions.

    waveFormsMean holds every row of amplifier.dat; channel_map.npy lists the
    rows KS actually sorted (0-based), and channel_positions.npy gives the
    (x, y) of each of those, in um. Using these two files directly means
    channels KS dropped (bad / unconnected) are excluded, and the peak
    channel is found in the same channel set used for the spatial features.

    Assumes KS was run on amplifier.dat with its rows in native order (i.e.
    channel_map indexes rows of the .dat file), which is the standard setup.

    Params
    ------
    waveform_struct : dict with 'waveFormsMean', shape (n_t, n_ch_dat, n_cells)
    ks_path : str   full path to the KS output folder, ending in '/'
    key : which waveforms to use; 'waveFormsAligned' or 'waveFormsMedian' if
        getSessionWaveforms.m was re-run with getSpikeWaveformAligned

    Returns
    -------
    wfs : (n_cells, n_ks_ch, n_t)
    pos : (n_ks_ch, 2) um
    chan_map : (n_ks_ch,) row of amplifier.dat for each KS channel
    '''
    wfs = np.asarray(waveform_struct[key], dtype=float)
    if wfs.ndim == 2:            # MATLAB drops the trailing dim for a single unit
        wfs = wfs[:, :, None]
    wfs = np.transpose(wfs, (2, 1, 0))
    chan_map = np.asarray(np.load(f"{ks_path}channel_map.npy")).astype(int).ravel()
    pos = np.asarray(np.load(f"{ks_path}channel_positions.npy"), dtype=float)
    if chan_map.max() >= wfs.shape[1]:
        raise ValueError(f"channel_map refers to row {chan_map.max()}, but "
                         f"waveFormsMean has only {wfs.shape[1]} channels")
    if pos.shape[0] != chan_map.size:
        raise ValueError("channel_positions and channel_map disagree in length")
    return wfs[:, chan_map, :], pos, chan_map


def shank_from_positions(pos, gap_um=100):
    '''
    Shank index per channel, from gaps in x. Cambridge NeuroTech multi-shank
    probes have shanks >= 150 um apart and columns within a shank much closer,
    so a 100 um gap separates shanks. Pass your own shank vector instead if
    you have one from the probe file.
    '''
    x = np.asarray(pos)[:, 0]
    ux = np.unique(x)
    grp = np.concatenate(([0], np.cumsum(np.diff(ux) > gap_um)))
    return grp[np.searchsorted(ux, x)]


def _ms_to_samp(ms, fs):
    return int(round(ms * 1e-3 * fs))


def _upsample(wf, factor):
    '''cubic-spline upsample; sample i of the input lands at i*factor'''
    if factor <= 1:
        return np.asarray(wf, dtype=float)
    n = wf.size
    return CubicSpline(np.arange(n), wf)(np.arange((n - 1) * factor + 1) / factor)


def _baseline(wf, n_base):
    '''subtract the median of the first n_base samples (last axis)'''
    wf = np.asarray(wf, dtype=float)
    return wf - np.median(wf[..., :n_base], axis=-1, keepdims=True)


def _half_width(z, i):
    '''full width at half of the trough z[i] (z[i] < 0), in samples'''
    half = z[i] / 2.0

    def cross(step):
        j = i
        while 0 <= j + step < z.size and z[j + step] < half:
            j += step
        k = j + step
        if not (0 <= k < z.size):
            return np.nan                 # never came back up: undefined
        return j + (half - z[j]) / (z[k] - z[j]) * step

    return cross(1) - cross(-1)


def waveform_shape(wf, fs=FS, spike_idx=SPIKE_IDX, upsample=10,
                   baseline_ms=0.5, main_win_ms=0.5, pre_win_ms=0.5,
                   post_win_ms=1.2, min_prom=0.05, ambiguous=(0.8, 1.25)):
    '''
    Shape features of one unit's mean waveform on its peak channel.

    The waveform is first flipped, if needed, so that its dominant extremum
    is a trough (z = -wf for peak-dominant units). All features are then
    defined on z, so a positive unit's "width" is peak -> following trough.
    Keep polarity groups separate downstream: for a positive-first,
    triphasic unit that interval is the depolarising phase, not the
    repolarisation measured for negative units.

    Params
    ------
    wf : (n_t,) mean waveform, peak channel
    spike_idx : sample of the KS spike time in wf (before upsampling)
    main_win_ms : dominant extremum is searched for within +/- this of spike_idx
    pre_win_ms, post_win_ms : search windows for the flanking lobes.
        1.2 ms post is just above bombcell's maxWvDuration (1150 us)
    min_prom : a flanking lobe must be a local extremum with prominence of at
        least this fraction of the main amplitude
    ambiguous : (lo, hi) band of pk_trough_ratio labelled polarity 0

    Returns
    -------
    dict with
      amp : amplitude of the dominant extremum
      pk_trough_ratio : largest positive / |largest negative| near the spike
      polarity : -1 trough-dominant, +1 peak-dominant, 0 ambiguous
                 (ambiguous units are measured as if trough-dominant)
      width_ms : dominant extremum -> largest opposite lobe within post_win_ms;
                 nan if there is no such lobe
      half_width_ms : FWHM of the dominant extremum
      asymmetry : (b - a) / (b + a), a / b = lobe before / after the extremum
      pre_ratio, post_ratio : a / amp, b / amp
      main_offset_ms : time of the dominant extremum minus the KS spike time
                       (large values => KS aligned to a different phase)
      deriv_ratio : log10(steepest rise / steepest fall) around the dominant
                    extremum, on the polarity-aligned waveform; the "peak
                    derivative ratio" of getWaveformProps.m. Negative when
                    the spike recovers more slowly than it falls.
    Un-mirrored versions, measured on the waveform as recorded (not flipped),
    so they vary continuously across the polarity boundary; use these instead
    of width / asymmetry when clustering positive and negative units together:
      asymmetry_raw : asymmetry around the most negative point
      trough_peak_lag_ms : time from the most negative point near the spike to
                    the largest positive point from pre_win_ms before it to
                    post_win_ms after it (the windows width_ms uses). Equals
                    width_ms for a classic negative spike; negative when the
                    peak comes first (positive-first / peak-dominant units),
                    so it acts as a signed width across polarities
      deriv_ratio_raw : log10(steepest rise / steepest fall) in that same
                    window; equals deriv_ratio for trough-dominant units and
                    its negative for peak-dominant ones
    '''
    out = dict.fromkeys(SHAPE_KEYS, np.nan)
    wf = np.asarray(wf, dtype=float)
    if wf.size == 0 or not np.all(np.isfinite(wf)):
        return out

    # upsample the waveform to avoid aliasing
    y = _upsample(_baseline(wf, max(_ms_to_samp(baseline_ms, fs), 1)), upsample)
    f = fs * max(upsample, 1)
    s0 = spike_idx * max(upsample, 1)
    w = _ms_to_samp(main_win_ms, f)
    lo, hi = max(s0 - w, 0), min(s0 + w + 1, y.size)

    # classify polarity and flip if needed
    pmax, nmin = y[lo:hi].max(), y[lo:hi].min()
    ratio = pmax / -nmin if nmin < 0 else np.inf
    pol = -1 if ratio < ambiguous[0] else (1 if ratio > ambiguous[1] else 0)
    z = -y if pol == 1 else y
    i = lo + int(np.argmin(z[lo:hi]))
    amp = -z[i]
    if amp <= 0:
        return out

    # lobe after the extremum: must be a genuine local max inside the window
    post = z[i:min(i + _ms_to_samp(post_win_ms, f) + 1, z.size)]
    pk, _ = find_peaks(post, prominence=min_prom * amp)
    if pk.size:
        j = pk[np.argmax(post[pk])]
        width = j / f * 1e3
        b = max(post[j], 0.0)
    else:
        width = np.nan
        b = max(post.max(), 0.0)

    # lobe before the extremum
    pre = z[max(i - _ms_to_samp(pre_win_ms, f), 0):i]
    a = max(pre.max(), 0.0) if pre.size else 0.0

    # steepest rise / steepest fall around the dominant extremum
    d_lo = max(i - _ms_to_samp(pre_win_ms, f), 0)
    d_hi = min(i + _ms_to_samp(post_win_ms, f) + 1, z.size)
    dz = np.diff(z[d_lo:d_hi])
    deriv_ratio = (np.log10(dz.max() / -dz.min())
                   if dz.size and dz.max() > 0 and dz.min() < 0 else np.nan)

    # un-mirrored features on y as recorded: anchored on the most negative
    # point near the spike, peak searched over the same windows as width_ms
    j_tr = lo + int(np.argmin(y[lo:hi]))
    w_lo = max(j_tr - _ms_to_samp(pre_win_ms, f), 0)
    w_hi = min(j_tr + _ms_to_samp(post_win_ms, f) + 1, y.size)
    j_pk = w_lo + int(np.argmax(y[w_lo:w_hi]))
    pre_raw = y[w_lo:j_tr]
    post_raw = y[j_tr:w_hi]
    a_raw = max(pre_raw.max(), 0.0) if pre_raw.size else 0.0
    b_raw = max(post_raw.max(), 0.0)
    dy = np.diff(y[w_lo:w_hi])
    deriv_raw = (np.log10(dy.max() / -dy.min())
                 if dy.size and dy.max() > 0 and dy.min() < 0 else np.nan)

    out.update(amp=amp, pk_trough_ratio=ratio, polarity=pol, width_ms=width,
               half_width_ms=_half_width(z, i) / f * 1e3,
               asymmetry=(b - a) / (b + a) if (a + b) > 0 else np.nan,
               pre_ratio=a / amp, post_ratio=b / amp,
               main_offset_ms=(i - s0) / f * 1e3,
               deriv_ratio=deriv_ratio,
               asymmetry_raw=(b_raw - a_raw) / (b_raw + a_raw) if (a_raw + b_raw) > 0 else np.nan,
               trough_peak_lag_ms=(j_pk - j_tr) / f * 1e3,
               deriv_ratio_raw=deriv_raw)
    return out


def spatial_footprint(wf_ch, pos, peak_ch, shank=None, fs=FS,
                      spike_idx=SPIKE_IDX, win_ms=0.5, baseline_ms=0.5,
                      rel_thresh=0.2, noise_k=6, radius_um=200, min_fit_ch=4,
                      contiguous=True, rise_tol=0.1):
    '''
    How far one unit's mean waveform spreads across the probe.

    Amplitude per channel is peak-to-peak within +/- win_ms of the spike, so
    it is polarity-agnostic and still catches signals that arrive late on
    other channels (e.g. a spike propagating along an axon). The MATLAB
    neuronSpread (getWaveformProps.m) used the value at the time of the
    peak-channel trough, which misses both.

    A channel counts as carrying the unit only if its amplitude clears both
    rel_thresh * peak amplitude and a noise floor. Without the floor, the
    weighted-distance measure sums noise from every distant channel, which
    makes small units look spatially broad (spread becomes a proxy for SNR).

    Params
    ------
    wf_ch : (n_ch, n_t)  mean waveform on every KS channel
    pos : (n_ch, 2)      channel positions, um
    peak_ch : int        index into wf_ch
    shank : (n_ch,) or None   restricts everything to the peak channel's shank
    noise_k : floor = noise_k * (median across channels of baseline s.d.).
        Peak-to-peak of Gaussian noise over a ~2 ms window is ~5 s.d.
    radius_um : channels used for the decay-length and latency fits
    contiguous : if True (default), only channels reachable from the peak
        channel by stepping between neighbouring channels without amplitude
        rising by more than rise_tol are used. Excludes some contamination from 
        synaptic partners. Check CCGs to be sure. 

    Returns
    -------
    dict with
      n_ch_above : channels over threshold (incl. the peak channel)
      extent_um : distance from peak channel to the farthest channel over threshold
      weighted_dist_um : amplitude-weighted mean distance (MATLAB neuronSpread,
                         with the noise floor subtracted)
      decay_um : length constant of an exponential fit, amp(d) ~ exp(-d / decay)
      latency_ms_per_100um, latency_r2 : linear fit of the time of the largest
          excursion vs. vertical offset from the peak channel. A consistent
          slope with high r2 is what a propagating (axonal) spike looks like;
          somatic spikes are near-synchronous on nearby channels.
    '''
    # check inputs
    out = dict.fromkeys(SPATIAL_KEYS, np.nan)
    wf_ch = np.asarray(wf_ch, dtype=float)
    if not np.all(np.isfinite(wf_ch)):
        return out
    pos = np.asarray(pos, dtype=float)

    # data params
    nb = max(_ms_to_samp(baseline_ms, fs), 2)
    wf_ch = _baseline(wf_ch, nb)
    w = _ms_to_samp(win_ms, fs)
    lo, hi = max(spike_idx - w, 0), min(spike_idx + w + 1, wf_ch.shape[1])
    seg = wf_ch[:, lo:hi]
    amp = seg.max(axis=1) - seg.min(axis=1)

    # set shank and noise floor
    same = (np.ones(len(pos), bool) if shank is None
            else np.asarray(shank) == np.asarray(shank)[peak_ch])
    noise = np.median(np.std(wf_ch[same, :nb], axis=1))
    floor = noise_k * noise
    thr = max(rel_thresh * amp[peak_ch], floor)
    d = np.linalg.norm(pos - pos[peak_ch], axis=1)

    # get spatial spread
    if contiguous:
        same = same & _monotone_region(amp, pos, peak_ch, same & (amp > floor), rise_tol)
    above = same & (amp >= thr)
    above[peak_ch] = True
    out['n_ch_above'] = int(above.sum())
    out['extent_um'] = float(d[above].max())

    excess = np.where(same, np.clip(amp - floor, 0, None), 0)
    if excess.sum() > 0:
        out['weighted_dist_um'] = float((d * excess).sum() / excess.sum())

    # decay, weighted by amplitude
    fit = same & (d <= radius_um) & (amp > floor)
    fit[peak_ch] = True
    if fit.sum() >= min_fit_ch:
        a_n = amp[fit] / amp[peak_ch]
        slope = np.polyfit(d[fit], np.log(a_n), 1, w=a_n)[0]
        out['decay_um'] = float(-1 / slope) if slope < 0 else np.inf

    # propagation latency for somatic vs. axonal classification
    lat_ch = fit & above
    if lat_ch.sum() >= min_fit_ch:
        sub = seg[lat_ch]
        k = np.argmax(np.abs(sub), axis=1)
        # parabolic interpolation for sub-sample timing
        kk = np.clip(k, 1, sub.shape[1] - 2)
        r = np.arange(sub.shape[0])
        y0, y1, y2 = np.abs(sub[r, kk - 1]), np.abs(sub[r, kk]), np.abs(sub[r, kk + 1])
        den = y0 - 2 * y1 + y2
        delta = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1), 0)
        t_ms = (kk + delta) / fs * 1e3
        dy = pos[lat_ch, 1] - pos[peak_ch, 1]
        if np.ptp(dy) > 0:
            p = np.polyfit(dy, t_ms, 1)
            resid = t_ms - np.polyval(p, dy)
            ss = np.sum((t_ms - t_ms.mean()) ** 2)
            out['latency_ms_per_100um'] = float(p[0] * 100)
            out['latency_r2'] = float(1 - resid @ resid / ss) if ss > 0 else np.nan
    return out


def _monotone_region(amp, pos, peak_ch, allowed, rise_tol=0.1):
    '''
    Channels reachable from peak_ch through neighbouring allowed channels
    without amplitude rising more than rise_tol (fraction) at any step.
    Neighbours are channels within 1.5x the smallest inter-channel distance.
    '''
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    step = 1.5 * np.min(D[D > 0])
    reach = np.zeros(len(amp), bool)
    reach[peak_ch] = True
    frontier = [peak_ch]
    while frontier:
        c = frontier.pop()
        for nb in np.flatnonzero((D[c] <= step) & allowed & ~reach):
            if amp[nb] <= amp[c] * (1 + rise_tol):
                reach[nb] = True
                frontier.append(nb)
    return reach


def compute_unit_features(wfs, pos, shank=None, fs=FS, spike_idx=SPIKE_IDX,
                          peak_win_ms=1.0, shape_kw=None, spatial_kw=None):
    '''
    Shape + spatial features for every unit in a session.

    Params
    ------
    wfs : (n_cells, n_ch, n_t)  from ks_channel_waveforms
    pos : (n_ch, 2)
    shank : (n_ch,) or None     defaults to shank_from_positions(pos)

    Returns
    -------
    dict of (n_cells,) arrays: 'peak_ch_ks' (row of channel_positions.npy,
    i.e. KS channel order -- NOT native Intan order and NOT the custom probe
    order; build_data_dict adds 'peak_ch_native' and 'peak_ch_custom'),
    SHAPE_KEYS,
    SPATIAL_KEYS, and 'legacy_width_ms' / 'legacy_asymm' (the old functions
    on the same waveform, for comparing against the existing scatter).
    '''
    shape_kw = shape_kw or {}
    spatial_kw = spatial_kw or {}
    if shank is None:
        shank = shank_from_positions(pos)
    n = wfs.shape[0]
    keys = ['peak_ch_ks', 'legacy_width_ms', 'legacy_asymm'] + SHAPE_KEYS + SPATIAL_KEYS
    out = {k: np.full(n, np.nan) for k in keys}
    w = _ms_to_samp(peak_win_ms, fs)
    for c in range(n):
        wf = wfs[c]
        if not np.all(np.isfinite(wf)):
            continue
        seg = _baseline(wf, _ms_to_samp(0.5, fs))[:, max(spike_idx - w, 0):spike_idx + w + 1]
        pk = int(np.argmax(seg.max(axis=1) - seg.min(axis=1)))
        out['peak_ch_ks'][c] = pk
        for k, v in waveform_shape(wf[pk], fs=fs, spike_idx=spike_idx, **shape_kw).items():
            out[k][c] = v
        for k, v in spatial_footprint(wf, pos, pk, shank=shank, fs=fs,
                                      spike_idx=spike_idx, **spatial_kw).items():
            out[k][c] = v
        out['legacy_width_ms'][c] = calc_spike_width_polarity(wf[pk], sampling_rate=fs)
        try:
            out['legacy_asymm'][c] = calc_amp_assym(wf[pk])
        except ValueError:          # trough at sample 0
            pass
    return out


def unit_stability(spike_frames, amps, n_frames, dt, amp_frac=0.7,
                   ref_pct=None, **chunk_kw):
    '''
    One number per unit: its firing rate while it is well recorded.

    Built on spike_amplitudes: spike_frames / amps come from
    load_spike_amplitudes, per-chunk counts and median amplitudes from
    chunk_amplitudes, and the reference is the same one reference_amplitude
    gives select_trials_by_drift. What this adds:

    - A chunk is 'stable' if its median amplitude >= amp_frac * reference
      (the unit's well-recorded level, not the session median). Stability is
      judged on amplitude only and the rate is then measured in the stable
      chunks; selecting on rate itself would inflate the rate.
    - Chunks with too few spikes to measure amplitude count as stable when the
      nearest measured chunks on both sides are stable: a unit that goes quiet
      between two good chunks is quiet, not gone. Dropping those chunks (as
      select_trials_by_drift does per trial, sensibly for trial selection)
      would bias the rate of sparse units upward.
    - Session-level summaries for classification (below).

    Drift shows up as rate falling together with amplitude (rate_amp_rho > 0).
    Caveat: amplitude also falls during high-rate bursts, so a bursty unit can
    show a modest negative rho for real reasons; 60 s chunks average most of
    that out. bombcell's time-chunk metrics (perc_spikes_missing) are the more
    thorough version of the same idea.

    Params
    ------
    spike_frames, amps : one entry of load_spike_amplitudes output
    n_frames : int     frames in the session (columns of aligned_spikes.npy)
    dt : float         s per frame (1 / fps)
    amp_frac : float   stable if chunk median >= amp_frac * reference; use the
                       same value as select_trials_by_drift's thresh
    ref_pct, **chunk_kw : passed to chunk_amplitudes; defaults are the
        spike_amplitudes REF_* constants (chunks >= 60 s, long enough for
        ~15 spikes, >= 5 spikes to measure, reference = 90th percentile)

    Returns
    -------
    dict with chunk_s (length used, s), mean_rate, stable_rate (Hz),
    frac_stable (of time), presence_ratio, rate_amp_rho (Spearman, across
    measured chunks), amp_drop (1 - min / reference chunk amplitude), and
    per-chunk arrays under 'chunks' (centers and half-width in frames, rate,
    med_amp, stable).
    '''
    import spike_amplitudes       # imported here so this module loads without it

    if ref_pct is None:
        ref_pct = spike_amplitudes.REF_PCT
    ch = spike_amplitudes.chunk_amplitudes(spike_frames, amps, n_frames, dt, **chunk_kw)
    med, counts, dur = ch['med_amp'], ch['counts'], ch['dur']
    measured = np.isfinite(med)
    rate = counts / dur
    n_chunks = med.size

    res = dict(chunk_s=ch['chunk_s'], mean_rate=counts.sum() / dur.sum(),
               stable_rate=np.nan, frac_stable=0.0,
               presence_ratio=np.mean(counts > 0), rate_amp_rho=np.nan,
               amp_drop=np.nan)
    stable = np.zeros(n_chunks, bool)
    if measured.any():
        # same value as spike_amplitudes.reference_amplitude, without re-chunking
        ref = np.nanpercentile(med, ref_pct)
        stable[measured] = med[measured] >= amp_frac * ref
        m_idx = np.flatnonzero(measured)
        for k in np.flatnonzero(~measured):
            left, right = m_idx[m_idx < k], m_idx[m_idx > k]
            if left.size and right.size:
                stable[k] = stable[left[-1]] and stable[right[0]]
        res['amp_drop'] = 1 - np.nanmin(med) / ref if ref > 0 else np.nan
        if measured.sum() >= 5:
            res['rate_amp_rho'] = stats.spearmanr(rate[measured], med[measured])[0]
    if stable.any():
        res['stable_rate'] = counts[stable].sum() / dur[stable].sum()
        res['frac_stable'] = dur[stable].sum() / dur.sum()
    res['chunks'] = dict(centers=ch['centers'], half=ch['half'], rate=rate,
                         med_amp=med, stable=stable)
    return res


def stability_for_units(amp_data, n_frames, dt, **kw):
    '''
    unit_stability for every cell in load_spike_amplitudes output (one entry
    per cluster ID passed to it, e.g. goodIDs, in that order).
    Scalar outputs are returned as (n_units,) arrays; per-chunk data as a list.
    '''
    scal = ['chunk_s', 'mean_rate', 'stable_rate', 'frac_stable', 'presence_ratio',
            'rate_amp_rho', 'amp_drop']
    out = {k: np.full(len(amp_data), np.nan) for k in scal}
    out['chunks'] = []
    for u, (frames, amps) in enumerate(amp_data):
        r = unit_stability(frames, amps, n_frames, dt, **kw)
        for k in scal:
            out[k][u] = r[k]
        out['chunks'].append(r['chunks'])
    return out


ACG_BIN_S = 0.5e-3       # CellExplorer narrow ACG: 0.5 ms bins
ACG_N_BINS = 100         # ... out to 50 ms
TRAIN_KEYS = ['n_spikes_used', 'inv_median_isi_hz', 'cv2', 'burst_index',
              'acg_tau_rise_ms', 'acg_tau_decay_ms', 'acg_tau_burst_ms',
              'acg_refrac_ms', 'acg_fit_r2']


def stable_intervals(chunks, frame_t, dt):
    '''
    Merge a unit's stable stability chunks into (start, end) intervals in
    seconds on the ephys clock (the clock of frame_times.npy).

    chunks : one entry of stability_for_units(...)['chunks']
    frame_t : frame_times.npy (s);  dt : s per frame
    '''
    if chunks is None:
        return []
    frame_t = np.asarray(frame_t, dtype=float)
    n_frames = frame_t.size
    out = []
    for c, ok in zip(chunks['centers'], chunks['stable']):
        if not ok:
            continue
        lo = int(np.clip(c - chunks['half'], 0, n_frames - 1))
        hi = int(np.clip(c + chunks['half'], 0, n_frames - 1))
        start, end = frame_t[lo], frame_t[hi] + dt
        if out and start <= out[-1][1] + 1e-9:     # contiguous with the previous chunk
            out[-1][1] = end
        else:
            out.append([start, end])
    return [tuple(iv) for iv in out]


def _interval_index(spike_t, intervals):
    '''which interval each spike falls in (-1 if none)'''
    idx = np.full(spike_t.size, -1)
    for k, (a, b) in enumerate(intervals):
        idx[(spike_t >= a) & (spike_t < b)] = k
    return idx


def narrow_acg(spike_t, seg=None, bin_s=ACG_BIN_S, n_bins=ACG_N_BINS):
    '''
    Autocorrelogram at lags 0..n_bins bins (one side), in Hz, as CellExplorer
    computes it (CCG, 'norm', 'rate'): spike pairs counted in bins centred on
    multiples of bin_s, divided by bin width and number of reference spikes.
    All spike pairs are counted, not just consecutive ISIs. Pairs spanning two
    different intervals (seg) are not counted, so gaps between stable
    stretches don't create false long lags.
    '''
    t = np.sort(np.asarray(spike_t, dtype=float))
    seg = np.zeros(t.size, int) if seg is None else np.asarray(seg)[np.argsort(spike_t)]
    counts = np.zeros(n_bins + 1)
    max_lag = (n_bins + 0.5) * bin_s
    for k in range(1, t.size):
        d = t[k:] - t[:-k]
        near = d < max_lag
        if not near.any():
            break                     # lags only grow with k
        use = near & (seg[k:] == seg[:-k])
        np.add.at(counts, np.rint(d[use] / bin_s).astype(int), 1)
    return counts / (bin_s * max(t.size, 1))


def _acg_model(x, a, b, c, d, e, f, g, h):
    '''CellExplorer fit_ACG.m: max(c*(exp(-(x-f)/a)-d*exp(-(x-f)/b))+h*exp(-(x-f)/g)+e, 0)'''
    return np.maximum(c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b))
                      + h * np.exp(-(x - f) / g) + e, 0)


def fit_acg(acg):
    '''
    Port of CellExplorer's fit_ACG.m (triple exponential, same start point and
    bounds) to scipy. acg: narrow_acg output (lags 0..50 ms in 0.5 ms bins).
    As in fit_ACG.m, the bins at -0.5, 0 and +0.5 ms are zeroed and the fit
    runs over lags 0.5..50 ms. scipy's bounded least squares is not MATLAB's
    fit(), so individual fits can land in different local minima; compare a
    few units against CellExplorer before relying on exact values.

    Returns dict: tau_decay, tau_rise, tau_burst, refrac (ms), r2 -- nan if
    the fit fails.
    '''
    from scipy.optimize import curve_fit
    out = dict(tau_decay=np.nan, tau_rise=np.nan, tau_burst=np.nan, refrac=np.nan, r2=np.nan)
    y = np.asarray(acg, dtype=float).copy()
    if y.size < 101 or not np.all(np.isfinite(y)) or not np.any(y[1:101] > 0):
        return out
    y[:2] = 0
    x = np.arange(1, 101) * 0.5
    y = y[1:101]
    a0 = [20, 1, 30, 2, 0.5, 5, 1.5, 2]
    lb = [1, 0.1, 0, 0, -30, 0, 0.1, 0]
    ub = [500, 50, 500, 15, 50, 20, 5, 100]
    try:
        p, _ = curve_fit(_acg_model, x, y, p0=a0, bounds=(lb, ub), maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    resid = y - _acg_model(x, *p)
    ss = np.sum((y - y.mean()) ** 2)
    out.update(tau_decay=p[0], tau_rise=p[1], tau_burst=p[6], refrac=p[5],
               r2=1 - np.sum(resid ** 2) / ss if ss > 0 else np.nan)
    return out


def spike_train_features(spike_t, intervals=None, min_spikes=100, burst_isi_s=6e-3):
    '''
    Firing-pattern features of one unit, measured only within intervals (the
    unit's stable stretches; whole train if None). ISIs and ACG pairs that
    span two intervals are not used.

    Returns dict (nan if fewer than min_spikes spikes in the intervals):
      n_spikes_used
      inv_median_isi_hz : 1 / median ISI ("inv. isi" in classifyUnitTypes)
      cv2 : mean of 2|ISI(n+1) - ISI(n)| / (ISI(n+1) + ISI(n)), consecutive
            ISI pairs; ~1 for Poisson firing, < 1 regular, > 1 bursty
      burst_index : fraction of ISIs < burst_isi_s
      acg_tau_rise_ms, acg_tau_decay_ms, acg_tau_burst_ms, acg_refrac_ms,
      acg_fit_r2 : CellExplorer ACG fit (fit_acg)
    '''
    out = dict.fromkeys(TRAIN_KEYS, np.nan)
    t = np.sort(np.asarray(spike_t, dtype=float))
    if intervals is not None:
        seg = _interval_index(t, intervals)
        keep = seg >= 0
        t, seg = t[keep], seg[keep]
    else:
        seg = np.zeros(t.size, int)
    out['n_spikes_used'] = t.size
    if t.size < min_spikes:
        return out

    isi = np.diff(t)
    same = seg[1:] == seg[:-1]                # ISIs within one interval
    isi_ok = isi[same]
    if isi_ok.size:
        out['inv_median_isi_hz'] = 1 / np.median(isi_ok)
        out['burst_index'] = np.mean(isi_ok < burst_isi_s)
    pair = same[1:] & same[:-1]               # consecutive ISIs in one interval
    i1, i2 = isi[:-1][pair], isi[1:][pair]
    if i1.size:
        out['cv2'] = np.mean(2 * np.abs(i2 - i1) / (i2 + i1))

    fit = fit_acg(narrow_acg(t, seg))
    out.update(acg_tau_rise_ms=fit['tau_rise'], acg_tau_decay_ms=fit['tau_decay'],
               acg_tau_burst_ms=fit['tau_burst'], acg_refrac_ms=fit['refrac'],
               acg_fit_r2=fit['r2'])
    return out


def spike_train_features_for_units(spike_t, spike_id, unit_ids, intervals_list=None, **kw):
    '''
    spike_train_features for each cluster ID in unit_ids (e.g. goodIDs), from
    the full-length spike_times (s) and spike_clusters. intervals_list: one
    list of intervals per unit (stable_intervals), or None for whole trains.
    Returns (n_units,) arrays keyed by TRAIN_KEYS.
    '''
    spike_t = np.asarray(spike_t, dtype=float).ravel()
    spike_id = np.asarray(spike_id).ravel()
    out = {k: np.full(len(unit_ids), np.nan) for k in TRAIN_KEYS}
    for u, uid in enumerate(np.asarray(unit_ids).astype(int).ravel()):
        iv = None if intervals_list is None else intervals_list[u]
        if iv is not None and len(iv) == 0:      # never well recorded
            continue
        r = spike_train_features(spike_t[spike_id == uid], iv, **kw)
        for k in TRAIN_KEYS:
            out[k][u] = r[k]
    return out


def _robust_z(X):
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    s = (q75 - q25) / 1.349
    s = np.where(s > 0, s, X.std(axis=0))
    s = np.where(s > 0, s, 1.0)
    return (X - med) / s


def cluster_negative_units(feats, rate_key='stable_rate', spread_key='decay_um',
                           width_key='width_ms', n_clusters=2, method='gmm',
                           max_k=4, min_frac_stable=0.0, seed=0, features=None,
                           polarities=(-1,), exclude=None, outlier_pct=None,
                           n_model_samples=100000):
    '''
    Cluster units in a chosen feature space; by default trough-dominant units
    in (log10 rate, width, log10 spread).

    Only units whose polarity is in `polarities` and that have finite values
    for every feature are clustered (and, optionally, frac_stable >=
    min_frac_stable; units flagged 'peak_shank_mismatch' by build_data_dict,
    or True in `exclude`, are left out). A log-scaled feature that is <= 0
    for a unit is non-finite after the log, so that unit is left out; counts
    are returned in 'n_nonpos'. Features are robust-z-scored (median / IQR).
    Clusters are numbered by the median of width_key if it is one of the
    features (0 = narrowest), else by the first feature.

    polarities : (-1,) clusters trough-dominant units only. (-1, 0, 1) fits
        all units together; then use features defined the same way for every
        polarity (log10 pk_trough_ratio, asymmetry_raw, trough_peak_lag_ms)
        rather than width / asymmetry, which are mirrored for peak-dominant
        units, so the same waveform shape would land in different places on
        either side of the polarity boundary.

    For method='gmm', BIC is reported for k = 1..max_k. If k = 1 wins, the
    data do not support splitting in this space, whatever k-means says.

    Outliers (gmm only): 'logpdf' is each clustered unit's log density under
    the fitted mixture, in the z-scored space. With outlier_pct set, the
    threshold is the outlier_pct percentile of log density over samples drawn
    from the fitted model itself, i.e. a unit is an outlier if it is less
    likely than (100 - outlier_pct)% of the units the model would produce.
    'outlier' marks them (clustered units only).

    Params
    ------
    features : list of (key, log10) pairs, optional
        Any per-unit keys of feats; overrides rate_key / spread_key / width_key.
    exclude : (n,) bool or None   units to leave out of the fit (e.g. spread cutoff)

    Returns
    -------
    dict with
      labels : (n,) int, cluster for clustered units, -1 otherwise
      clustered : (n,) bool
      X : (n, n_features) the features, log10 applied where requested
      prob : (n, n_clusters) posterior (gmm only; nan elsewhere)
      logpdf : (n,) log density under the mixture (gmm only; nan elsewhere)
      logpdf_threshold : float or nan;  outlier : (n,) bool
      bic : {k: BIC} (gmm only)
      names : feature names; keys, log : the keys and log flags
      n_nonpos : {key: n units in `polarities` with a value <= 0} (log features)
    '''
    if features is None:
        features = [(rate_key, True), (width_key, False), (spread_key, True)]
    keys = [k for k, _ in features]
    logs = [bool(lg) for _, lg in features]
    n = len(feats['polarity'])
    in_pol = np.isin(np.asarray(feats['polarity']), polarities)

    cols, n_nonpos = [], {}
    for key, lg in features:
        v = np.asarray(feats[key], dtype=float)
        if lg:
            n_nonpos[key] = int(np.sum(in_pol & np.isfinite(v) & (v <= 0)))
            with np.errstate(divide='ignore', invalid='ignore'):
                v = np.log10(v)
        cols.append(v)
    X = np.column_stack(cols)

    ok = in_pol & np.all(np.isfinite(X), axis=1)
    if 'peak_shank_mismatch' in feats:      # features and keep mask on different shanks
        ok &= ~np.asarray(feats['peak_shank_mismatch']).astype(bool)
    if min_frac_stable > 0:
        ok &= np.asarray(feats['frac_stable']) >= min_frac_stable
    if exclude is not None:
        ok &= ~np.asarray(exclude).astype(bool)
    labels = np.full(n, -1)
    prob = np.full((n, n_clusters), np.nan)
    logpdf = np.full(n, np.nan)
    outlier = np.zeros(n, bool)
    lp_thresh = np.nan
    bic = {}
    if ok.sum() > n_clusters * 5:
        Z = _robust_z(X[ok])
        if method == 'gmm':
            for k in range(1, max_k + 1):
                g = GaussianMixture(k, covariance_type='full', n_init=10,
                                    random_state=seed).fit(Z)
                bic[k] = g.bic(Z)
            model = GaussianMixture(n_clusters, covariance_type='full', n_init=20,
                                    random_state=seed).fit(Z)
            lab = model.predict(Z)
            p = model.predict_proba(Z)
            logpdf[ok] = model.score_samples(Z)
            if outlier_pct is not None:
                draws, _ = model.sample(n_model_samples)
                lp_thresh = np.percentile(model.score_samples(draws), outlier_pct)
                outlier = ok & (logpdf < lp_thresh)
        else:
            model = KMeans(n_clusters, n_init=100, random_state=seed).fit(Z)
            lab = model.labels_
            p = None
        sort_col = keys.index(width_key) if width_key in keys else 0
        order = np.argsort([np.median(X[ok][lab == k, sort_col]) for k in range(n_clusters)])
        remap = np.empty(n_clusters, int)
        remap[order] = np.arange(n_clusters)
        labels[ok] = remap[lab]
        if p is not None:
            prob[ok] = p[:, order]
    names = [f'log10 {k}' if lg else k for k, lg in features]
    return dict(labels=labels, clustered=ok, X=X, prob=prob, logpdf=logpdf,
                logpdf_threshold=lp_thresh, outlier=outlier, bic=bic,
                names=names, keys=keys, log=logs, n_nonpos=n_nonpos)


# clusters any polarity set, not only trough-dominant units
cluster_units = cluster_negative_units


LOBE_EPS = 0.05   # lobe floor, as a fraction of the main extremum


def _lobe_log_ratio(f):
    '''
    log10 of (lobe after / lobe before) the main extremum, on the
    polarity-aligned waveform, with a floor of LOBE_EPS so a missing lobe
    gives a finite value. Carries the same information as 'asymmetry'
    (asymmetry = tanh(ln(ratio) / 2) without the floor) but doesn't pile up
    at +-1 when one lobe is absent, which a GMM would otherwise model with a
    narrow component of its own.
    '''
    pre = np.asarray(f['pre_ratio'], dtype=float)
    post = np.asarray(f['post_ratio'], dtype=float)
    return np.log10((post + LOBE_EPS) / (pre + LOBE_EPS))


# features computed from saved ones, so they need no recomputation:
# name -> (function of the features dict, keys it needs)
DERIVED_FEATURES = {
    'lobe_log_ratio': (_lobe_log_ratio, ('pre_ratio', 'post_ratio')),
}


def add_derived_features(feats):
    '''Add any DERIVED_FEATURES not already in feats whose inputs are present.'''
    for name, (fn, needs) in DERIVED_FEATURES.items():
        if name not in feats and all(k in feats for k in needs):
            feats[name] = fn(feats)
    return feats



''' ------------------------------------------------------------------------
Cell-type clustering across a whole data dict

Defaults for cluster_waveform_features. Wrappers (build_data_dict) can pass
their own values instead of editing these.
------------------------------------------------------------------------ '''
# A GMM is fit to the cells of every session at once; each feature is
# (key, log10 transform).
# The default set follows Chettih's classifyUnitTypes.m (rate, inverse
# median ISI, CV2, width, derivative ratio, asymmetry, ACG rise time), with
# the v2 width and the drift-corrected rate.
WF_CLUSTER_SOURCE = 'wf_features_v2'     # v2 waveforms: high-pass only, re-aligned
WF_CLUSTER_FEATURES = [
    ('stable_rate', True),         # firing rate while well recorded
    ('inv_median_isi_hz', True),   # 1 / median ISI
    ('cv2', False),                # local irregularity of ISIs
    ('width_ms', False),           # trough-to-peak (repolarisation)
    ('deriv_ratio', False),        # steepest rise / steepest fall (log10)
    ('asymmetry', False),          # lobe after vs. before the trough
    ('acg_tau_rise_ms', True),     # ACG rise time constant (CellExplorer fit)
]
# Which polarities to fit. (-1,): trough-dominant only, other polarities get
# their own codes. (-1, 0, 1): everything in one model -- then swap the
# mirrored shape features for ('pk_trough_ratio', True), ('asymmetry_raw', False)
# and ('trough_peak_lag_ms', False), which are defined the same way for every
# polarity (see cluster_negative_units).
WF_CLUSTER_POLARITIES = (-1,)
WF_N_CLUSTERS = None       # None: use the k with the lowest BIC (1..WF_MAX_K)
WF_MAX_K = 8
WF_MIN_POSTERIOR = 0.9     # cells assigned less confidently than this stay unclassified
WF_OUTLIER_PCT = 1.0       # cells less likely than 99% of the model's own draws are outliers
                           # (Chettih's logpdf < 0 works out to ~0.1% for their model)
WF_MIN_FRAC_STABLE = 0.05  # cells well recorded for less of the session are not clustered
# (feature, maximum): cells spreading further are treated as noise and left
# out of the fit, as Chettih does with neuronSpread < 60 um. Our spread is
# measured differently (noise floor, one shank), so pick the value from the
# distribution printed by cluster_waveform_features; None = no cutoff.
WF_SPREAD_CUTOFF = ('weighted_dist_um', None)
WF_CLUSTER_SEED = 0

# values of 'wf_cluster' other than cluster numbers (0, 1, ... in order of
# increasing median width, or of the first feature if width is not used)
WF_CLUSTER_CODES = {
    -1: 'in the fitted polarity set but unclassified: a feature missing or not '
        'log-scalable, unstable, peak channel on another shank, or posterior < min_posterior',
    -2: 'peak-dominant (positive polarity), not in the fitted set',
    -3: 'ambiguous polarity, not in the fitted set',
    -4: 'no waveform',
    -5: 'outlier: log density below the outlier_pct threshold',
    -6: 'spread above spread_cutoff (likely noise), not fit',
}


def cluster_waveform_features(data_dict, source=WF_CLUSTER_SOURCE, features=WF_CLUSTER_FEATURES,
                              n_clusters=WF_N_CLUSTERS, max_k=WF_MAX_K,
                              min_posterior=WF_MIN_POSTERIOR, min_frac_stable=WF_MIN_FRAC_STABLE,
                              polarities=WF_CLUSTER_POLARITIES, outlier_pct=WF_OUTLIER_PCT,
                              spread_cutoff=WF_SPREAD_CUTOFF, seed=WF_CLUSTER_SEED,
                              overwrite=False):
    '''
    GMM clustering of waveform / firing features across every cell in the dict.

    Cells from all birds and sessions are pooled and clustered together, so a
    cluster number means the same thing in every session. Cells whose polarity
    is in `polarities` are fit (cluster_negative_units); the
    other polarity groups get their own codes. Cells spreading beyond
    spread_cutoff are treated as noise and left out of the fit. Cluster
    numbers are ordered by median width (0 = narrowest). As in Chettih's
    classifyUnitTypes.m, assignment is conservative: a cell gets a cluster
    only if its posterior is at least min_posterior and it is not an outlier
    (log density under the mixture below the outlier_pct percentile of the
    model's own draws).

    Writes
    ------
    data_dict[bird][session]['wf_cluster'] : int array, one per cell, in the
        same order as the session's feature dict, cluster_ids and the rows of
        aligned_spikes.npy (checked against 'cluster_ids'). Values: cluster
        number, or a code in WF_CLUSTER_CODES.
    data_dict[bird][session]['wf_cluster_prob'] : posterior of the most likely
        cluster (nan for cells not fit)
    data_dict[bird][session]['wf_cluster_logpdf'] : log density under the
        mixture, in the z-scored feature space (nan for cells not fit)
    data_dict[bird]['all_wf_cluster'] : the bird's sessions concatenated, in
        'all_sessions' order (sessions without features are skipped)
    data_dict[bird]['wf_cluster_info'] : features, k, BIC, codes and per-cluster
        medians (identical for every bird)

    Old 'excitatory_idx' / 'inhibitory_idx' labels are removed so they can't be
    used by mistake.

    Params
    ------
    data_dict : the build_data_dict dict, with data_dict[bird][session][source]
    source : feature dict to cluster, e.g. 'wf_features_v2'
    features : list of (key, log10) pairs
    n_clusters : int, or None for the k with the lowest BIC (1..max_k)
    min_posterior, outlier_pct : conservative assignment (see above)
    min_frac_stable : cells well recorded for less of the session are not fit
    polarities : polarity values to fit, e.g. (-1,) or (-1, 0, 1)
    spread_cutoff : (feature, maximum) or (feature, None); cells above it get -6
    overwrite : recompute even if nothing has changed
    Defaults are the WF_* constants above.

    Skipping: the settings and a hash of every input value the clustering
    uses (the clustering features, polarity, stability, spread, cluster IDs,
    and which sessions are included) are stored in wf_cluster_info. If both
    match the stored ones, the existing labels are kept and nothing is
    refit, so this is cheap to call every time; any change to the settings,
    a new or recomputed session, or overwrite=True triggers a refit.
    '''
    import hashlib
    import json
    features = [tuple(f) for f in features]

    birds = [b for b in data_dict
             if isinstance(data_dict[b], dict) and 'all_sessions' in data_dict[b]]

    # pool every session that has the feature set, in a fixed order
    pooled, session_keys, session_sizes = {}, [], []
    for bird in birds:
        for session_id in data_dict[bird]['all_sessions']:
            session_feats = data_dict[bird][session_id].get(source)
            if session_feats is None:
                continue
            n_cells = len(session_feats['polarity'])
            for key, values in session_feats.items():
                if isinstance(values, np.ndarray) and values.shape[:1] == (n_cells,):
                    pooled.setdefault(key, []).append(values)
            session_keys.append((bird, session_id))
            session_sizes.append(n_cells)

    if not session_keys:
        print(f"  no sessions have '{source}': nothing to cluster")
        return data_dict
    # only keys present in every session, so rows stay aligned
    feats = {k: np.concatenate(v) for k, v in pooled.items() if len(v) == len(session_keys)}
    feats = add_derived_features(feats)
    missing = [k for k, _ in features if k not in feats]
    if missing:
        raise KeyError(f"cluster features not in every session's '{source}': {missing}")
    n_total = len(feats['polarity'])

    # fingerprint: the settings plus every input value the result depends on
    settings = dict(source=source, features=[[k, bool(lg)] for k, lg in features],
                    n_clusters=n_clusters, max_k=max_k, min_posterior=min_posterior,
                    outlier_pct=outlier_pct, min_frac_stable=min_frac_stable,
                    polarities=[float(p) for p in polarities],
                    spread_cutoff=list(spread_cutoff) if spread_cutoff else None, seed=seed)
    h = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode())
    h.update(repr(session_keys).encode())
    used = [k for k, _ in features] + ['polarity', 'frac_stable', 'peak_shank_mismatch',
                                       'cluster_id'] + ([spread_cutoff[0]] if spread_cutoff else [])
    for key in sorted(set(used)):
        if key in feats:
            h.update(key.encode())
            h.update(np.ascontiguousarray(feats[key], dtype=float).tobytes())
    for bird, session_id in session_keys:          # used by the alignment check
        ids = data_dict[bird][session_id].get('cluster_ids')
        if ids is not None:
            h.update(np.ascontiguousarray(ids, dtype=float).tobytes())
    fingerprint = h.hexdigest()

    stored = [data_dict[b].get('wf_cluster_info', {}).get('fingerprint') for b in birds]
    if not overwrite and stored and all(fp == fingerprint for fp in stored):
        print('  cell-type clusters are up to date (same settings and features): '
              'kept. Pass overwrite=True to refit anyway.')
        return data_dict

    # old labels: drop the k-means ones, and clusters for sessions no longer included
    for bird in birds:
        for session_id in data_dict[bird]['all_sessions']:
            session_data = data_dict[bird][session_id]
            for old_key in ('excitatory_idx', 'inhibitory_idx'):
                session_data.pop(old_key, None)
            if session_data.get(source) is None:
                for key in ('wf_cluster', 'wf_cluster_prob', 'wf_cluster_logpdf'):
                    session_data.pop(key, None)
    pol = feats['polarity']

    # spread cutoff: cells spreading further are treated as noise, not fit
    too_spread = np.zeros(n_total, bool)
    spread_key, spread_max = spread_cutoff if spread_cutoff else (None, None)
    if spread_key is not None and spread_key in feats:
        v = np.asarray(feats[spread_key], dtype=float)
        has = np.isfinite(pol) & np.isfinite(v)
        q = np.percentile(v[has], [50, 90, 95, 99]) if has.any() else [np.nan] * 4
        print(f"\n  {spread_key} across cells with a waveform: median {q[0]:.0f}, "
              f"90th {q[1]:.0f}, 95th {q[2]:.0f}, 99th {q[3]:.0f}")
        if spread_max is not None:
            too_spread = has & (v > spread_max)

    fit_kw = dict(features=features, max_k=max_k, min_frac_stable=min_frac_stable,
                  seed=seed, polarities=polarities, exclude=too_spread)
    # number of clusters: fixed, or the lowest BIC
    if n_clusters is None:
        res = cluster_negative_units(feats, n_clusters=1, **fit_kw)
        if not res['bic']:
            print('  too few cells to choose k by BIC: nothing clustered')
            return data_dict
        n_clusters = min(res['bic'], key=res['bic'].get)
    res = cluster_negative_units(
        feats, n_clusters=n_clusters, outlier_pct=outlier_pct, **fit_kw)

    # labels: polarity groups first, then confident, non-outlier assignments,
    # then the spread cutoff last so it overrides everything
    in_pol = np.isin(pol, polarities)
    labels = np.full(n_total, -4)
    labels[(pol == 1) & ~in_pol] = -2
    labels[(pol == 0) & ~in_pol] = -3
    labels[in_pol] = -1
    prob = np.full(n_total, np.nan)
    fit = res['clustered']
    if fit.any():
        prob[fit] = np.max(res['prob'][fit], axis=1)
        confident = fit & (prob >= min_posterior) & ~res['outlier']
        labels[confident] = res['labels'][confident]
        labels[res['outlier']] = -5
    labels[too_spread] = -6

    # summary
    print(f"\n  waveform clustering: {n_total} cells from {len(session_keys)} sessions, "
          f"features {[k for k, _ in features]}")
    for key, n_bad in res['n_nonpos'].items():
        if n_bad:
            print(f'    {n_bad} cells in the fitted set have {key} <= 0 (not log-scalable): unclassified')
    if res['bic']:
        best = min(res['bic'], key=res['bic'].get)
        print('    GMM BIC by k: ' + ', '.join(f'{k}: {v:.0f}' for k, v in res['bic'].items())
              + f'  (lowest at k={best}; using k={n_clusters})')
    cluster_summary = {}
    for k in range(n_clusters):
        m = labels == k
        med = {}
        for d, (key, lg) in enumerate(features):
            v = np.median(res['X'][m, d]) if m.any() else np.nan
            med[key] = 10 ** v if lg else v
        cluster_summary[k] = dict(n=int(m.sum()), median=med)
        # a GMM with enough components gives a clump of extreme cells its own
        # small cluster, where they no longer look like outliers; real classes
        # can be small too, so flag rather than drop
        small = m.sum() < 0.02 * max(fit.sum(), 1)
        print(f'    cluster {k}: n={m.sum()}, median '
              + ', '.join(f'{key} {v:.3g}' for key, v in med.items())
              + ('   <- under 2% of fitted cells: check whether these are outliers' if small else ''))
    n_low = int(np.sum(fit & (prob < min_posterior) & ~res['outlier']))
    print(f'    fit but posterior < {min_posterior}: {n_low}; '
          + ', '.join(f'code {c}: {int(np.sum(labels == c))}' for c in sorted(WF_CLUSTER_CODES)))

    # why the -1 cells are unclassified (a cell can have several reasons)
    unc = labels == -1
    if unc.any():
        reasons = {f'posterior < {min_posterior}': unc & fit & (prob < min_posterior)}
        not_fit = unc & ~fit
        for d, (key, lg) in enumerate(features):
            reasons[f'no {key}' + (' (<= 0, log)' if lg else '')] = not_fit & ~np.isfinite(res['X'][:, d])
        if min_frac_stable > 0 and 'frac_stable' in feats:
            fs_ = np.asarray(feats['frac_stable'], dtype=float)
            reasons[f'frac_stable < {min_frac_stable}'] = not_fit & ~(fs_ >= min_frac_stable)
        if 'peak_shank_mismatch' in feats:
            reasons['peak channel on another shank'] = not_fit & np.asarray(
                feats['peak_shank_mismatch']).astype(bool)
        print(f'    unclassified (-1), n={int(unc.sum())}: '
              + ', '.join(f'{name} {int(m.sum())}' for name, m in reasons.items() if m.any()))
        if fit.any():
            q = np.percentile(prob[fit], [10, 25, 50])
            print(f'    posterior of fitted cells: 10th pct {q[0]:.2f}, 25th {q[1]:.2f}, median {q[2]:.2f}')

    # split back into sessions, checking alignment against cluster_ids
    bounds = np.cumsum([0] + session_sizes)
    per_bird = {}
    for i, (bird, session_id) in enumerate(session_keys):
        rows = slice(bounds[i], bounds[i + 1])
        session_data = data_dict[bird][session_id]
        expected_ids = session_data.get('cluster_ids')
        if expected_ids is not None and 'cluster_id' in feats:
            ids = feats['cluster_id'][rows].astype(int)
            if not np.array_equal(ids, np.asarray(expected_ids).astype(int).ravel()):
                print(f"    WARNING {bird}_{session_id}: feature rows don't match 'cluster_ids'; "
                      f"wf_cluster not saved (re-run collect_waveform_data with overwrite=True)")
                for key in ('wf_cluster', 'wf_cluster_prob', 'wf_cluster_logpdf'):
                    session_data.pop(key, None)
                continue
        session_data['wf_cluster'] = labels[rows].copy()
        session_data['wf_cluster_prob'] = prob[rows].copy()
        session_data['wf_cluster_logpdf'] = res['logpdf'][rows].copy()
        per_bird.setdefault(bird, []).append(labels[rows])

    info = dict(settings=settings, fingerprint=fingerprint,
                source=source, features=features, n_clusters=int(n_clusters),
                polarities=tuple(polarities), bic=dict(res['bic']),
                min_posterior=min_posterior, outlier_pct=outlier_pct,
                logpdf_threshold=res['logpdf_threshold'], spread_cutoff=spread_cutoff,
                min_frac_stable=min_frac_stable, seed=seed, codes=dict(WF_CLUSTER_CODES),
                n_cells=n_total, cluster_summary=cluster_summary)
    for bird in birds:
        data_dict[bird]['all_wf_cluster'] = (np.concatenate(per_bird[bird]) if bird in per_bird
                                             else np.asarray([], dtype=int))
        data_dict[bird]['wf_cluster_info'] = info
    return data_dict


''' stim-related analyses: may move or delete '''
def calc_avg_stim(filt_data, sampling_rate=30000, t_pre=0.02,
                    start_t=2e-3, end_t=16e-3):
    '''
    Get the average across all stim events for a given time winow.

    start_t, end_t : float, seconds; defines time window post stim
    '''
    # set the time window to look at
    start_t_adj = start_t + t_pre
    end_t_adj = end_t + t_pre
    start_idx = np.round(start_t_adj*sampling_rate).astype(int)
    end_idx = np.round(end_t_adj*sampling_rate).astype(int)
    total_samples = end_idx - start_idx
    t_window = np.linspace(start_t, end_t, total_samples)

    # get the avg stim event
    avg_stim = np.median(filt_data[:, start_idx:end_idx], axis=-1)
    return avg_stim, t_window


def find_spikes(avg_stim, sampling_rate=30000, 
                prominance_thresh=3, dist_thresh=2e-3):
    # data params
    n_channels = avg_stim.shape[0]
    dist_samples = np.round(dist_thresh*sampling_rate).astype(int)
    
    # get all the peaks for each channel
    spikes_idx = []
    sp_props_all = []
    total_spikes = 0
    channel_max = np.zeros(n_channels)
    ch_max_sp_idx = np.zeros(n_channels)
    for i, stim_ch in enumerate(avg_stim):
        spikes, sp_props = find_peaks(stim_ch,
                                        height=0.5,
                                        prominence=prominance_thresh,
                                        distance=dist_samples)
        total_spikes += spikes.shape[0]
        if spikes.shape[0] > 0:
            channel_max[i] = np.max(sp_props['peak_heights'])
            ch_max_sp_idx[i] = np.argmax(sp_props['peak_heights'])
        spikes_idx.append(spikes.astype(int))
        sp_props_all.append(sp_props)
    ch_max_sp_idx = ch_max_sp_idx.astype(int)
    
    return spikes_idx, sp_props_all, total_spikes, channel_max, ch_max_sp_idx


def find_isolated_spikes(avg_stim, n_templates=50,
                            sampling_rate=30000, t_pre=0.02,
                            prominance_thresh=3, dist_thresh=2e-3,
                            wf_pre=0.5e-3, wf_post=0.5e-3, n_wf_channels=7):
    '''
    Extract waveform templates from a given time window post-stim.
    Only keep templates that are not part of another template, starting with the biggest spike.

    prominance_thresh, dist_thresh : floats for scipy.signal.find_peaks
    wf_pre, wf_post : float, seconds; defines time window for waveform template centered on peak
    n_wf_channels : int; defines number of channels for waveform template, centered on peak
    '''
    n_channels, n_timepts = avg_stim.shape

    # to store variables
    templates = np.asarray([])
    all_sp_ch_idx = np.asarray([])
    all_sp_t = np.asarray([])

    # convert params to samples
    wf_pre_samples = np.round(wf_pre*sampling_rate).astype(int)
    wf_post_samples = np.round(wf_post*sampling_rate).astype(int)

    # for plotting
    wf_window_ms = np.linspace(-wf_pre*1000, wf_post*1000, wf_pre_samples+wf_post_samples)

    # get the initial spike events
    (spikes_idx, sp_props_all, total_spikes,
        channel_max, ch_max_sp_idx) = find_spikes(avg_stim,
                                                    prominance_thresh=prominance_thresh,
                                                    dist_thresh=2e-3)

    # find all spikes that aren't part of other templates
    avg_stim_residual = avg_stim.copy()
    n_timepts = avg_stim.shape[1]
    while (templates.shape[0] < n_templates) & (total_spikes > 0):
        # reset variables
        temp_template = np.zeros_like(avg_stim)

        # find the largest remaining spike
        sp_ch_idx = np.argmax(channel_max)
        max_sp_props = sp_props_all[sp_ch_idx]
        sp_t_idx = spikes_idx[sp_ch_idx][ch_max_sp_idx[sp_ch_idx]]

        # get the template window
        sp_start_ch = sp_ch_idx - n_wf_channels//2
        sp_end_ch = sp_ch_idx + n_wf_channels//2 + 1
        sp_start_t = sp_t_idx - wf_pre_samples
        sp_end_t = sp_t_idx + wf_post_samples
        
        # check edges
        if sp_start_ch < 0:
            sp_start_ch = 0
            sp_end_ch = n_wf_channels
        elif sp_end_ch > n_channels:
            sp_start_ch = n_channels - n_wf_channels - 1
            sp_end_ch = -1
        if sp_start_t < 0:
            sp_start_t = 0
            sp_end_t = wf_pre_samples + wf_post_samples
        elif sp_end_t > n_timepts:
            sp_start_t = n_timepts - (wf_pre_samples + wf_post_samples)
            sp_end_t = -1
        
        # extract the template waveform and save the indices
        new_template = avg_stim_residual[None, sp_start_ch:sp_end_ch, sp_start_t:sp_end_t]
        if templates.shape[0] == 0:
            templates = new_template
        else:
            templates = np.concatenate((templates, new_template), axis=0)
        all_sp_ch_idx = np.append(all_sp_ch_idx, sp_ch_idx)
        all_sp_t = np.append(all_sp_t, sp_t_idx)
            
        # find the residual activity
        temp_template[sp_start_ch:sp_end_ch, sp_start_t:sp_end_t] = new_template.squeeze().copy()
        avg_stim_residual = avg_stim_residual - temp_template
        
        # get remaining peaks
        spikes_idx, sp_props_all, total_spikes, channel_max, ch_max_sp_idx = find_spikes(avg_stim_residual)
        print(f"found {templates.shape[0]} templates, analyzing {total_spikes} remaining spikes")
    all_sp_ch_idx = all_sp_ch_idx.astype(int)
    all_sp_t = all_sp_t.astype(int)

    return templates, all_sp_ch_idx, all_sp_t, wf_window_ms


def trial_trial_correlations(filt_data, templates, 
                                wf_ch_idx, wf_times,
                                sampling_rate=30000, t_pre=0.02,
                                t_buffer=2e-3):
    ''' 
    Sweep the templates over the filtered data on each trial to find the best correlation. 

    Each waveform template has a channel associated with the biggest peak and spans n_wf_channels.
    Here, we search for occurances of that template across just those channels and within
    a limited time window before and after the peak time defined by t_buffer.

    Correlations are calculated separately for each channel, then summed across channels to get a
    composite correlation for all template channels over time (so stronger correlations will win out).

    Params
    ------
    filt_data : filtered ephys data; shape (n_channels, n_timepts, n_stim)
    templates : waveform templates for putative antidromic responses; 
                shape (n_wf_cells, n_wf_channels, n_wf_pts)
    wf_ch_idx : channel index for best spike in each template
    wf_times : (seconds) time post stim that the best spike in each template peaked
    sampling_rate : (Hz)
    t_pre : (seconds) param defining how much data was taken before the stim
    t_buffer : (seconds) time window before and after the template peak to search for spikes

    Returns
    -------
    all_correlations : correlation of each template with the data;
                        shape (n_wf_cells, n_samples, n_stim);
                        where n_samples is 2 * t_buffer * sampling rate
                        note that edges will be zeros to avoid edge effects
                        (see documentation for mode = 'valid' in numpy.convolve)
    t_windows : (seconds) timepoints over which the correlation was performed
                (relative to stim time)
    '''
    # data params
    n_samples = int(t_buffer*2*sampling_rate)
    n_wf_cells, n_wf_channels, n_wf_pts = templates.shape
    n_channels, n_timepts, n_stim = filt_data.shape

    # to store the correlation values and indices
    all_correlations = np.zeros((n_wf_cells, n_samples, n_stim))
    t_windows = np.zeros((n_wf_cells, n_samples))
    for i, temp in enumerate(templates):
        # get the data for this template
        sp_ch = wf_ch_idx[i]
        sp_t = wf_times[i]
        
        # set the indices
        start_ch = sp_ch - n_wf_channels//2
        end_ch = sp_ch + n_wf_channels//2 + 1
        start_t = sp_t - t_buffer
        end_t = sp_t + t_buffer
        
        # check edges
        if start_ch < 0:
            start_ch = 0
            end_ch = n_wf_channels + 1
        elif end_ch > n_channels:
            start_ch = n_channels - (n_wf_channels + 1)
            end_ch = -1
        if start_t < 0:
            start_t = 0
            end_t = wf_pre_samples + wf_post_samples + 1
        elif end_t > n_timepts:
            start_t = n_timepts - (wf_pre_samples + wf_post_samples + 1)
            end_t = -1
        
        # set the time window to look at
        start_t_adj = start_t + t_pre
        end_t_adj = end_t + t_pre
        start_idx = np.round(start_t_adj*sampling_rate).astype(int)
        end_idx = np.round(end_t_adj*sampling_rate).astype(int)
        t_windows[i] = np.linspace(start_t, end_t, end_idx - start_idx)
        
        # examine a chunk of data around the peak for best correlation   
        corr = np.zeros_like(filt_data[start_ch:end_ch, start_idx:end_idx])
        start_corr_idx = n_wf_pts//2
        end_corr_idx = (-n_wf_pts//2) + 1
        for s in range(n_stim):
            data_chunk = filt_data[start_ch:end_ch, start_idx:end_idx, s]
            for ch in range(data_chunk.shape[0]):
                corr[ch, start_corr_idx:end_corr_idx, s] = np.correlate(data_chunk[ch], 
                                                                            temp[ch], 
                                                                            mode='valid')
        corr_composite = np.sum(corr, axis=0) # combine correlations across channels
        all_correlations[i] = corr_composite

    return all_correlations, t_windows
