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
              'asymmetry', 'pre_ratio', 'post_ratio', 'main_offset_ms']
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

    out.update(amp=amp, pk_trough_ratio=ratio, polarity=pol, width_ms=width,
               half_width_ms=_half_width(z, i) / f * 1e3,
               asymmetry=(b - a) / (b + a) if (a + b) > 0 else np.nan,
               pre_ratio=a / amp, post_ratio=b / amp,
               main_offset_ms=(i - s0) / f * 1e3)
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
    per-chunk arrays under 'chunks' (centers in frames, rate, med_amp, stable).
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
    res['chunks'] = dict(centers=ch['centers'], rate=rate, med_amp=med, stable=stable)
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


def _robust_z(X):
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    s = (q75 - q25) / 1.349
    s = np.where(s > 0, s, X.std(axis=0))
    s = np.where(s > 0, s, 1.0)
    return (X - med) / s


def cluster_negative_units(feats, rate_key='stable_rate', spread_key='decay_um',
                           width_key='width_ms', n_clusters=2, method='gmm',
                           max_k=4, min_frac_stable=0.0, seed=0, features=None):
    '''
    Cluster trough-dominant units in a chosen feature space; by default
    (log10 rate, width, log10 spread).

    Only units with polarity == -1 and finite values for every feature are
    clustered (and, optionally, frac_stable >= min_frac_stable; units
    flagged 'peak_shank_mismatch' by build_data_dict are left out). A
    log-scaled feature that is <= 0 for a unit (e.g. weighted_dist_um or
    extent_um of 0 for a unit seen on one channel) is non-finite after the log,
    so that unit is left out; counts are returned in 'n_nonpos'. Features
    are robust-z-scored (median / IQR), so a few extreme units do not set the
    scale. Clusters are numbered by the median of width_key if it is one of
    the features (0 = narrowest), else by the first feature, rather than by
    size: in LHy there is no reason to expect the larger cluster to be any
    particular cell type.

    For method='gmm', BIC is reported for k = 1..max_k. If k = 1 wins, the
    data do not support splitting in this space, whatever k-means says.

    Params
    ------
    features : list of (key, log10) pairs, optional
        Any per-unit keys of feats, e.g.
        [('stable_rate', True), ('width_ms', False), ('asymmetry', False)].
        Overrides rate_key / spread_key / width_key when given.

    Returns
    -------
    dict with
      labels : (n,) int, cluster for clustered units, -1 otherwise
      clustered : (n,) bool
      X : (n, n_features) the features, log10 applied where requested
      prob : (n, n_clusters) posterior (gmm only; nan elsewhere)
      bic : {k: BIC} (gmm only)
      names : feature names; keys, log : the keys and log flags
      n_nonpos : {key: n trough-dominant units with a value <= 0} (log features)
    '''
    if features is None:
        features = [(rate_key, True), (width_key, False), (spread_key, True)]
    keys = [k for k, _ in features]
    logs = [bool(lg) for _, lg in features]
    n = len(feats['polarity'])
    neg = np.asarray(feats['polarity']) == -1

    cols, n_nonpos = [], {}
    for key, lg in features:
        v = np.asarray(feats[key], dtype=float)
        if lg:
            n_nonpos[key] = int(np.sum(neg & np.isfinite(v) & (v <= 0)))
            with np.errstate(divide='ignore', invalid='ignore'):
                v = np.log10(v)
        cols.append(v)
    X = np.column_stack(cols)

    ok = neg & np.all(np.isfinite(X), axis=1)
    if 'peak_shank_mismatch' in feats:      # features and keep mask on different shanks
        ok &= ~np.asarray(feats['peak_shank_mismatch']).astype(bool)
    if min_frac_stable > 0:
        ok &= np.asarray(feats['frac_stable']) >= min_frac_stable
    labels = np.full(n, -1)
    prob = np.full((n, n_clusters), np.nan)
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
    return dict(labels=labels, clustered=ok, X=X, prob=prob, bic=bic,
                names=names, keys=keys, log=logs, n_nonpos=n_nonpos)



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


def get_bird_wf_features(data_dict, bird, feature_key='wf_features_v2'):
    '''
    All v2 waveform features for one bird, concatenated across its sessions.

    Each value is a (n_cells,) array. Rows follow session order, and within a
    session the same kept cells as aligned_spikes.npy and cluster_ids.
    'session' and 'cluster_id' say which cell each row is.
    '''
    pooled = {}
    sessions = []
    for session_id in data_dict[bird]['all_sessions']:
        session_feats = data_dict[bird][session_id].get(feature_key)
        if session_feats is None:          # no ephys, or v2 not computed yet
            continue
        n_cells = len(session_feats['polarity'])
        for key, values in session_feats.items():
            # per-cell arrays only (skips stability_chunks and other non-array entries)
            if isinstance(values, np.ndarray) and values.shape[:1] == (n_cells,):
                pooled.setdefault(key, []).append(values)
        sessions.append(np.full(n_cells, session_id, dtype=object))

    bird_feats = {key: np.concatenate(arrays) for key, arrays in pooled.items()}
    bird_feats['session'] = np.concatenate(sessions) if sessions else np.array([])
    return bird_feats