'''
KS spike amplitudes, for the skinny panel drawn beside each raster.

The panel shows mean spike amplitude per trial, normalised to the unit's
session median, so a unit drifting off the probe is visible as a fall in the
trace rather than something you have to infer from a thinning raster.

Nothing here filters anything: the amplitude window overlaps the response
being plotted, which is fine ONLY because no trial is discarded on the
strength of it.  See the note in trial_amplitudes before reusing these
numbers to select trials.

Frame conventions match event_psth.py: frames are video frames, indexing the
columns of aligned_spikes.npy.
'''

import numpy as np


def load_spike_amplitudes(session_dir, data_dir, ks_dir, cluster_ids, n_frames,
                          sampling_rate=30000, fps=50):
    '''
    KS per-spike amplitudes, expressed in video-frame time, one entry per cell.

    amplitudes.npy is indexed per spike, in the same order as spike_times.npy
    and spike_clusters.npy, so the three are masked together here.  The
    sample -> frame conversion mirrors neural_analysis.align_spikes_behavior
    (shift by the first frame time, bin on frame_samples), so the frames these
    return line up with the columns of aligned_spikes.npy.

    Params
    ------
    session_dir : str    f"{root_dir}{bird}/{bird}_{session_id}/"
    data_dir : str       f"{session_dir}behavior_data/"
    ks_dir : str         f"{bird}_{ephys_id}/{ks_folder}/"
    cluster_ids : int array, shape (n_cells,)
        phy cluster ID per row of aligned_spikes.npy, i.e. what
        cluster_ids_for_session returns, AFTER the cell filter is applied
    n_frames : int

    Returns
    -------
    per_cell : list of (frames, amps), one tuple per entry in cluster_ids.
        Empty arrays for clusters with no in-session spikes.
    '''
    # load the neural data
    spike_samp = np.squeeze(np.load(f"{session_dir}{ks_dir}spike_times.npy"))
    spike_id = np.squeeze(np.load(f"{session_dir}{ks_dir}spike_clusters.npy"))
    amps = np.squeeze(np.load(f"{session_dir}{ks_dir}amplitudes.npy"))

    # load the video frame times
    frame_t = np.squeeze(np.load(f"{data_dir}frame_times.npy"))
    start_t = frame_t[0]
    frame_t = frame_t - start_t
    frame_samples = np.append(frame_t, frame_t[-1] + 1 / fps) * sampling_rate

    # keep only spikes from within the behavior session
    spike_t = spike_samp - start_t * sampling_rate
    in_session = (spike_t >= 0) & (spike_t <= frame_samples[-1])
    spike_t = spike_t[in_session]
    spike_id = spike_id[in_session]
    amps = amps[in_session]

    # amplitude per video frame
    per_cell = []
    for cid in np.asarray(cluster_ids).astype(int):
        sel = spike_id == cid
        fr = np.searchsorted(frame_samples, spike_t[sel], side='right') - 1
        fr = np.clip(fr, 0, n_frames - 1)
        per_cell.append((fr, amps[sel]))
    return per_cell


def trial_amplitudes(spike_frames, amps, align_frames, half_width,
                     ref=None, min_spikes=1, stat='mean'):
    '''
    Mean spike amplitude within each trial's raster window, normalised to the
    unit's session median.

    Note: by default, ref is the median over the whole session. 
    Pass ref explicitly to measure against a median taken, e.g., 
    when the cell was clearly stable on the probe.

    Params
    ------
    spike_frames, amps : arrays, shape (n_spikes,)  from load_spike_amplitudes
    align_frames : int array, shape (n_events,)
        raster alignment frames, already in raster row order
    half_width : int      frames each side, i.e. the raster half-window
    ref : float or None   session reference amplitude; median of amps if None
    min_spikes : int      trials with fewer spikes in window return nan
    stat : 'mean' | 'median'

    Returns
    -------
    amp_trial : float array, shape (n_events,)
        amplitude / ref, nan where the trial had too few spikes
    n_spikes : int array, shape (n_events,)
    '''
    spike_frames = np.asarray(spike_frames)
    amps = np.asarray(amps, dtype=float)
    align_frames = np.asarray(align_frames).astype(int)
    n_events = align_frames.shape[0]

    amp_trial = np.full(n_events, np.nan)
    n_spikes = np.zeros(n_events, dtype=int)
    if spike_frames.size == 0:
        return amp_trial, n_spikes

    if ref is None:
        ref = np.median(amps)
    if not np.isfinite(ref) or ref == 0:
        return amp_trial, n_spikes

    # spikes are already frame-indexed, so one searchsorted per window edge
    order = np.argsort(spike_frames, kind='stable')
    sf_sorted = spike_frames[order]
    amp_sorted = amps[order]

    lo = np.searchsorted(sf_sorted, align_frames - half_width, side='left')
    hi = np.searchsorted(sf_sorted, align_frames + half_width + 1, side='left')

    take = np.median if stat == 'median' else np.mean
    for i in range(n_events):
        chunk = amp_sorted[lo[i]:hi[i]]
        n_spikes[i] = chunk.size
        if chunk.size >= min_spikes:
            amp_trial[i] = take(chunk) / ref
    return amp_trial, n_spikes


def plot_amp_panel(ax, amp_trial, groups_sorted=None, block_edges=(),
                   colors=None, default_color='xkcd:gray', divider_lw=0.8,
                   lw=0.8, label='amp.', axis_label=12, xmax=None):
    '''
    Skinny amplitude trace to sit immediately right of a raster: x is spike
    amplitude relative to the unit's session median, y is raster row.

    Rows must already be in raster order. Trials with no spikes are nan and
    simply break the line.

    Params
    ------
    ax : the amplitude axes
    amp_trial : float array, shape (n_events,)   raster-ordered
    groups_sorted : array, shape (n_events,) or None
        group label per row, so each occupancy/feeder block gets its colour
    block_edges : iterable of int
        first row of each new block, matching the raster dividers
    colors : dict or None    group label -> colour
    xmax : float or None     upper x limit; derived from the data if None

    Returns
    -------
    xmax : float   so a caller can match limits across panels if it wants
    '''
    amp_trial = np.asarray(amp_trial, dtype=float)
    n_events = amp_trial.shape[0]
    rows = np.arange(n_events)

    if xmax is None:
        finite = amp_trial[np.isfinite(amp_trial)]
        top = float(np.max(finite)) if finite.size else 1.0
        xmax = max(np.ceil(top * 10) / 10, 0.5)   # round up to a clean tenth

    if groups_sorted is None:
        ax.plot(amp_trial, rows, color=default_color, lw=lw)
    else:
        groups_sorted = np.asarray(groups_sorted)
        # draw each contiguous block separately so the line does not jump
        # across a group boundary
        starts = np.concatenate(([0], np.asarray(list(block_edges), dtype=int),
                                 [n_events]))
        starts = np.unique(starts)
        for a, b in zip(starts[:-1], starts[1:]):
            if b <= a:
                continue
            g_id = groups_sorted[a]
            col = (colors[g_id] if colors is not None and g_id < len(colors)
                    else default_color)
            ax.plot(amp_trial[a:b], rows[a:b], color=col, lw=lw)

    # dividers matching the raster blocks
    for edge in block_edges:
        ax.axhline(edge - 0.5, color='xkcd:gray', lw=divider_lw, zorder=3)

    # keep it clean: bottom spine only, two ticks
    for side in ['top', 'left', 'right']:
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, xmax])
    ax.set_xlabel(label, fontsize=axis_label)
    ax.tick_params(axis='x', labelsize=max(axis_label - 3, 6))
    return xmax
