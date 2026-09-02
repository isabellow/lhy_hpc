'''
Event-aligned rasters and tuning curves.

Everything here works on one cell at a time for rasters
and on all cells at once for tuning curves.

Frame conventions
-----------------
`spike_fr` is spike counts per video frame, shape (n_cells, n_frames), as
saved by save_aligned_spikes.py.  Dividing by dt turns it into Hz.
Window bounds are (start, end) offsets in frames relative to the alignment
frame; start is normally negative and the window is half-open [start, end).
'''

import numpy as np
from scipy.ndimage import gaussian_filter1d


def window_frames(t_start, t_end, dt):
    '''
    Convert a window in seconds to (start_frame, end_frame) and the matching
    time axis.  Keeps the axis and the slice length in sync, which the old
    scripts did by hand in two places each.
    '''
    fr_start = int(round(t_start / dt))
    fr_end = int(round(t_end / dt))
    t_pts = np.arange(fr_start, fr_end) * dt
    return fr_start, fr_end, t_pts


def build_raster(spike_train, align_frames, half_width):
    '''
    Spike counts around each event for a single cell.

    Params
    ------
    spike_train : array, shape (n_frames,)
        spike counts per video frame for one cell
    align_frames : array of ints, shape (n_events,)
        frame each event is aligned to
    half_width : int
        frames to take on each side of the alignment frame

    Returns
    -------
    raster : array, shape (n_events, 2*half_width + 1)
        zero-padded where the window runs off either end of the session, so
        events near the session edges are kept rather than dropped or, as in
        the old spikes_by_cache, silently truncated to a shorter row
    '''
    spike_train = np.asarray(spike_train)
    align_frames = np.asarray(align_frames).astype(int)
    n_frames = spike_train.shape[0]
    n_t = 2 * half_width + 1

    raster = np.zeros((align_frames.shape[0], n_t), dtype=spike_train.dtype)
    for i, a in enumerate(align_frames):
        start, end = a - half_width, a + half_width + 1
        start_clip, end_clip = max(start, 0), min(end, n_frames)
        if end_clip <= start_clip:
            continue
        raster[i, start_clip - start: start_clip - start + (end_clip - start_clip)] = \
            spike_train[start_clip:end_clip]
    return raster


def raster_scatter(raster, t_pts, dt, rng=None):
    '''
    Turn a count matrix into (time, row) scatter coordinates.

    Spikes are sorted at 30 kHz but binned into 50 Hz video frames, so without
    a within-frame offset every spike in a frame lands on the same x value:
    the raster shows vertical stripes and a frame with three spikes is
    indistinguishable from a frame with one.  A deterministic rng keeps the
    same cell looking the same each time it is drawn.
    '''
    rows, cols = np.nonzero(raster)
    if rows.size == 0:
        return np.zeros(0), np.zeros(0)
    reps = raster[rows, cols].astype(int)
    rows = np.repeat(rows, reps)
    cols = np.repeat(cols, reps)
    if rng is None:
        jitter = 0.0
    else:
        jitter = rng.uniform(0.0, dt, size=cols.size)
    return t_pts[cols] + jitter, rows

def subsample_for_raster(onsets, offsets, groups=None, max_per_group=100, rng=None):
    '''
    Cap how many rows a raster draws per group.
    Groups at or under max_per_group are returned untouched.

    Params
    ------
    onsets, offsets : arrays, shape (n_events,)
    groups : array, shape (n_events,) or None
        group label per event; None puts everything in one group
    max_per_group : int
        events sampled per group when a group exceeds this number
    rng : np.random.Generator or None
        pass a seeded generator for a reproducible subsample

    Returns
    -------
    sub_onsets, sub_offsets, sub_groups : arrays, subsampled
    keep_idx : int array
        indices into the ORIGINAL arrays that were kept, in case a caller
        needs to pull matching values from a fourth array
    n_total : dict, group_id -> true count before subsampling
    '''
    # set optional params
    if groups is None:
        groups = np.zeros(onsets.shape[0], dtype=int)
    groups = np.asarray(groups)
    if rng is None:
        rng = np.random.default_rng()

    # subsampling index
    keep_idx = []
    n_total = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n_total[int(g)] = int(idx.shape[0])
        if idx.shape[0] > max_per_group:
            idx = np.sort(rng.choice(idx, size=max_per_group, replace=False))
        keep_idx.append(idx)
    keep_idx = (np.concatenate(keep_idx) if keep_idx
                else np.asarray([], dtype=int))

    return onsets[keep_idx], offsets[keep_idx], groups[keep_idx], keep_idx, n_total

def sort_events_by_duration(onsets, offsets, groups=None):
    '''
    Order events for a raster: by group first (so each group is a contiguous
    block), then by duration within group.

    Returns
    -------
    order : int array, shape (n_events,)
        indices that sort the events
    block_edges : int array
        row index where each new group starts, excluding 0.  Used both to draw
        divider lines and to colour each block, so the raster and the tuning
        curves can no longer disagree about which colour a group gets - the
        bug in the old feeder raster, which coloured blocks by running order
        while the tuning curves coloured by feeder ID.
    '''
    onsets = np.asarray(onsets)
    offsets = np.asarray(offsets)
    duration = offsets - onsets
    if groups is None:
        groups = np.zeros(onsets.shape[0], dtype=int)
    groups = np.asarray(groups)

    # lexsort applies the LAST key first, so this is group-major, duration-minor
    order = np.lexsort((duration, groups))
    sorted_groups = groups[order]
    block_edges = np.where(np.diff(sorted_groups) != 0)[0] + 1
    return order, block_edges

def sort_events_by_time_group(onsets, offsets, groups=None):
    onsets = np.asarray(onsets)
    offsets = np.asarray(offsets)
    if groups is None:
        groups = np.zeros(onsets.shape[0], dtype=int)
    groups = np.asarray(groups)

    # lexsort applies the LAST key first, so this is group-major, time-minor
    order = np.lexsort((onsets, groups))
    sorted_groups = groups[order]
    block_edges = np.where(np.diff(sorted_groups) != 0)[0] + 1
    return order, block_edges


def event_psth(spike_fr, align_frames, window, dt,
               sigma_frames=0.0, keep=None):
    '''
    Mean firing rate (Hz) for every cell in one window around a set of events.

    Params
    ------
    spike_fr : array, shape (n_cells, n_frames)
        spike counts per video frame
    align_frames : array of ints, shape (n_events,)
    window : (start_frame, end_frame)
        offsets relative to the alignment frame; start is normally negative
    dt : float
        seconds per video frame
    sigma_frames : float
        Gaussian smoothing sigma in frames (0 = no smoothing)
    keep : bool array or None
        events to include.  Shape (n_events,) applies the same selection to
        every cell; shape (n_cells, n_events) gives each cell its own set of
        events, which is what a per-cell drift cut needs.  Events whose
        window leaves the session are dropped on top of either.

    Returns
    -------
    psth : array, shape (n_cells, n_t)   -- zeros for a cell with no events
    n_used : int array, shape (n_cells,)
    '''
    spike_fr = np.asarray(spike_fr)
    align_frames = np.asarray(align_frames).astype(int)
    n_cells, n_frames = spike_fr.shape
    fr_start, fr_end = window
    n_t = fr_end - fr_start

    in_bounds = (align_frames + fr_start >= 0) & (align_frames + fr_end <= n_frames)
    per_cell = None
    if keep is not None:
        keep = np.asarray(keep).astype(bool)
        if keep.ndim == 1:
            in_bounds &= keep
        else:
            in_bounds &= keep.any(axis=0)   # no cell wants it: drop it outright
            per_cell = keep
    used = align_frames[in_bounds]

    if used.shape[0] == 0:
        return np.zeros((n_cells, n_t)), np.zeros(n_cells, dtype=int)

    snippets = np.stack([spike_fr[:, a + fr_start: a + fr_end] for a in used], axis=0)

    if per_cell is None:
        psth = snippets.mean(axis=0) / dt
        n_used = np.full(n_cells, used.shape[0], dtype=int)
    else:
        # per-cell selection: nan out the events a cell is not using and take
        # a nanmean, so all cells are still averaged in one pass.  Costs a
        # float copy of the snippet block, which is the price of not looping
        per_cell = per_cell[:, in_bounds]
        snippets = snippets.astype(float)
        snippets[~per_cell.T] = np.nan
        n_used = per_cell.sum(axis=1).astype(int)
        # a cell with nothing left is left at zero: nanmean of an all-nan
        # slice is a nan plus a RuntimeWarning, and a nan blanks the panel
        psth = np.zeros((n_cells, n_t))
        drawn = n_used > 0
        if np.any(drawn):
            psth[drawn] = np.nanmean(snippets[:, drawn], axis=0) / dt

    if sigma_frames > 0:
        psth = gaussian_filter1d(psth, sigma_frames, axis=1, mode='nearest')
    return psth, n_used


def event_psth_on_off(spike_fr, onsets, offsets, on_window, off_window, dt,
                      groups=None, sigma_frames=0.0, min_duration=None,
                      keep=None):
    '''
    Onset- and offset-aligned tuning curves, optionally split into groups
    (feeder identity, occupied vs. empty, ...).

    An event is used only if BOTH its windows fit inside the session, so the
    two curves for a group are always averages over the same set of events.

    Params
    ------
    spike_fr : array, shape (n_cells, n_frames)
    onsets, offsets : int arrays, shape (n_events,)
    on_window, off_window : (start_frame, end_frame)
        relative to the event onset and offset respectively
    dt : float
    groups : array, shape (n_events,) or None
        group label per event; None puts everything in one group
    sigma_frames : float
    min_duration : int or None
        drop events shorter than this many frames.  Set it to the combined
        window length to stop the onset and offset windows from overlapping
        and double-counting the middle of a short event.
    keep : bool array, shape (n_events,) or (n_cells, n_events) or None
        events to include, per event or per cell-and-event.  The same mask is
        used for the onset and the offset curve, so a group's two curves are
        still averages over the same events for any given cell.

    Returns
    -------
    onset_psth  : array, shape (n_cells, n_groups, n_t_on)
    offset_psth : array, shape (n_cells, n_groups, n_t_off)
    group_ids   : array, shape (n_groups,)
    n_used      : int array, shape (n_cells, n_groups)
        events behind each curve.  Per cell because `keep` may be, and every
        row is identical when it is not.
    '''
    spike_fr = np.asarray(spike_fr)
    onsets = np.asarray(onsets).astype(int)
    offsets = np.asarray(offsets).astype(int)
    n_cells = spike_fr.shape[0]

    if groups is None:
        groups = np.zeros(onsets.shape[0], dtype=int)
    groups = np.asarray(groups)
    group_ids = np.unique(groups)
    n_groups = group_ids.shape[0]

    n_t_on = on_window[1] - on_window[0]
    n_t_off = off_window[1] - off_window[0]
    onset_psth = np.zeros((n_cells, n_groups, n_t_on))
    offset_psth = np.zeros((n_cells, n_groups, n_t_off))
    n_used = np.zeros((n_cells, n_groups), dtype=int)

    if keep is not None:
        keep = np.asarray(keep).astype(bool)

    # an event is usable only if both of its windows are complete
    n_frames = spike_fr.shape[1]
    usable = ((onsets + on_window[0] >= 0) &
              (onsets + on_window[1] <= n_frames) &
              (offsets + off_window[0] >= 0) &
              (offsets + off_window[1] <= n_frames))
    if min_duration is not None:
        usable &= (offsets - onsets) >= min_duration

    for g_idx, g in enumerate(group_ids):
        sel = usable & (groups == g)
        if not np.any(sel):
            continue
        g_keep = None if keep is None else keep[..., sel]
        on_psth, n_on = event_psth(spike_fr, onsets[sel], on_window, dt,
                                   sigma_frames=sigma_frames, keep=g_keep)
        off_psth, _ = event_psth(spike_fr, offsets[sel], off_window, dt,
                                 sigma_frames=sigma_frames, keep=g_keep)
        onset_psth[:, g_idx] = on_psth
        offset_psth[:, g_idx] = off_psth
        n_used[:, g_idx] = n_on

    return onset_psth, offset_psth, group_ids, n_used


def shared_ylim(*psth_blocks, pad=1.0, minimum=1.0):
    '''
    One tuning-curve ceiling across every panel that should share a y axis.

    Ignores empty selections, so a group with no usable events cannot turn the
    limit into nan and blank the panel (the old feeder script called np.max on
    an empty slice whenever no feeder cleared the minimum-visit count, and died
    with 'zero-size array to reduction operation').
    '''
    top = 0.0
    for block in psth_blocks:
        if block is None:
            continue
        block = np.asarray(block, dtype=float)
        if block.size == 0:
            continue
        finite = block[np.isfinite(block)]
        if finite.size == 0:
            continue
        top = max(top, float(np.max(finite)))
    return max(float(np.ceil(top)) + pad, minimum)


def plot_psth_trace(ax, t_pts, rate, dt, color, sigma_frames=0.0, lw=3,
                    label=None, **kwargs):
    '''
    Draw one tuning curve, as a smoothed line or as an unfilled per-bin
    histogram depending on how the PSTH was computed.

    sigma_frames > 0  -- the curve was Gaussian-smoothed, so a line is honest
                         about it and ax.plot is used as before.
    sigma_frames == 0 -- nothing was smoothed, so the trace is drawn as an
                         unfilled step histogram: bin i spans
                         [t_pts[i], t_pts[i] + dt), matching the half-open
                         frame windows used everywhere else in this module.
                         A one-frame change then shows as a step instead of
                         being interpolated into a slope by ax.plot.

    Pass the same sigma_frames given to event_psth / event_psth_on_off and
    the two can never disagree about what is being shown.

    Returns the Line2D handle, or None if t_pts is empty.
    '''
    t_pts = np.asarray(t_pts)
    rate = np.asarray(rate)
    if t_pts.shape[0] == 0:
        return None

    if sigma_frames > 0:
        line, = ax.plot(t_pts, rate, color=color, lw=lw, label=label, **kwargs)
        return line

    # close the last bin: without this the final frame is a single point
    # rather than a dt-wide step, and the trace looks like it stops short
    line, = ax.step(np.append(t_pts, t_pts[-1] + dt), np.append(rate, rate[-1]),
                    where='post', color=color, lw=lw, label=label, **kwargs)
    return line


def raster_marker_size(ax, fig, n_rows, max_area=10000.0):
    '''
    Scale the raster tick size so rows just touch, whatever the panel height
    and event count.  Replaces the magic 18510/(n**2) constant in the old
    feeder script, which only looked right for one figure size.
    '''
    ax_h_pts = ax.get_position().height * fig.get_figheight() * 72.0
    area = (ax_h_pts / max(n_rows, 1)) ** 2
    return float(min(area, max_area))
