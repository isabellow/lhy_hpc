import numpy as np
import pandas as pd
import os 

import spikeinterface as si # core only
import spikeinterface.preprocessing as spre
import dartsort
from dataclasses import replace

from scipy.ndimage import median_filter
from scipy.interpolate import PchipInterpolator

def ks4_preprocess(recording, filt=True, cmr=True, whiten=True):
    # Bandpass / highpass
    if filt:
        recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype="float32")

    # Common median reference (similar to KS behavior)
    if cmr:
        recording = spre.common_reference(recording, reference="global", operator="median")

    # Whitening
    if whiten:
        recording = spre.whiten(recording, dtype="float32")

    return recording

def _log_spaced_knot_idx(n_apply, fs, knot_one_ms=0.5, n_knots=12):
    """
    Get sample indices for log-spaced spline knots.

    Knots are concentrated early and become increasingly sparse
    Always includes index 0 (join anchor) and n_apply-1 (endpoint)

    Parameters
    ----------
    knot_one_ms : float
        time of the first knot after t=0
        default 0.5 ms.
    n_knots : int
        number of log-spaced knots between knot_one_ms and n_apply
    
    Together with t=0 results in n_knots + 1 total knots.
    """
    t_first = max(1, int(knot_one_ms * fs/1000))
    t_last  = n_apply - 1

    if t_last <= t_first:
        return np.array([0, t_last], dtype=int)

    log_idx = np.unique(
        np.round(
            np.logspace(np.log10(t_first), np.log10(t_last), n_knots)
        ).astype(int)
    )
    return np.unique(np.concatenate([[0], log_idx]))

def remove_stim_events(recording, stim_t, buffer=6, stim_dur=200e-6):
    '''
    Remove stimulation artifacts (+ buffer) from a recording file.

    stim_t : ndarray, shape (n_stims,)
        stim start times in samples relative to recording start
    buffer : int
        number of samples to exclude before and after the stim event
    stim_dur : float
        duration of the stim event in seconds

    Returns the edited recording, as well as start times and duration 
    of stim blank period in samples
    '''
    # recording params
    n_samples = recording.get_num_frames()
    fs = recording.get_sampling_frequency()
    stim_dur_samp = int(stim_dur*fs)

    # get the start/end time for each stim event
    mask_starts = stim_t - buffer
    mask_ends = stim_t + stim_dur_samp + buffer
    mask_duration = stim_dur_samp + (buffer*2)

    # define the windows for stim-free recording
    rec_starts = np.insert(mask_ends, 0, 0)
    rec_ends = np.append(mask_starts, n_samples)

    # make a recording file excluding stim events
    no_stim_recordings = []
    for start, end in zip(rec_starts, rec_ends):
        chunk = recording.frame_slice(start_frame=start, end_frame=end)
        no_stim_recordings.append(chunk)
    rec_no_stim = si.concatenate_recordings(no_stim_recordings)

    return rec_no_stim, mask_starts, mask_duration


def remove_stim_events_dc(recording, stim_t, output_folder, 
                            buffer=6, stim_dur=200e-6, dc_win_sec=0.1e-3):
    '''
    Remove stimulation artifacts (+ buffer) from a recording file.

    To minimize ringing from filtering across sharp changes in baseline voltage, 
    correct the DC offset after each stim event.

    stim_t : ndarray, shape (n_stims,)
        stim start times in samples relative to recording start
    output_folder : string
        file path for temp binary file to store modified recording
    buffer : int
        number of samples to exclude after the stim event
        half of this number of samples will be excluded before the stim event
    stim_dur : float
        duration of the stim event in seconds
    dc_win_sec : float
        duration of time pre/post stim removal window to get median voltage

    Returns the edited recording, as well as start times and duration 
    of stim blank period in samples
    '''
    # recording params
    n_samples = recording.get_num_frames()
    n_channels = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    # convert from seconds to frames
    stim_dur_samp = int(stim_dur*fs)
    dc_win = int(dc_win_sec*fs)

    # get the start/end time for each stim event
    mask_starts = stim_t - buffer//2
    mask_ends = stim_t + stim_dur_samp + buffer
    mask_duration = stim_dur_samp + buffer + buffer//2

    # avoid loading the full recording into traces
    first_start = mask_starts[0] - dc_win
    last_end = mask_ends[-1] + dc_win
    rec_starts = np.insert(mask_ends, 0, first_start)
    rec_ends = np.append(mask_starts, last_end)

    # make a directory for binary files
    output_binary_file = f"{output_folder}/stim_removed.bin"
    if os.path.exists(output_binary_file):
        os.remove(output_binary_file)

    # exclude stim events from stim period
    prev_tail_median = None
    with open(output_binary_file, "ab") as f:
        for start, end in zip(rec_starts, rec_ends):
            chunk = recording.frame_slice(start_frame=start, end_frame=end)
            traces = chunk.get_traces().astype("float32")
    
            # compute the DC offset and adjust
            if prev_tail_median is not None:
                curr_head_median = np.median(traces[:dc_win], axis=0)
                offset = curr_head_median - prev_tail_median
                traces -= offset
            prev_tail_median = np.median(traces[-dc_win:], axis=0)
    
            # append the adjusted traces to the binary
            traces.tofile(f)
            del traces
            del chunk

    # reconstruct the stim period, minus stim events
    rec_no_stim = si.read_binary(
        file_paths=output_binary_file,
        sampling_frequency=fs,
        num_channels=n_channels,
        dtype="float32",
        time_axis=0,
        channel_ids=recording.channel_ids
    )
    rec_no_stim.copy_metadata(recording)
    rec_no_stim.set_probe(recording.get_probe(), in_place=True)
    rec_no_stim.set_property("group", recording.get_property("group"))
    rec_no_stim.set_property("channel_name", recording.get_property("channel_name"))

    # sanity check
    expected_samples = np.sum(rec_ends - rec_starts)
    assert expected_samples == rec_no_stim.get_num_frames()

    # recombine with recordings before/after the stim period
    rec_before_stim = recording.frame_slice(start_frame=0, end_frame=first_start)
    rec_after_stim = recording.frame_slice(start_frame=last_end, end_frame=n_samples)
    rec_stim_removed = si.concatenate_recordings([rec_before_stim.astype("float32"), 
                                                  rec_no_stim, 
                                                  rec_after_stim.astype("float32")])

    return rec_stim_removed, mask_starts, mask_duration


def remove_stim_events_exp_dc(recording, stim_t, output_folder,
                               buffer=6, stim_dur=200e-6,
                               dc_win_sec=0.1e-3,
                               tau_win_sec=3e-3,
                               apply_win_sec=80e-3,
                               min_dc_uv=5.0):
    """
    Remove stim artifacts and correct the post-stim DC offset by fitting
    a per-channel exponential decay at each splice point.

    This function subtracts  A * exp(-t/tau)  per channel, where:
      A is the DC at t~0 (ensures value continuity at the splice)
      tau is fitted from a second DC estimate at tau_win_sec after the splice
            (ensures slope continuity: derivative is continuous at t=0).

    When fitting is ambiguous, tau defaults to 50 ms.

    Parameters
    ----------
    recording : spikeinterface Recording
    stim_t : ndarray (n_stims,)
        Stim start times in samples relative to recording start.
    output_folder : str
        Path for the temporary binary file.
    buffer : int
        Extra samples to blank around the stim event
        before: buffer//2, after: stim_dur_samp + buffer
    stim_dur : float
        Stim pulse duration in seconds.
    dc_win_sec : float
        Window (s) for the short median estimates of DC level.
        Keep small (≤0.3 ms) to avoid averaging over the slope.
    tau_win_sec : float
        Time after the splice (s) at which the DC level is sampled to
        estimate tau.  Choose a value that is:
          • after any fast initial transient (> ~0.3 ms)
          • well before full DC recovery (< tau/2)
          • shorter than the inter-stim interval
        Typical starting value: 1–5 ms.
    apply_win_sec : float
        Duration over which to apply the exponential correction (s).
        Should be ≥ 3–5× the expected tau.
    min_dc_uv : float
        Minimum DC offset (μV) at which to bother correcting.

    Returns
    -------
    rec_stim_removed : recording without stim artifact
    mask_starts : ndarray
        start times in samples from recording start of the artifact removals
    mask_duration : int
        duration in samples of the artifact removal
    """
    # recording params
    n_samples = recording.get_num_frames()
    n_channels = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    # convert from seconds to frames
    stim_dur_samp = int(stim_dur * fs)
    dc_win      = max(3, int(dc_win_sec * fs))
    tau_win   = int(tau_win_sec * fs)      # samples
    apply_win   = int(apply_win_sec * fs)      # samples

    # get the start/end time for each artifact removal
    mask_starts   = stim_t - buffer // 2
    mask_ends     = stim_t + stim_dur_samp + buffer
    mask_duration = stim_dur_samp + buffer + buffer // 2

    # avoid loading the full recording into traces
    first_start = mask_starts[0] - dc_win
    last_end    = mask_ends[-1] + max(apply_win, dc_win)
    rec_starts  = np.insert(mask_ends, 0, first_start)
    rec_ends    = np.append(mask_starts, last_end)

    # make a directory for binary files
    output_binary_file = f"{output_folder}/stim_removed_exp.bin"
    if os.path.exists(output_binary_file):
        os.remove(output_binary_file)

    # exclude stim events from stim period
    prev_tail_median = None
    with open(output_binary_file, "ab") as f:
        for start, end in zip(rec_starts, rec_ends):
            chunk   = recording.frame_slice(start_frame=start, end_frame=end)
            traces  = chunk.get_traces().astype("float32")
            n_chunk = traces.shape[0]

            # adjust post-stim baseline
            if prev_tail_median is not None:
                n_apply = min(apply_win, n_chunk)

                # baseline voltage at concatenation point (value continuity)
                A = np.median(traces[:dc_win], axis=0) - prev_tail_median

                # set default tau for this stim event
                tau_arr = np.full(n_channels, 50e-3, dtype="float64")

                # given enough samples, estimate tau from the data
                if n_chunk > tau_win + dc_win:
                    D_tau = (
                        np.median(traces[tau_win:tau_win+dc_win], axis=0)
                        - prev_tail_median
                    )
                    tau_win_sec_round = tau_win / fs
                    for ch in range(n_channels):
                        a, d = A[ch], D_tau[ch]
                        
                        # only valid if same sign AND |d| < |a|
                        if (np.abs(a) < min_dc_uv or
                            np.sign(a) != np.sign(d) or
                            np.abs(d) >= np.abs(a)):
                            continue   # keep default tau

                        # ratio in (0, 1) for a genuine decay
                        da_ratio = d / a  
                        if not (0.0 < da_ratio < 1.0):
                            continue

                        # estimate tau
                        tau_est = -tau_win_sec_round / np.log(da_ratio)
                        if 3e-4 < tau_est < 2.0:   # 0.3 ms – 2 s
                            tau_arr[ch] = tau_est

                # subtract off exponentially decaying baseline
                t = np.arange(n_apply, dtype="float32") / fs
                correction = (
                    A[np.newaxis, :]
                    * np.exp(-t[:, np.newaxis] / tau_arr[np.newaxis, :])
                )
                traces[:n_apply] -= correction

            # append adjusted traces
            traces.tofile(f)

            # prep for next round
            prev_tail_median = np.median(traces[-dc_win:], axis=0)
            del traces
            del chunk

    # reconstruct the stim period (minus artifacts)
    rec_no_stim = si.read_binary(
        file_paths=output_binary_file,
        sampling_frequency=fs,
        num_channels=n_channels,
        dtype="float32",
        time_axis=0,
        channel_ids=recording.channel_ids,
    )
    rec_no_stim.copy_metadata(recording)
    rec_no_stim.set_probe(recording.get_probe(), in_place=True)
    rec_no_stim.set_property("group", recording.get_property("group"))
    rec_no_stim.set_property("channel_name", recording.get_property("channel_name"))

    expected_samples = np.sum(rec_ends - rec_starts)
    assert expected_samples == rec_no_stim.get_num_frames()

    # recombine with recordings before/after the stim period
    rec_before_stim = recording.frame_slice(start_frame=0, end_frame=first_start)
    rec_after_stim = recording.frame_slice(start_frame=last_end, end_frame=n_samples)
    rec_stim_removed = si.concatenate_recordings([rec_before_stim.astype("float32"), 
                                                  rec_no_stim, 
                                                  rec_after_stim.astype("float32")])


    return rec_stim_removed, mask_starts, mask_duration


def remove_stim_events_spline_dc(recording, stim_t, output_folder,
                                  buffer=6, stim_dur=200e-6,
                                  dc_win_sec=0.1e-3,
                                  spike_kernel_ms=2.0,
                                  apply_win_sec=80e-3,
                                  knot_one_ms=0.5,
                                  n_log_knots=12,
                                  min_dc_uv=5.0):
    """
    Remove stim artifacts by fitting a PCHIP spline to the spike-suppressed
    post-stim baseline and subtracting it.

    Compared to the exponential approach
    -------------------------------------
    - Model-free: no assumed functional form, handles mono-exp, double-exp,
      or any arbitrary recovery shape.
    - More robust near t=0: baseline is estimated from a ~1ms running median
      (~30 samples) rather than a 3-sample median, which suppresses spike
      contamination far more effectively.
    - End-constrained: the spline is forced to 0 at apply_win, so there is
      no residual DC at the end of the correction window. A non-zero residual
      from the exponential approach is the primary cause of the sharp offset
      at the following join when the correction window expires.

    Spline anchor points
    --------------------
    t = 0          : set to the measured DC at the join (exact value continuity)
    t = apply_win  : forced to 0 (no residual DC)
    intermediate   : sampled from the spike-suppressed running median every
                     knot_spacing_ms

    Parameters
    ----------
    spike_kernel_ms : float
        Duration of the running median kernel used for spike suppression (ms).
        Must be longer than spike width (~0.3 ms) and much shorter than the
        DC decay time constant. Default 1 ms; increase to 2–3 ms for dense
        spiking channels.
    knot_one_ms : float
        time (ms) of the first log-spaced knot after the t=0 anchor.
    n_log_knots : int
        number of log-spaced knots between knot_one_ms and apply_win.
    apply_win_sec : float
        Duration over which to apply the correction (s). Should comfortably
        exceed the DC recovery time so the end-constraint to 0 doesn't pull
        the spline down too aggressively. Err on the side of being generous.
    """
    # recording params
    n_samples  = recording.get_num_frames()
    n_channels = recording.get_num_channels()
    fs         = recording.get_sampling_frequency()

    # convert time bins to samples
    stim_dur_samp = int(stim_dur * fs)
    dc_win = max(3, int(dc_win_sec * fs))
    spike_kernel = max(3, int(spike_kernel_ms * fs/1000))
    if spike_kernel % 2 == 0:
        spike_kernel += 1 # keep odd for a symmetric window
    apply_win = int(apply_win_sec * fs)

    # start/end time for each stim blank
    mask_starts   = stim_t - buffer//2
    mask_ends     = stim_t + stim_dur_samp + buffer
    mask_duration = stim_dur_samp + buffer + buffer//2

    # to avoid loading the full recording into traces
    first_start = mask_starts[0] - dc_win
    last_end    = mask_ends[-1] + max(apply_win, dc_win)
    rec_starts  = np.insert(mask_ends, 0, first_start)
    rec_ends    = np.append(mask_starts, last_end)

    # directory for trace binary file
    output_binary_file = f"{output_folder}/stim_removed_spline.bin"
    if os.path.exists(output_binary_file):
        os.remove(output_binary_file)

    # exclude stim events
    prev_tail_median = None
    with open(output_binary_file, "ab") as f:
        for start, end in zip(rec_starts, rec_ends):
            chunk   = recording.frame_slice(start_frame=start, end_frame=end)
            traces  = chunk.get_traces().astype("float32")
            n_chunk = traces.shape[0]

            # adjust post-stim baseline
            if prev_tail_median is not None:
                n_apply = min(apply_win, n_chunk)

                # smooth out spikes from the baseline voltage
                # biased towards left edge for continuity
                baseline = median_filter(
                    traces[:n_apply],
                    size=(spike_kernel, 1),
                    mode='nearest',
                )

                # express as offset from the target (pre-stim) level
                baseline_offset = baseline - prev_tail_median[np.newaxis, :]

                # spline knot timepoints (log-spaced)
                knot_idx = _log_spaced_knot_idx(n_apply, fs, knot_one_ms, n_log_knots)
                t_knots = knot_idx / fs
                t_full  = np.arange(n_apply, dtype='float32') / fs

                # fit spline for baseline correction
                correction = np.zeros((n_apply, n_channels), dtype='float32')
                for ch in range(n_channels):
                    # spline knot voltages
                    y_knots = baseline_offset[knot_idx, ch].copy()

                    # anchor t=0: 
                    # value continuity at the join point
                    y_knots[0] = np.median(traces[:dc_win, ch]) - prev_tail_median[ch]

                    # anchor t=end:
                    # value continuity after the correction window
                    y_knots[-1] = 0.0

                    if np.max(np.abs(y_knots)) < min_dc_uv:
                        continue

                    try:
                        # monotone cubic interpolation
                        spl = PchipInterpolator(t_knots, y_knots)
                        correction[:, ch] = spl(t_full).astype('float32')
                    except Exception:
                        # fallback: linear interpolation through knots
                        correction[:, ch] = np.interp(t_full, t_knots, y_knots).astype('float32')

                # subtract off the fitted baseline
                traces[:n_apply] -= correction

            # append adjusted traces
            traces.tofile(f)

            # prep for next round
            prev_tail_median = np.median(traces[-dc_win:], axis=0)
            del traces
            del chunk

    # reconstruct the stim period (minus artifacts)
    rec_no_stim = si.read_binary(
        file_paths=output_binary_file,
        sampling_frequency=fs,
        num_channels=n_channels,
        dtype="float32",
        time_axis=0,
        channel_ids=recording.channel_ids,
    )
    rec_no_stim.copy_metadata(recording)
    rec_no_stim.set_probe(recording.get_probe(), in_place=True)
    rec_no_stim.set_property("group", recording.get_property("group"))
    rec_no_stim.set_property("channel_name", recording.get_property("channel_name"))

    expected_samples = np.sum(rec_ends - rec_starts)
    assert expected_samples == rec_no_stim.get_num_frames()

    # recombine with recordings before/after the stim period
    rec_before_stim = recording.frame_slice(start_frame=0, end_frame=first_start)
    rec_after_stim = recording.frame_slice(start_frame=last_end, end_frame=n_samples)
    rec_stim_removed = si.concatenate_recordings([rec_before_stim.astype("float32"), 
                                                  rec_no_stim, 
                                                  rec_after_stim.astype("float32")])


    return rec_stim_removed, mask_starts, mask_duration


def split_by_stim(rec_no_stim, mask_starts, mask_duration, post_mask_t):
    '''
    Split a recording file into spontaneous and stim periods.

    This should be used after removing stim events using remove_stim_events,
    which outputs mask_starts and mask_duration.

    For a recording with stim events removed starting at mask_starts and
    lasting for a fixed duration mask_duration, extract post stim events
    of duration post_mask_t. Concatenate the pre-stim windows for a 
    spontaneous-spiking-only recording and the post-stim windows for a
    recording that includes hash.

    Keep track of global time (relative to original recording) to reassemble
    the final spike trains for alignment to behavior.
    '''
    # recording params
    n_samples = rec_no_stim.get_num_frames()
    fs = rec_no_stim.get_sampling_frequency()
    post_stim_frames = int(post_mask_t*fs)
    n_stim = mask_starts.shape[0]

    # get adjusted time windows for post stim period
    mask_periods = np.full(n_stim, mask_duration)
    removed_samples = np.cumsum(np.insert(mask_periods, 0, 0))
    post_stim_start = mask_starts - removed_samples[:-1]
    post_stim_end = post_stim_start + post_stim_frames

    # chop up and concatenate post stim periods (hash)
    post_stim_recordings = []
    for start, end in zip(post_stim_start, post_stim_end):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        post_stim_recordings.append(snippet)
    rec_post_stim = si.concatenate_recordings(post_stim_recordings)

    # save the local and global start/end times
    global_hash_ends = mask_starts + mask_duration + post_stim_frames
    post_stim_map = {}
    post_stim_map['global_starts'] = mask_starts + mask_duration
    post_stim_map['global_ends'] = global_hash_ends
    post_stim_map['snippet_dur'] = np.full(n_stim, post_stim_frames)

    # get time windows for pre stim period
    pre_stim_start = np.insert(post_stim_end, 0, 0)
    pre_stim_end = np.append(post_stim_start, n_samples)

    # chop up and concatenate pre stim periods (spontaneous)
    pre_stim_recordings = []
    for start, end in zip(pre_stim_start, pre_stim_end):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        pre_stim_recordings.append(snippet)
    rec_pre_stim = si.concatenate_recordings(pre_stim_recordings)

    # save the local and global start/end times
    pre_stim_map = {}
    pre_stim_map['global_starts'] = np.insert(global_hash_ends, 0, 0)
    pre_stim_map['global_ends'] = np.append(mask_starts, n_samples + n_stim*mask_duration)
    pre_stim_map['snippet_dur'] = pre_stim_end - pre_stim_start

    # store the recordings
    recordings = {}
    recordings['pre_stim'] = rec_pre_stim
    recordings['post_stim'] = rec_post_stim

    # store the timing information
    maps = {}
    maps['pre_stim'] = pre_stim_map
    maps['post_stim'] = post_stim_map

    return recordings, maps

def split_by_stim_pad(rec_no_stim, mask_starts, mask_duration, post_mask_t, pad_t=0.0):
    '''
    Split a recording file into spontaneous and stim periods.

    This should be used after removing stim events using remove_stim_events,
    which outputs mask_starts and mask_duration.

    For a recording with stim events removed starting at mask_starts and
    lasting for a fixed duration mask_duration, extract post stim events
    of duration post_mask_t. Concatenate the pre-stim windows for a 
    spontaneous-spiking-only recording and the post-stim windows for a
    recording that includes hash.

    Keep track of global time (relative to original recording) to reassemble
    the final spike trains for alignment to behavior.

    Includes padding (defined by pad_t : float, seconds) on each side of the
    post-stim window to avoid corrupting spikes at the window boundaries. Spikes
    in the padding should be discarded using 'true_global_starts' / 'true_global_ends'
    in post_stim_map and the function remove_pad().

    Returns
    -------
    recordings : dict
        contains the two recording files (spontaneous and stim periods)
        'pre_stim' : recording chunks without the post-stim period
        'post_stim' : just the recording chunks within the post_mask_t (plus padding)
    maps : dict of dicts
        contains the global start and end times for the two recordings
        so that they can be reassembled to match the original recording timeline
        
        *note* maps['post_stim']['global_starts'/'snippet_dur'] match the padded
        time windows to map detected spikes correctly back to the original timeline.
        use maps['post_stim']['true_global_starts'/'true_global_ends'] and the function
        remove_pad() to then remove spikes in the padding.
    '''
    # recording params
    n_samples = rec_no_stim.get_num_frames()
    fs = rec_no_stim.get_sampling_frequency()
    post_stim_frames = int(post_mask_t*fs)
    pad_frames = int(pad_t*fs)
    n_stim = mask_starts.shape[0]

    # get adjusted time windows for post stim period (local to rec_no_stim)
    mask_periods = np.full(n_stim, mask_duration)
    removed_samples = np.cumsum(np.insert(mask_periods, 0, 0))
    post_stim_start = mask_starts - removed_samples[:-1]
    post_stim_end = post_stim_start + post_stim_frames

    # padded local windows used to build the matching recording
    post_stim_start_padded = np.clip(post_stim_start - pad_frames, 0, n_samples)
    post_stim_end_padded = np.clip(post_stim_end + pad_frames, 0, n_samples)

    # chop up and concatenate post stim periods (hash), with padding
    post_stim_recordings = []
    for start, end in zip(post_stim_start_padded, post_stim_end_padded):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        post_stim_recordings.append(snippet)
    rec_post_stim = si.concatenate_recordings(post_stim_recordings)

    # save the local and global start/end times
    global_hash_ends = mask_starts + mask_duration + post_stim_frames
    post_stim_map = {}
    post_stim_map['global_starts'] = mask_starts + mask_duration - (post_stim_start - post_stim_start_padded)
    post_stim_map['global_ends'] = global_hash_ends + (post_stim_end_padded - post_stim_end)
    post_stim_map['snippet_dur'] = post_stim_end_padded - post_stim_start_padded
    post_stim_map['true_global_starts'] = mask_starts + mask_duration
    post_stim_map['true_global_ends'] = global_hash_ends

    # get time windows for pre stim period
    pre_stim_start = np.insert(post_stim_end, 0, 0)
    pre_stim_end = np.append(post_stim_start, n_samples)

    # chop up and concatenate pre stim periods (spontaneous)
    pre_stim_recordings = []
    for start, end in zip(pre_stim_start, pre_stim_end):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        pre_stim_recordings.append(snippet)
    rec_pre_stim = si.concatenate_recordings(pre_stim_recordings)

    # save the local and global start/end times
    pre_stim_map = {}
    pre_stim_map['global_starts'] = np.insert(global_hash_ends, 0, 0)
    pre_stim_map['global_ends'] = np.append(mask_starts, n_samples + n_stim*mask_duration)
    pre_stim_map['snippet_dur'] = pre_stim_end - pre_stim_start

    # store the recordings
    recordings = {}
    recordings['pre_stim'] = rec_pre_stim
    recordings['post_stim'] = rec_post_stim

    # store the timing information
    maps = {}
    maps['pre_stim'] = pre_stim_map
    maps['post_stim'] = post_stim_map

    return recordings, maps

def split_hash_spontaneous(rec_no_stim, mask_starts, mask_duration,
                            start_chunk_t, end_chunk_t, pad_t=0.0):
    '''
    Split a recording file into spontaneous and stim-response periods.

    This should be used after removing stim events using remove_stim_events,
    which outputs mask_starts and mask_duration.

    For a recording with stim events removed starting at mask_starts and
    lasting for a fixed duration mask_duration, extract the antidromic hash
    starting at start_chunk_t from the end of the removed stim event.

    Concatenate spontaneous spiking periods (before and up to start_chunk_t after stim)
    and antidromic response windows into separate recording objects.

    Keep track of global time (relative to original recording) to reassemble
    the final spike trains for alignment to behavior.

    Params
    ------
    rec_no_stim : recording object
        recording file with stim events removed
    mask_starts : ndarray, shape (n_stim,)
        stim removal start samples
    mask_duration : int
        n_samples removed starting at mask_starts
    start_chunk_t, end_chunk_t : float
        start and end time in seconds, relative to the end of the stim mask,
        of the antidromic hash to be split out into a separate recording object
    pad_t : float
        padding on each side of the post-stim window to avoid corrupting
        spikes at the window boundaries. Spikes in the padding should be
        discarded using 'true_global_starts' / 'true_global_ends'  in the
        hash_map and the function remove_pad().

    Returns
    -------
    recordings : dict
        contains the two recording files (spontaneous and stim periods)
        'pre_stim' : recording chunks without the post-stim period
        'post_stim' : just the recording chunks within the post_mask_t (plus padding)
    maps : dict of dicts
        contains the global start and end times for the two recordings
        so that they can be reassembled to match the original recording timeline
        
        *note* maps['post_stim']['global_starts'/'snippet_dur'] match the padded
        time windows to map detected spikes correctly back to the original timeline.
        use maps['post_stim']['true_global_starts'/'true_global_ends'] and the function
        remove_pad() to then remove spikes in the padding.
    '''
    # recording params
    n_samples = rec_no_stim.get_num_frames()
    fs = rec_no_stim.get_sampling_frequency()
    start_chunk_frames = int(start_chunk_t*fs)
    end_chunk_frames = int(end_chunk_t*fs)
    pad_frames = int(pad_t*fs)
    n_stim = mask_starts.shape[0]

    # local (to rec_no_stim) time immediately after the stim mask ends
    mask_periods = np.full(n_stim, mask_duration)
    removed_samples = np.cumsum(np.insert(mask_periods, 0, 0))
    mask_end_local = mask_starts - removed_samples[:-1]

    # response window, local to rec_no_stim
    response_start = mask_end_local + start_chunk_frames
    response_end = mask_end_local + end_chunk_frames

    # padded local windows used to build the matching recording
    response_start_padded = np.clip(response_start - pad_frames, 0, n_samples)
    response_end_padded = np.clip(response_end + pad_frames, 0, n_samples)

    # chop up and concatenate response periods (hash), with padding
    hash_recordings = []
    for start, end in zip(response_start_padded, response_end_padded):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        hash_recordings.append(snippet)
    rec_hash = si.concatenate_recordings(hash_recordings)

    # save the local and global start/end times
    true_global_starts = mask_starts + mask_duration + start_chunk_frames
    true_global_ends = mask_starts + mask_duration + end_chunk_frames
    hash_map = {}
    hash_map['global_starts'] = true_global_starts - (response_start - response_start_padded)
    hash_map['global_ends'] = true_global_ends + (response_end_padded - response_end)
    hash_map['snippet_dur'] = response_end_padded - response_start_padded
    hash_map['true_global_starts'] = true_global_starts
    hash_map['true_global_ends'] = true_global_ends

    # get time windows for spontaneous period
    spont_start = np.insert(response_end, 0, 0)
    spont_end = np.append(response_start, n_samples)

    # chop up and concatenate spontaneous periods
    spont_recordings = []
    for start, end in zip(spont_start, spont_end):
        snippet = rec_no_stim.frame_slice(start_frame=start, end_frame=end)
        spont_recordings.append(snippet)
    rec_spont = si.concatenate_recordings(spont_recordings)

    # save the local and global start/end times
    spont_map = {}
    spont_map['global_starts'] = np.insert(true_global_ends, 0, 0)
    spont_map['global_ends'] = np.append(mask_starts, n_samples + n_stim*mask_duration)
    spont_map['snippet_dur'] = spont_end - spont_start

    # store the recordings
    recordings = {}
    recordings['pre_stim'] = rec_spont
    recordings['post_stim'] = rec_hash

    # store the timing information
    maps = {}
    maps['pre_stim'] = spont_map
    maps['post_stim'] = hash_map

    return recordings, maps

def get_global_spikes(spike_t_local, mapping):
    '''
    Converts spike times aligned to a subsampled recording (e.g. recording minus stim period)
    to spike time aligned to the full recording.

    spike_t_local : ndarray, shape (n_spikes,)
        spike times in samples (aligned to subsampled recording snippets)
    mapping : list of dicts
        contains global start/end times and duration for each recording snippet
    '''
    # global timing information
    global_starts = mapping['global_starts']
    global_ends = mapping['global_ends']

    # local timing information
    durations = mapping['snippet_dur']
    n_snippets = durations.shape[0]
    local_starts = np.cumsum(np.insert(durations[:-1], 0, 0))
    local_ends = np.cumsum(durations)
    
    # realign spike times to the full recording
    spike_t_global = np.full(spike_t_local.shape[0], np.nan)
    for snip_idx, (ls, le) in enumerate(zip(local_starts, local_ends)):
        if le == local_ends[-1]:
            spk_idx = (spike_t_local >= ls) & (spike_t_local <= le)
        else:
            spk_idx = (spike_t_local >= ls) & (spike_t_local < le)
        spike_t_global[spk_idx] = spike_t_local[spk_idx] - ls + global_starts[snip_idx]
    
    return spike_t_global

def remove_pad(spike_t_global, true_starts, true_ends):
    '''
    Remove extraneous spikes falling within padded time segments
    (anything outside true_starts to true ends)

    Params
    ------
    spike_t_global : ndarray, shape (n_spikes,)
        Spike times in global sample coordinates (e.g. output of get_global_spikes).
    true_starts, true_ends : ndarray, shape (n_windows,)
        'true_global_starts' / 'true_global_ends'
        From split_by_stim's post_stim_map.
        Windows are assumed non-overlapping.

    Returns
    -------
    keep : bool, shape (n_spikes,)
        True if spike is within true_starts to true_ends windows
    '''
    order = np.argsort(true_starts)
    true_starts = true_starts[order]
    true_ends = true_ends[order]

    # index of the window whose start is <= each spike time
    idx = np.searchsorted(true_starts, spike_t_global, side='right') - 1
    idx = np.clip(idx, 0, len(true_starts) - 1)
    keep = (spike_t_global >= true_starts[idx]) & (spike_t_global < true_ends[idx])
    
    return keep

def get_ks_cluster_groups(kilosort_dir):
    # get the cluster group (good, MUA)
    cluster_info = pd.read_csv(f"{kilosort_dir}/sorter_output/cluster_KSLabel.tsv", sep="\t")
    clu_group = []
    for i, row in cluster_info.iterrows():
        clu_group.append(row["KSLabel"])
    return clu_group


def get_spike_positions(analyzer, subsample=False):
    '''
    Computes spike positions on the probe for use with SpikePositionsView in phy

    Params
    ------
    analyzer : SpikeInterface object
        the analyzer object we are exporting to phy
    subsample : bool
        if True, subsamples spikes for speed
    '''
    analyzer.compute("spike_locations", method="center_of_mass")
    locations = analyzer.get_extension("spike_locations").get_data()  # structured array, one row per spike

    # match the spike order phy's exporter used
    spike_order = analyzer.sorting.to_spike_vector(concatenated=False)[0]
    n_spikes = len(spike_order)

    # create an array of spike positions
    if subsample:
        max_spikes_for_locations = 200000
        rng = np.random.default_rng(0)
        if n_spikes > max_spikes_for_locations:
            keep_idx = rng.choice(n_spikes, size=max_spikes_for_locations, replace=False)
        else:
            keep_idx = np.arange(n_spikes)

        spike_positions = np.full((n_spikes, 2), np.nan, dtype="float64")
        spike_positions[keep_idx, 0] = locations["x"][keep_idx]
        spike_positions[keep_idx, 1] = locations["y"][keep_idx]
    else:
        spike_positions = np.column_stack([locations["x"], locations["y"]]).astype("float64")

    return spike_positions


def load_ks4_sorting(recording, kilosort_dir):
    times = np.load(f"{kilosort_dir}/spike_times.npy")
    labels_raw = np.load(f"{kilosort_dir}/spike_clusters.npy")
    pos = np.load(f"{kilosort_dir}/spike_positions.npy")
    amp = np.load(f"{kilosort_dir}/amplitudes.npy")

    # load the sorting, but use 0s for channels because KS
    # doesn't save them. we'll back them out in the next step.
    ks_st = dartsort.DARTsortSorting(
        times_samples=times,
        labels=labels,
        channels=np.zeros_like(labels),
        ephemeral_features=dict(amplitudes=amp)
    )

    # make quick templates (not suitable for matching)
    templates = dartsort.TemplateData.from_config(
        recording=recording,
        sorting=ks_st,
        template_cfg=dartsort.raw_template_cfg,
    )
    
    # set spike channels from template main channels
    main_channels = np.ptp(templates.templates, axis=1).argmax(1)
    
    # final spike train with channel info
    ks_st = dartsort.DARTsortSorting(
        times_samples=times,
        labels=labels,
        channels=main_channels[labels],
        ephemeral_features=dict(
            positions=pos,
            amplitudes=amp,
            times_seconds=recording.sample_index_to_time(times),
        ),
    )
    return ks_st


def split_templates_by_shank(ds_folder, template_data, shank_idx):
    '''
    ds_folder : string
        path to dartsort files
    shank_idx : array of bools, shape (n_channels,)
        which channels belong to each shank
    '''
    # divide channels by shank
    n_channels = shank_idx.shape[0]
    all_channels = np.arange(n_channels)
    shank0_ch = all_channels[shank_idx==0]
    shank1_ch = all_channels[shank_idx==1]

    # get the best channel for each ks unit
    with np.load(f"{ds_folder}/ks_st_realigned.npz") as data:
        ks_labels, label_idx = np.unique(data['labels'], return_index=True)
        best_ch = data['channels'][label_idx]

    # match to the dartsort templates
    with np.load(f"{ds_folder}/template_data.npz") as data:
        ds_labels = data['unit_ids']
        shared_labels, ds_idx, ks_idx = np.intersect1d(ks_labels, ds_labels, return_indices=True)
        best_ch = best_ch[ds_idx]

    # divide by shank
    shank_0_idx = np.isin(best_ch, shank0_ch)
    shank0_units = ds_labels[shank_0_idx]
    shank1_units = ds_labels[~shank_0_idx]

    # package into a new template data structure with correct units/channels
    template_data_0 = template_data[shank0_units]
    template_data_0 = replace(
        template_data_0,
        templates=template_data_0.templates[:, :, shank0_ch],
        spike_counts_by_channel=template_data_0.spike_counts_by_channel[:, shank0_ch],
        registered_geom=template_data_0.registered_geom[shank0_ch],
    )

    template_data_1 = template_data[shank1_units]
    template_data_1 = replace(
        template_data_1,
        templates=template_data_1.templates[:, :, shank1_ch],
        spike_counts_by_channel=template_data_1.spike_counts_by_channel[:, shank1_ch],
        registered_geom=template_data_1.registered_geom[shank1_ch],
    )

    return template_data_0, template_data_1