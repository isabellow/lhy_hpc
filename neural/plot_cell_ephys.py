#!/usr/bin/env python
"""
plot_cell_ephys.py
==================
Based on Payne et al, 2021 Fig 1D

Raw (high-pass filtered) voltage trace on the left,
~20 individual spike waveforms overlaid with the mean waveform on the right.

Everything is taken from the cell's best (max amplitude) channel.

Usage
-----
    # every good unit in a session
    python plot_cell_ephys.py --intan-folder .../LIM63_240610_131820/ \
                              --ks-dir .../kilosort4/ --out-dir ../figures/LIM63_240610/

    # one unit
    python plot_cell_ephys.py ... --cluster-id 137

From a notebook:

    from plot_cell_ephys import run_session
    results = run_session(intan_folder=..., ks_dir=..., out_dir=...)

How the I/O is organised
------------------------
Pulling filtered traces off disk is the expensive part, so the recording is read
in two bounded passes and never in full:

  Pass 1 (once per session, ~40 s of data, shared by ALL units)
      A pool of short chunks evenly spaced through the recording is read once
      with all channels, and every unit's waveform snippets are cut out of the
      same chunks. Sampling the whole session this way also averages over drift
      instead of characterising a unit from one moment in time.

  Pass 2 (per unit, one 4 s window)
      The example trace. Each unit needs a window where it was actually firing,
      so this one can't be shared; the window is chosen from the spike times
      alone (no reads) and then read exactly once.

Peak channels come from Kilosort's templates, which costs no raw reads at all.

Total: ~40 s + 4 s x n_units of filtered data, versus roughly one full pass over
the recording per unit per operation in a naive implementation.

Assumptions / caveats
---------------------
* Spike sample indices must index into the recording you load. If you sorted a
  stitched / stim-blanked / cropped recording, pass that SpikeInterface binary
  with --si-folder rather than the raw Intan folder.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

PRE_MS = 0.5      # ms before the spike time   (matches getSessionWaveforms.m)
TOTAL_MS = 2.0    # ms total waveform duration


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
_WARNED_UNSCALED = False


def get_traces_uv(rec, start_frame, end_frame, channel_ids=None):
    """rec.get_traces() in microvolts, across SpikeInterface versions.

    SI <=0.102 uses return_scaled=, >=0.103 uses return_in_uV=.
    """
    global _WARNED_UNSCALED
    kw = dict(start_frame=int(start_frame), end_frame=int(end_frame),
              channel_ids=channel_ids)
    for key in ("return_in_uV", "return_scaled"):
        try:
            return rec.get_traces(**kw, **{key: True})
        except TypeError:
            continue
        except ValueError:
            break
    if not _WARNED_UNSCALED:
        warnings.warn("No uV gain on this recording; traces are in raw ADC units.")
        _WARNED_UNSCALED = True
    return rec.get_traces(**kw)


def load_recording(intan_folder=None, si_folder=None, stream_id="0",
                   freq_min=300.0, freq_max=6000.0, cmr=True, channel_ids=None):
    """Return (raw, filtered) recordings. Lazy -- nothing is read here."""
    if si_folder is not None:
        rec = si.load(str(si_folder))
    elif intan_folder is not None:
        rec = se.read_intan(str(Path(intan_folder) / "info.rhd"), stream_id=stream_id)
    else:
        raise ValueError("Provide either intan_folder or si_folder.")
    if channel_ids is not None:
        rec = rec.select_channels(list(channel_ids))
    rec_filt = spre.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
    if cmr:
        rec_filt = spre.common_reference(rec_filt, reference="global", operator="median")
    return rec, rec_filt


def read_params_fs(ks_dir):
    p = Path(ks_dir) / "params.py"
    if not p.exists():
        return None
    m = re.search(r"sample_rate\s*=\s*([0-9.eE+-]+)", p.read_text())
    return float(m.group(1)) if m else None


def read_cluster_labels(ks_dir):
    """{cluster_id: label} from cluster_group.tsv, falling back to KSLabel."""
    ks_dir = Path(ks_dir)
    for fname, col in (("cluster_group.tsv", "group"),
                       ("cluster_KSLabel.tsv", "KSLabel")):
        f = ks_dir / fname
        if not f.exists():
            continue
        rows = [ln.split("\t") for ln in f.read_text().splitlines() if ln.strip()]
        header, rows = rows[0], rows[1:]
        if "cluster_id" not in header or col not in header:
            continue
        ci, li = header.index("cluster_id"), header.index(col)
        return {int(r[ci]): r[li].strip() for r in rows if len(r) > max(ci, li)}
    return {}


def load_phy_spikes(ks_dir):
    """Group spikes by cluster in a single sort.

    Returns
    -------
    spikes    : {cluster_id: sorted int64 sample indices}
    templates : {cluster_id: template index per spike} ({} if unavailable)
    """
    ks_dir = Path(ks_dir)
    st = np.load(ks_dir / "spike_times.npy").astype(np.int64).ravel()
    sc = np.load(ks_dir / "spike_clusters.npy").astype(np.int64).ravel()
    keep = st >= 0
    st, sc = st[keep], sc[keep]

    tf = ks_dir / "spike_templates.npy"
    tmpl = np.load(tf).astype(np.int64).ravel()[keep] if tf.exists() else None

    # sort by cluster, then by time: one pass, and each group is time-sorted
    order = np.lexsort((st, sc))
    st, sc = st[order], sc[order]
    tmpl = tmpl[order] if tmpl is not None else None

    uniq, starts = np.unique(sc, return_index=True)
    ends = np.append(starts[1:], sc.size)
    spikes = {int(u): st[a:b] for u, a, b in zip(uniq, starts, ends)}
    templates = ({int(u): tmpl[a:b] for u, a, b in zip(uniq, starts, ends)}
                 if tmpl is not None else {})
    return spikes, templates


def best_channels_from_templates(ks_dir, cluster_ids, spike_templates, n_channels):
    """Peak channel per cluster from Kilosort's templates -- no raw reads.

    Returns {cluster_id: 0-based channel index into the loaded recording}, or
    None if the template files aren't usable.
    """
    ks_dir = Path(ks_dir)
    tpath = ks_dir / "templates.npy"
    if not tpath.exists() or not spike_templates:
        return None
    templates = np.load(tpath)                      # (n_templates, n_samp, n_chan)
    peak = np.argmax(templates.max(1) - templates.min(1), axis=1)

    cmap_path = ks_dir / "channel_map.npy"
    cmap = (np.load(cmap_path).astype(int).ravel() if cmap_path.exists()
            else np.arange(templates.shape[2]))
    if cmap.size != n_channels:
        warnings.warn(
            f"channel_map.npy has {cmap.size} channels but the loaded recording "
            f"has {n_channels}. The sorted recording is probably not the one "
            "loaded -- falling back to peak channels from the raw data.")
        return None

    out = {}
    for cid in cluster_ids:
        t = spike_templates.get(cid)
        if t is None or t.size == 0:
            return None
        modal = int(np.bincount(t).argmax())        # merged clusters: commonest template
        out[cid] = int(cmap[peak[modal]])
    return out


# --------------------------------------------------------------------------- #
# Pass 1: one chunk pool, shared by every unit
# --------------------------------------------------------------------------- #
def collect_waveforms(rec_filt, spikes, best_ch, pre_samp, n_samp,
                      n_chunks=40, chunk_sec=1.0, n_wf=300, verbose=True):
    """Cut waveform snippets for every unit out of a shared pool of chunks.

    Chunks are evenly spaced across the recording; each is read once with all
    channels and then served to every unit. Per unit this accumulates
    single-channel snippets on its peak channel plus a running all-channel mean
    (cheap, and gives the multi-channel footprint for free).
    """
    fs = rec_filt.get_sampling_frequency()
    n_total = rec_filt.get_num_samples()
    n_chan = rec_filt.get_num_channels()
    chunk_n = int(round(chunk_sec * fs))
    post = n_samp - pre_samp

    starts = np.unique(np.linspace(0, max(n_total - chunk_n, 0),
                                   n_chunks).astype(np.int64))
    cids = list(spikes)
    snips = {c: [] for c in cids}
    acc = {c: np.zeros((n_samp, n_chan)) for c in cids}
    count = {c: 0 for c in cids}

    for i, s0 in enumerate(starts):
        s1 = int(min(s0 + chunk_n, n_total))
        active = [c for c in cids if count[c] < n_wf]
        if not active:
            break
        traces = get_traces_uv(rec_filt, s0, s1)          # all channels, filtered once
        for c in active:
            spk = spikes[c]
            lo = np.searchsorted(spk, s0 + pre_samp)
            hi = np.searchsorted(spk, s1 - post)
            ch = best_ch[c]
            for s in spk[lo:hi]:
                k = int(s - s0 - pre_samp)
                snip = traces[k:k + n_samp]
                acc[c] += snip
                snips[c].append(snip[:, ch].copy())
                count[c] += 1
                if count[c] >= n_wf:
                    break
        if verbose and (i + 1) % 10 == 0:
            print(f"  chunk {i + 1}/{len(starts)}", flush=True)

    out = {}
    for c in cids:
        wf = np.asarray(snips[c]) if snips[c] else np.zeros((0, n_samp))
        out[c] = dict(waveforms=wf,
                      mean_all_channels=acc[c] / count[c] if count[c] else None)
    return out


# --------------------------------------------------------------------------- #
# Pass 2: one example trace per unit
# --------------------------------------------------------------------------- #
def window_candidates(spk, n_total, fs, win_sec=4.0, step_sec=0.25,
                      mode="max", t_start=None, n_candidates=8):
    """Rank window start samples by how much the unit fires there. No disk I/O.

    mode='max'     : most spikes
    mode='typical' : 75th percentile of spike count, often more representative
    mode='time'    : start at t_start seconds
    """
    if mode == "time":
        if t_start is None:
            raise ValueError("mode='time' requires t_start")
        return [int(round(t_start * fs))]

    step = max(int(round(step_sec * fs)), 1)
    nbins = max(n_total // step, 1)
    counts = np.bincount(np.clip(spk // step, 0, nbins - 1),
                         minlength=nbins).astype(float)
    nwin = max(int(round(win_sec / step_sec)), 1)
    if counts.size < nwin:
        return [0]
    win_counts = np.convolve(counts, np.ones(nwin), mode="valid")

    if mode == "max":
        order = np.argsort(win_counts)[::-1]
    elif mode == "typical":
        nz = win_counts[win_counts > 0]
        target = np.percentile(nz, 75) if nz.size else 0.0
        order = np.argsort(np.abs(win_counts - target))
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return [int(i * step) for i in order[:n_candidates]]


def read_trace_window(rec_filt, channel_index, candidates, win_sec,
                      artifact_uv=2000.0):
    """Read candidate windows until one is artifact-free. Usually one read.

    Returns (start_sample, single-channel trace, all-channel window).
    """
    fs = rec_filt.get_sampling_frequency()
    n_total = rec_filt.get_num_samples()
    win_n = int(round(win_sec * fs))
    first = None
    for s0 in candidates:
        s0 = int(np.clip(s0, 0, max(n_total - win_n, 0)))
        traces = get_traces_uv(rec_filt, s0, min(s0 + win_n, n_total))
        trace = traces[:, channel_index]
        if first is None:
            first = (s0, trace, traces)
        if np.max(np.abs(trace)) < artifact_uv:
            return s0, trace, traces
    warnings.warn("Every candidate window exceeded artifact_uv; using the best one.")
    return first


def harvest_extra_waveforms(traces, spk, s0, channel_index, pre_samp, n_samp,
                            n_needed):
    """Extra snippets from a window already in memory -- no extra reads."""
    post = n_samp - pre_samp
    s1 = s0 + traces.shape[0]
    lo = np.searchsorted(spk, s0 + pre_samp)
    hi = np.searchsorted(spk, s1 - post)
    out = [traces[int(s - s0 - pre_samp):int(s - s0 - pre_samp) + n_samp,
                  channel_index].copy()
           for s in spk[lo:hi][:n_needed]]
    return np.asarray(out) if out else np.zeros((0, n_samp))


def waveform_baseline(wf, wf_t, before_ms=-0.5):
    """Pre-spike baseline. Works for (n_samp,) and (n_samp, n_chan) arrays."""
    m = wf_t < before_ms
    if not m.any():
        m = np.arange(wf_t.size) < max(wf_t.size // 4, 1)
    return np.median(wf[m], axis=0)


def _half_width_ms(wf, fs):
    """Full width at half maximum of the dominant peak, in ms.

    Polarity-agnostic, so it can be compared across up- and down-going spikes
    (unlike trough-to-peak, which is only defined for down-going ones).
    """
    i = int(np.argmax(np.abs(wf)))
    half = wf[i] / 2.0
    sgn = np.sign(wf[i])

    def cross(step):
        j = i
        while 0 <= j + step < wf.size and sgn * (wf[j + step] - half) > 0:
            j += step
        k = j + step
        if not (0 <= k < wf.size):
            return float(j)
        denom = wf[k] - wf[j]
        return float(j) if denom == 0 else j + (half - wf[j]) / denom * step

    return (cross(1) - cross(-1)) / fs * 1e3


def waveform_metrics(mean_wf, wf_t, fs):
    """Polarity, trough/peak width, and half-width of a mean waveform.

    A cell is called 'positive' when the largest excursion from baseline is
    upward. For those, width is measured peak-to-following-trough, i.e. the
    mirror image of the Payne et al. trough-to-following-peak definition.
    Widths are therefore NOT comparable across polarities -- see half_width_ms
    for a measure that is.
    """
    wf = np.asarray(mean_wf, float) - waveform_baseline(mean_wf, wf_t)
    positive = wf.max() > abs(wf.min())
    first = int(np.argmax(wf)) if positive else int(np.argmin(wf))
    if first >= wf.size - 1:
        width = np.nan
    else:
        rest = wf[first:]
        second = int(np.argmin(rest)) if positive else int(np.argmax(rest))
        width = second / fs * 1e3
    return dict(polarity="positive" if positive else "negative",
                spike_width_ms=width,
                half_width_ms=_half_width_ms(wf, fs))


def channel_extent(mean_all, wf_t, thresh_uv=20.0):
    """How many channels carry the spike: peak-to-peak above thresh_uv."""
    if mean_all is None:
        return -1, None
    m = np.asarray(mean_all, float) - waveform_baseline(mean_all, wf_t)
    amps = m.max(axis=0) - m.min(axis=0)
    return int(np.sum(amps > thresh_uv)), amps


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _scalebar(ax, x, y, dx, dy, xlabel, ylabel, color="k", lw=1.5, fs=7):
    ax.plot([x, x], [y, y + dy], color=color, lw=lw, clip_on=False)
    ax.plot([x, x + dx], [y, y], color=color, lw=lw, clip_on=False)
    ax.text(x + dx / 2, y - abs(dy) * 0.08, xlabel, ha="center", va="top",
            fontsize=fs, color=color, clip_on=False)
    ax.text(x - dx * 0.06, y + dy / 2, ylabel, ha="right", va="center",
            fontsize=fs, color=color, clip_on=False)


def plot_fig1d(data, n_show=20, color=None, width_thresh_ms=0.2,
               figsize=(9, 2.6), rasterize=True, scalebar_step_uv=25):
    """Draw the Fig. 1D-style panel from one unit's data dict.

    The trace and the waveforms are drawn on a single shared uV scale with
    their baselines aligned, so the one vertical scale bar (bottom right)
    applies to both panels.
    """
    t = data["trace_t"]
    wf_t = data["waveform_t"]
    # Align baselines on zero: pre-spike median for the waveforms, median of
    # the trace itself. The same shift is applied to the individual waveforms
    # and the mean so their relationship is preserved.
    wf_base = waveform_baseline(data["mean_waveform"], wf_t)
    wfs = data["waveforms"] - wf_base
    mean_wf = data["mean_waveform"] - wf_base
    trace = data["trace"] - np.median(data["trace"])

    if color is None:
        if data.get("polarity") == "positive":
            color = "xkcd:saffron"
        elif data["spike_width_ms"] >= width_thresh_ms:
            color = "xkcd:scarlet"
        else:
            color = "xkcd:cobalt blue"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.08)
    ax_tr, ax_wf = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    rng = np.random.default_rng(0)
    idx = rng.choice(wfs.shape[0], min(n_show, wfs.shape[0]), replace=False)
    show = wfs[idx]

    # one shared y range, wide enough for both panels
    lo = min(np.percentile(trace, 0.05) * 1.15, show.min())
    hi = max(np.percentile(trace, 99.95) * 1.15, show.max())
    span = hi - lo
    ylim = (lo - 0.20 * span, hi + 0.14 * span)

    # left: voltage trace, this cell's spikes ticked above
    ax_tr.plot(t, trace, color="k", lw=0.4, rasterized=rasterize)
    st = data["trace_spike_t"]
    ax_tr.plot(st, np.full_like(st, hi + 0.07 * span), marker="|", ls="none",
               color=color, ms=6, mew=1.2)
    ax_tr.set_xlim(t[0], t[-1])
    ax_tr.set_ylim(*ylim)
    ax_tr.axis("off")

    # right: individual waveforms + mean, same y scale
    ax_wf.plot(wf_t, show.T, color="k", lw=0.4, alpha=0.35)
    ax_wf.plot(wf_t, mean_wf, color=color, lw=2)
    ax_wf.set_xlim(wf_t[0], wf_t[-1])
    ax_wf.set_ylim(*ylim)
    ax_wf.axis("off")

    # Vertical bar drawn about as long on the page as the 1 ms bar, rounded to
    # a multiple of scalebar_step_uv. Needs the axes aspect, hence get_position.
    pos = ax_wf.get_position()
    uv_per_inch = (ylim[1] - ylim[0]) / (pos.height * figsize[1])
    ms_per_inch = (wf_t[-1] - wf_t[0]) / (pos.width * figsize[0])
    bar_uv = uv_per_inch * (1.0 / ms_per_inch)
    bar_uv = max(scalebar_step_uv,
                 int(round(bar_uv / scalebar_step_uv)) * scalebar_step_uv)

    bar_y = ylim[0] + 0.02 * span
    ax_tr.plot([t[0] + 0.02 * np.ptp(t), t[0] + 0.02 * np.ptp(t) + 0.5],
               [bar_y, bar_y], color="k", lw=1.5, clip_on=False)
    ax_tr.text(t[0] + 0.02 * np.ptp(t) + 0.25, bar_y - 0.03 * span, "0.5 s",
               ha="center", va="top", fontsize=7, clip_on=False)
    _scalebar(ax_wf, wf_t[0] + 0.1, bar_y, 1.0, bar_uv,
              "1 ms", f"{bar_uv:g} µV")

    extent = data.get("n_channels_extent", -1)
    fig.suptitle(
        f"cell {data['cluster_id']}"
        f"{' (' + data['label'] + ')' if data['label'] else ''}   "
        f"ch {data['max_channel_name']}   {data['mean_rate_hz']:.1f} Hz   "
        f"n = {data['n_spikes']} spikes   width {data['spike_width_ms']:.2f} ms   "
        + (f"extent {extent} ch" if extent >= 0 else ""),
        fontsize=8, y=1.02, x=0.02, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def run_session(intan_folder=None, si_folder=None, ks_dir=None, out_dir=None,
                cluster_ids=None, only_good=True, stream_id="0",
                channel_ids=None, cmr=True, freq_min=300.0, freq_max=6000.0,
                n_chunks=40, chunk_sec=1.0, n_wf=300, min_wf=20, n_show=20,
                win_sec=4.0, window_mode="max", t_start=None,
                artifact_uv=2000.0, peak_source="templates", extent_thresh_uv=20.0,
                save_pdf=False, save_npz=False, summary_csv=True, verbose=True):
    """Make a figure for one unit or every good unit in a session.

    Returns {cluster_id: data dict}.
    """
    t0 = time.time()
    rec, rec_filt = load_recording(intan_folder=intan_folder, si_folder=si_folder,
                                   stream_id=stream_id, freq_min=freq_min,
                                   freq_max=freq_max, cmr=cmr,
                                   channel_ids=channel_ids)
    fs = rec.get_sampling_frequency()
    n_total = rec.get_num_samples()
    n_chan = rec.get_num_channels()

    ks_fs = read_params_fs(ks_dir)
    if ks_fs is not None and not np.isclose(ks_fs, fs, rtol=1e-4):
        warnings.warn(f"params.py sample rate ({ks_fs}) != recording ({fs}).")

    spikes_all, templates_all = load_phy_spikes(ks_dir)
    labels = read_cluster_labels(ks_dir)

    if cluster_ids is not None:
        cids = [int(c) for c in np.atleast_1d(cluster_ids)]
    elif only_good and labels:
        cids = sorted(c for c in spikes_all if labels.get(c, "") == "good")
    else:
        cids = sorted(spikes_all)
    missing = [c for c in cids if c not in spikes_all]
    if missing:
        raise ValueError(f"No spikes for cluster(s) {missing} in {ks_dir}")
    if not cids:
        raise ValueError("No clusters selected (no 'good' labels in cluster_group.tsv?)")
    spikes = {c: spikes_all[c] for c in cids}

    last = max(int(s[-1]) for s in spikes.values())
    if last > n_total:
        raise ValueError(
            f"Spikes extend past the end of this recording ({last} > {n_total} "
            "samples). The sorted recording is probably not the one loaded -- "
            "see the caveats in the docstring.")

    pre_samp = int(round(PRE_MS * 1e-3 * fs))
    n_samp = int(round(TOTAL_MS * 1e-3 * fs))

    # peak channels -- free when Kilosort's templates are available
    best_ch = None
    if peak_source == "templates":
        best_ch = best_channels_from_templates(ks_dir, cids, templates_all, n_chan)
    if best_ch is None:
        if verbose:
            print("peak channels: deriving from raw data (small extra pass)")
        probe = collect_waveforms(rec_filt, spikes, {c: 0 for c in cids}, pre_samp,
                                  n_samp, n_chunks=max(n_chunks // 3, 6),
                                  chunk_sec=chunk_sec, n_wf=50, verbose=False)
        best_ch = {c: (int(np.argmax(probe[c]["mean_all_channels"].max(0)
                                     - probe[c]["mean_all_channels"].min(0)))
                       if probe[c]["mean_all_channels"] is not None else 0)
                   for c in cids}

    names = rec.get_property("channel_name")

    # ---- pass 1: shared chunk pool ----
    if verbose:
        print(f"{len(cids)} units | pass 1: {n_chunks} x {chunk_sec:g} s chunks "
              f"({n_chunks * chunk_sec:.0f} s of data, shared)", flush=True)
    pool = collect_waveforms(rec_filt, spikes, best_ch, pre_samp, n_samp,
                             n_chunks=n_chunks, chunk_sec=chunk_sec, n_wf=n_wf,
                             verbose=verbose)

    # ---- pass 2: per-unit trace window ----
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    results, low_wf = {}, []
    for n, c in enumerate(cids, 1):
        spk, ch = spikes[c], best_ch[c]
        cands = window_candidates(spk, n_total, fs, win_sec=win_sec,
                                  mode=window_mode, t_start=t_start)
        s0, trace, window = read_trace_window(rec_filt, ch, cands, win_sec,
                                              artifact_uv=artifact_uv)

        wfs = pool[c]["waveforms"]
        if wfs.shape[0] < min_wf:      # rare low-rate unit: top up for free
            extra = harvest_extra_waveforms(window, spk, s0, ch, pre_samp, n_samp,
                                            min_wf - wfs.shape[0])
            if extra.size:
                wfs = np.vstack([wfs, extra])
        if wfs.shape[0] == 0:
            warnings.warn(f"cluster {c}: no usable waveform snippets; skipping")
            continue
        if wfs.shape[0] < min_wf:
            low_wf.append((c, wfs.shape[0]))
        mean_wf = wfs.mean(axis=0)

        in_win = spk[(spk >= s0) & (spk < s0 + trace.size)]
        wf_t = (np.arange(n_samp) - pre_samp) / fs * 1e3
        extent, amps = channel_extent(pool[c]["mean_all_channels"], wf_t,
                                      thresh_uv=extent_thresh_uv)
        data = dict(
            cluster_id=c, label=labels.get(c, ""), fs=fs,
            max_channel_index=ch, max_channel_id=rec.channel_ids[ch],
            max_channel_name=(str(names[ch]) if names is not None
                              else str(rec.channel_ids[ch])),
            n_spikes=int(spk.size), n_waveforms=int(wfs.shape[0]),
            mean_rate_hz=float(spk.size / (n_total / fs)),
            trace=trace, trace_t=np.arange(trace.size) / fs,
            trace_start_s=s0 / fs, trace_spike_t=(in_win - s0) / fs,
            waveforms=wfs, mean_waveform=mean_wf,
            waveform_t=wf_t,
            mean_waveform_all_channels=pool[c]["mean_all_channels"],
            peak_to_peak_uv=float(np.ptp(mean_wf)),
            n_channels_extent=extent,
            channel_amplitudes_uv=amps,
            **waveform_metrics(mean_wf, wf_t, fs),
        )
        results[c] = data

        if out_dir is not None:
            fig = plot_fig1d(data, n_show=n_show)
            stem = out_dir / f"cell_{c}_wf_trace"
            fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
            if save_pdf:
                fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)
            if save_npz:
                np.savez_compressed(f"{stem}.npz",
                                    **{k: v for k, v in data.items() if v is not None})
        if verbose and (n % 10 == 0 or n == len(cids)):
            print(f"  unit {n}/{len(cids)}  ({time.time() - t0:.0f} s)", flush=True)

    if low_wf:
        warnings.warn(f"{len(low_wf)} unit(s) had fewer than {min_wf} waveforms "
                      f"(lowest: {min(k for _, k in low_wf)}). Raise --n-chunks "
                      "or --chunk-sec for these.")

    if out_dir is not None and summary_csv and results:
        cols = ["cluster_id", "label", "max_channel_name", "max_channel_index",
                "n_spikes", "mean_rate_hz", "peak_to_peak_uv", "polarity",
                "spike_width_ms", "half_width_ms", "n_channels_extent",
                "n_waveforms", "trace_start_s"]
        with open(out_dir / "unit_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for c in sorted(results):
                w.writerow([results[c][k] for k in cols])
        if verbose:
            print(f"wrote {out_dir / 'unit_summary.csv'}")

    if verbose:
        print(f"done: {len(results)} units in {time.time() - t0:.1f} s")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intan-folder", help="folder with info.rhd + amplifier.dat")
    p.add_argument("--si-folder", help="SpikeInterface binary folder that was sorted")
    p.add_argument("--ks-dir", required=True, help="Kilosort/Phy output folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cluster-id", type=int, nargs="+", default=None,
                   help="one or more clusters; default is every good unit")
    p.add_argument("--all-clusters", action="store_true",
                   help="include mua/unsorted, not just 'good'")
    p.add_argument("--stream-id", default="0")
    p.add_argument("--channel-ids", nargs="+", default=None,
                   help="subset of channel ids to load (must match what was sorted)")
    p.add_argument("--n-chunks", type=int, default=40,
                   help="chunks in the shared pool (default 40)")
    p.add_argument("--chunk-sec", type=float, default=1.0)
    p.add_argument("--n-wf", type=int, default=300, help="max waveforms per unit")
    p.add_argument("--min-wf", type=int, default=20)
    p.add_argument("--n-show", type=int, default=20, help="waveforms drawn")
    p.add_argument("--win-sec", type=float, default=4.0)
    p.add_argument("--window-mode", default="max", choices=["max", "typical", "time"])
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--peak-source", default="templates", choices=["templates", "raw"])
    p.add_argument("--artifact-uv", type=float, default=2000.0)
    p.add_argument("--extent-thresh-uv", type=float, default=20.0,
                   help="channel counts toward the extent if its mean waveform "
                        "peak-to-peak exceeds this (default 20 uV)")
    p.add_argument("--no-cmr", action="store_true")
    p.add_argument("--pdf", action="store_true", help="also save vector PDFs")
    p.add_argument("--save-npz", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    matplotlib.use("Agg")
    run_session(
        intan_folder=args.intan_folder, si_folder=args.si_folder,
        ks_dir=args.ks_dir, out_dir=args.out_dir, cluster_ids=args.cluster_id,
        only_good=not args.all_clusters, stream_id=args.stream_id,
        channel_ids=args.channel_ids, cmr=not args.no_cmr,
        n_chunks=args.n_chunks, chunk_sec=args.chunk_sec, n_wf=args.n_wf,
        min_wf=args.min_wf, n_show=args.n_show, win_sec=args.win_sec,
        window_mode=args.window_mode, t_start=args.t_start,
        peak_source=args.peak_source, artifact_uv=args.artifact_uv,
        extent_thresh_uv=args.extent_thresh_uv,
        save_pdf=args.pdf, save_npz=args.save_npz, verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
