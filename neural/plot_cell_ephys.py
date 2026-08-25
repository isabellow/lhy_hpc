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
    python plot_cell_ephys.py \
        --intan-folder  Z:/Isabel/data/hpc_implants/LIM63/LIM63_240610/LIM63_240610_131820/ \
        --ks-dir        Z:/.../LIM63_240610_131820/kilosort4/ \
        --cluster-id    137 \
        --out-dir       ../figures/example_cells/

Or import and call from a notebook:

    from plot_cell_ephys import make_cell_figure
    fig, data = make_cell_figure(intan_folder, ks_dir, cluster_id=137)

Assumptions / caveats
---------------------
* Kilosort was run on the *full, continuous* recording, so that sample indices in
  spike_times.npy index directly into the raw Intan data. If you sorted a
  stitched / stim-blanked / cropped recording (e.g. the split recordings in
  scratch_si_ks_templates.ipynb), pass that saved SpikeInterface binary with
  --si-folder instead of --intan-folder, or the trace and the spike times will
  not line up.
* If you dropped noisy channels before sorting, pass the same subset with
  --channel-ids so that channel indices match. Otherwise "max channel" is still
  computed correctly from the raw data (this script finds it empirically), but
  channel *names* may be off relative to your Kilosort channel_map.
* Traces and waveforms are in microvolts (Intan gain 0.195 uV/bit, applied by
  SpikeInterface).
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

# waveform window, matching getSessionWaveforms.m
PRE_MS = 0.5      # ms before the spike time
TOTAL_MS = 2.0    # ms total waveform duration


# --------------------------------------------------------------------------- #
# small compatibility / IO helpers
# --------------------------------------------------------------------------- #
_WARNED_UNSCALED = False


def get_traces_uv(rec, start_frame, end_frame, channel_ids=None):
    """
    rec.get_traces() in microvolts, across SpikeInterface versions.

    SI <=0.102 uses return_scaled=, >=0.103 uses return_in_uV=.
    """
    global _WARNED_UNSCALED
    kw = dict(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        channel_ids=channel_ids,
    )
    for key in ("return_in_uV", "return_scaled"):
        try:
            return rec.get_traces(**kw, **{key: True})
        except TypeError:
            continue
        except ValueError:
            break  # gains not set on this recording
    if not _WARNED_UNSCALED:
        warnings.warn("No uV gain on this recording; traces are in raw ADC units.")
        _WARNED_UNSCALED = True
    return rec.get_traces(**kw)


def load_recording(intan_folder=None, si_folder=None, stream_id="0",
                   freq_min=300.0, freq_max=6000.0, cmr=True, channel_ids=None):
    """Return (raw_recording, filtered_recording).

    Either point at the Intan session folder (containing info.rhd + amplifier.dat)
    or at a SpikeInterface binary folder that was actually fed to Kilosort.
    """
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
    """Sampling rate from Kilosort/Phy params.py (None if unavailable)."""
    p = Path(ks_dir) / "params.py"
    if not p.exists():
        return None
    m = re.search(r"sample_rate\s*=\s*([0-9.eE+-]+)", p.read_text())
    return float(m.group(1)) if m else None


def read_cluster_label(ks_dir, cluster_id):
    """Phy curation label for a cluster ('good', 'mua', ...), or '' if unknown."""
    ks_dir = Path(ks_dir)
    for fname, col in (("cluster_group.tsv", "group"),
                       ("cluster_KSLabel.tsv", "KSLabel")):
        f = ks_dir / fname
        if not f.exists():
            continue
        rows = [ln.split("\t") for ln in f.read_text().splitlines() if ln.strip()]
        header, rows = rows[0], rows[1:]
        try:
            ci, li = header.index("cluster_id"), header.index(col)
        except ValueError:
            continue
        for r in rows:
            if len(r) > max(ci, li) and int(r[ci]) == int(cluster_id):
                return r[li].strip()
    return ""


def load_cluster_spikes(ks_dir, cluster_id):
    """Spike sample indices for one cluster (int64, sorted)."""
    ks_dir = Path(ks_dir)
    st = np.load(ks_dir / "spike_times.npy").astype(np.int64).ravel()
    sc = np.load(ks_dir / "spike_clusters.npy").astype(np.int64).ravel()
    spk = np.sort(st[sc == int(cluster_id)])
    spk = spk[spk >= 0]
    if spk.size == 0:
        raise ValueError(f"No spikes found for cluster {cluster_id} in {ks_dir}")
    return spk


def max_site_from_struct(mat_path, cluster_id):
    """Optional: read max_site for a cluster out of waveformStruct.mat.

    Returns a 0-based row index into amplifier.dat (MATLAB max_site is 1-based).
    """
    from scipy.io import loadmat

    wv = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["wvStruct"]
    ids = np.atleast_1d(np.asarray(wv.goodIDs)).astype(int)
    sites = np.atleast_1d(np.asarray(wv.max_site)).astype(int)
    hits = np.flatnonzero(ids == int(cluster_id))
    if hits.size == 0:
        raise ValueError(f"Cluster {cluster_id} not in {mat_path}")
    return int(sites[hits[0]]) - 1


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract_snippets(rec, spike_samples, pre_samp, n_samp,
                     channel_ids=None, block_sec=30.0):
    """Cut waveform snippets around spikes.

    Traces are pulled in contiguous blocks rather than one call per spike, so
    the bandpass filter runs once per block instead of once per spike.

    Returns
    -------
    snips : (n_kept, n_samp, n_channels) float array, uV
    kept  : bool mask into spike_samples
    """
    n_total = rec.get_num_samples()
    post = n_samp - pre_samp
    kept = (spike_samples - pre_samp >= 0) & (spike_samples + post <= n_total)
    spk = np.sort(spike_samples[kept])
    if spk.size == 0:
        raise ValueError("No spikes far enough from the edges of the recording.")

    block = max(int(block_sec * rec.get_sampling_frequency()), n_samp * 4)
    snips, i = [], 0
    while i < spk.size:
        b0 = int(spk[i] - pre_samp)
        b1 = int(min(b0 + block, n_total))
        j = i
        while j < spk.size and spk[j] + post <= b1:
            j += 1
        if j == i:  # single spike wider than the block
            b1, j = int(min(spk[i] + post, n_total)), i + 1
        traces = get_traces_uv(rec, b0, b1, channel_ids)
        for s in spk[i:j]:
            k = int(s - pre_samp - b0)
            snips.append(traces[k:k + n_samp])
        i = j
    return np.asarray(snips, dtype=float), kept


def pick_max_channel(rec_filt, spike_samples, pre_samp, n_samp,
                     n_spikes=300, seed=0):
    """Channel with the largest peak-to-peak mean waveform (0-based index)."""
    rng = np.random.default_rng(seed)
    sub = spike_samples
    if sub.size > n_spikes:
        sub = np.sort(rng.choice(sub, n_spikes, replace=False))
    snips, _ = extract_snippets(rec_filt, sub, pre_samp, n_samp)
    mean_wf = snips.mean(axis=0)                       # (n_samp, n_channels)
    amps = mean_wf.max(axis=0) - mean_wf.min(axis=0)
    return int(np.argmax(amps)), mean_wf, amps


def pick_trace_window(rec_filt, spike_samples, channel_id, win_sec=4.0,
                      step_sec=0.25, mode="max", t_start=None,
                      artifact_uv=2000.0, n_candidates=25):
    """Choose a window of the recording in which the cell is firing well.

    mode='max'      : window with the most spikes
    mode='typical'  : window at the 75th percentile of spike count
    mode='time'     : start at t_start seconds (no search)

    Candidate windows are rejected if the trace saturates (|V| > artifact_uv),
    which keeps you off movement / chewing / stim artifacts.
    """
    fs = rec_filt.get_sampling_frequency()
    n_total = rec_filt.get_num_samples()
    win_samp = int(round(win_sec * fs))

    if mode == "time":
        if t_start is None:
            raise ValueError("mode='time' requires t_start")
        s0 = int(round(t_start * fs))
        return s0, get_traces_uv(rec_filt, s0, min(s0 + win_samp, n_total),
                                 [channel_id])[:, 0]

    step = max(int(round(step_sec * fs)), 1)
    nbins = max(n_total // step, 1)
    counts = np.bincount(np.clip(spike_samples // step, 0, nbins - 1),
                         minlength=nbins).astype(float)
    nwin = max(int(round(win_sec / step_sec)), 1)
    if counts.size < nwin:
        raise ValueError("Recording is shorter than the requested window.")
    win_counts = np.convolve(counts, np.ones(nwin), mode="valid")

    if mode == "max":
        order = np.argsort(win_counts)[::-1]
    elif mode == "typical":
        target = np.percentile(win_counts[win_counts > 0], 75)
        order = np.argsort(np.abs(win_counts - target))
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    for idx in order[:n_candidates]:
        s0 = int(idx * step)
        s1 = min(s0 + win_samp, n_total)
        trace = get_traces_uv(rec_filt, s0, s1, [channel_id])[:, 0]
        if np.max(np.abs(trace)) < artifact_uv:
            return s0, trace
    # nothing clean found; fall back to the best window anyway
    s0 = int(order[0] * step)
    warnings.warn("Every candidate window exceeded artifact_uv; using the best one.")
    return s0, get_traces_uv(rec_filt, s0, min(s0 + win_samp, n_total),
                             [channel_id])[:, 0]


def spike_width_ms(wf, fs):
    """Trough-to-following-peak width in ms (Payne et al. definition)."""
    trough = int(np.argmin(wf))
    if trough >= wf.size - 1:
        return np.nan
    peak = trough + int(np.argmax(wf[trough:]))
    return (peak - trough) / fs * 1e3


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _scalebar(ax, x, y, dx, dy, xlabel, ylabel, color="k", lw=1.5, fs=7):
    """L-shaped scale bar in data coordinates."""
    ax.plot([x, x], [y, y + dy], color=color, lw=lw, clip_on=False)
    ax.plot([x, x + dx], [y, y], color=color, lw=lw, clip_on=False)
    ax.text(x + dx / 2, y - abs(dy) * 0.08, xlabel, ha="center", va="top",
            fontsize=fs, color=color, clip_on=False)
    ax.text(x - dx * 0.06, y + dy / 2, ylabel, ha="right", va="center",
            fontsize=fs, color=color, clip_on=False)


def plot_fig1d(data, n_show=20, color=None, width_thresh_ms=0.4, figsize=(9, 2.6)):
    """Draw the Payne Fig. 1D-style panel from the dict returned by extract_cell_data."""
    fs = data["fs"]
    trace, t = data["trace"], data["trace_t"]
    spike_t = data["trace_spike_t"]
    wfs, mean_wf = data["waveforms"], data["mean_waveform"]
    wf_t = data["waveform_t"]

    if color is None:
        color = "#e64980" if data["spike_width_ms"] >= width_thresh_ms else "#3b82f6"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.08)
    ax_tr, ax_wf = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- left: voltage trace with this cell's spikes marked ----
    ax_tr.plot(t, trace, color="k", lw=0.4)
    # Robust limits: tall enough for a typical spike, but a couple of
    # overlapping spikes shouldn't squash everything else flat.
    lo = 1.3 * min(np.percentile(trace, 0.05), mean_wf.min())
    hi = 1.3 * max(np.percentile(trace, 99.95), mean_wf.max())
    span = hi - lo
    tick_y = hi + 0.08 * span
    ax_tr.plot(spike_t, np.full_like(spike_t, tick_y), marker="|", ls="none",
               color=color, ms=6, mew=1.2)
    ax_tr.set_xlim(t[0], t[-1])
    ax_tr.set_ylim(lo - 0.35 * span, tick_y + 0.1 * span)
    ax_tr.axis("off")
    _scalebar(ax_tr, t[0] + 0.02 * np.ptp(t), lo - 0.28 * span,
              0.5, 100, "0.5 s", "100 µV")

    # ---- right: individual waveforms + mean ----
    rng = np.random.default_rng(0)
    idx = rng.choice(wfs.shape[0], min(n_show, wfs.shape[0]), replace=False)
    ax_wf.plot(wf_t, wfs[idx].T, color="k", lw=0.4, alpha=0.35)
    ax_wf.plot(wf_t, mean_wf, color=color, lw=2)
    ax_wf.set_xlim(wf_t[0], wf_t[-1])
    ax_wf.axis("off")
    wspan = np.ptp(wfs[idx])
    ax_wf.set_ylim(np.min(wfs[idx]) - 0.3 * wspan, np.max(wfs[idx]) + 0.1 * wspan)
    _scalebar(ax_wf, wf_t[0] + 0.1, np.min(wfs[idx]) - 0.22 * wspan,
              1.0, 100, "1 ms", "100 µV")

    title = (f"cluster {data['cluster_id']}"
             f"{' (' + data['label'] + ')' if data['label'] else ''}   "
             f"ch {data['max_channel_name']}   "
             f"{data['mean_rate_hz']:.1f} Hz   "
             f"n = {data['n_spikes']} spikes   "
             f"width {data['spike_width_ms']:.2f} ms")
    fig.suptitle(title, fontsize=8, y=1.02, x=0.02, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def extract_cell_data(intan_folder=None, ks_dir=None, cluster_id=None,
                      si_folder=None, stream_id="0", channel_ids=None,
                      cmr=True, freq_min=300.0, freq_max=6000.0,
                      win_sec=4.0, window_mode="max", t_start=None,
                      artifact_uv=2000.0, n_wf=1000, max_channel=None,
                      wf_struct=None, seed=0):
    """Pull everything needed for the figure. Returns a dict of arrays."""
    rec, rec_filt = load_recording(intan_folder=intan_folder, si_folder=si_folder,
                                   stream_id=stream_id, freq_min=freq_min,
                                   freq_max=freq_max, cmr=cmr,
                                   channel_ids=channel_ids)
    fs = rec.get_sampling_frequency()
    n_total = rec.get_num_samples()

    ks_fs = read_params_fs(ks_dir)
    if ks_fs is not None and not np.isclose(ks_fs, fs, rtol=1e-4):
        warnings.warn(f"params.py sample rate ({ks_fs}) != recording ({fs}).")

    spk = load_cluster_spikes(ks_dir, cluster_id)
    if spk[-1] > n_total:
        raise ValueError(
            f"Cluster {cluster_id} has spikes past the end of this recording "
            f"({spk[-1]} > {n_total} samples). The sorted recording is probably "
            "not the one you just loaded — see the caveats in the docstring.")

    pre_samp = int(round(PRE_MS * 1e-3 * fs))
    n_samp = int(round(TOTAL_MS * 1e-3 * fs))

    # best channel
    if wf_struct is not None or max_channel is not None:
        if channel_ids is not None:
            warnings.warn("A channel subset is loaded, so max_site / max_channel "
                          "indices from the full probe will not line up.")
    if wf_struct is not None:
        max_ch_idx = max_site_from_struct(wf_struct, cluster_id)
        mean_all = None
    elif max_channel is not None:
        max_ch_idx = int(max_channel)
        mean_all = None
    else:
        max_ch_idx, mean_all, _ = pick_max_channel(rec_filt, spk, pre_samp,
                                                   n_samp, seed=seed)
    channel_id = rec.channel_ids[max_ch_idx]
    names = rec.get_property("channel_name")
    ch_name = str(names[max_ch_idx]) if names is not None else str(channel_id)

    # waveforms on that channel
    rng = np.random.default_rng(seed)
    sub = spk if spk.size <= n_wf else np.sort(rng.choice(spk, n_wf, replace=False))
    wfs, _ = extract_snippets(rec_filt, sub, pre_samp, n_samp,
                              channel_ids=[channel_id])
    wfs = wfs[:, :, 0]
    mean_wf = wfs.mean(axis=0)

    # example trace
    s0, trace = pick_trace_window(rec_filt, spk, channel_id, win_sec=win_sec,
                                  mode=window_mode, t_start=t_start,
                                  artifact_uv=artifact_uv)
    s1 = s0 + trace.size
    in_win = spk[(spk >= s0) & (spk < s1)]

    return dict(
        cluster_id=int(cluster_id),
        label=read_cluster_label(ks_dir, cluster_id),
        fs=fs,
        max_channel_index=max_ch_idx,
        max_channel_id=channel_id,
        max_channel_name=ch_name,
        n_spikes=int(spk.size),
        mean_rate_hz=float(spk.size / (n_total / fs)),
        spike_times_s=spk / fs,
        trace=trace,
        trace_t=np.arange(trace.size) / fs,
        trace_start_s=s0 / fs,
        trace_spike_t=(in_win - s0) / fs,
        waveforms=wfs,
        mean_waveform=mean_wf,
        waveform_t=(np.arange(n_samp) - pre_samp) / fs * 1e3,   # ms
        mean_waveform_all_channels=mean_all,
        spike_width_ms=spike_width_ms(mean_wf, fs),
        peak_to_peak_uv=float(np.ptp(mean_wf)),
    )


def make_cell_figure(intan_folder=None, ks_dir=None, cluster_id=None,
                     out_dir=None, n_show=20, save_npz=False, **kwargs):
    """extract_cell_data + plot_fig1d, optionally saving to out_dir."""
    data = extract_cell_data(intan_folder=intan_folder, ks_dir=ks_dir,
                             cluster_id=cluster_id, **kwargs)
    fig = plot_fig1d(data, n_show=n_show)
    print(f"cluster {data['cluster_id']} ({data['label'] or 'unlabeled'}): "
          f"{data['n_spikes']} spikes, {data['mean_rate_hz']:.2f} Hz, "
          f"best channel {data['max_channel_name']} (index {data['max_channel_index']}), "
          f"p-p {data['peak_to_peak_uv']:.0f} uV, width {data['spike_width_ms']:.2f} ms, "
          f"trace at t = {data['trace_start_s']:.2f} s")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"cell_{cluster_id}_fig1d"
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        if save_npz:
            np.savez_compressed(
                f"{stem}.npz",
                **{k: v for k, v in data.items() if v is not None})
        print(f"saved {stem}.png / .pdf")
    return fig, data


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intan-folder", help="folder with info.rhd + amplifier.dat")
    p.add_argument("--si-folder", help="SpikeInterface binary folder that was sorted")
    p.add_argument("--ks-dir", required=True, help="Kilosort/Phy output folder")
    p.add_argument("--cluster-id", type=int, required=True)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--stream-id", default="0")
    p.add_argument("--win-sec", type=float, default=4.0)
    p.add_argument("--window-mode", default="max", choices=["max", "typical", "time"])
    p.add_argument("--t-start", type=float, default=None,
                   help="window start in seconds (with --window-mode time)")
    p.add_argument("--n-wf", type=int, default=1000, help="spikes used for the mean")
    p.add_argument("--n-show", type=int, default=20, help="individual waveforms drawn")
    p.add_argument("--max-channel", type=int, default=None,
                   help="0-based channel index; default is found from the data")
    p.add_argument("--wf-struct", default=None,
                   help="waveformStruct.mat to read max_site from instead")
    p.add_argument("--channel-ids", nargs="+", default=None,
                   help="subset of channel ids to load (must match what was sorted)")
    p.add_argument("--no-cmr", action="store_true", help="skip common median reference")
    p.add_argument("--artifact-uv", type=float, default=2000.0)
    p.add_argument("--save-npz", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    make_cell_figure(
        intan_folder=args.intan_folder, si_folder=args.si_folder,
        ks_dir=args.ks_dir, cluster_id=args.cluster_id, out_dir=args.out_dir,
        stream_id=args.stream_id, win_sec=args.win_sec,
        window_mode=args.window_mode, t_start=args.t_start,
        n_wf=args.n_wf, n_show=args.n_show, max_channel=args.max_channel,
        wf_struct=args.wf_struct, cmr=not args.no_cmr,
        channel_ids=args.channel_ids,
        artifact_uv=args.artifact_uv, save_npz=args.save_npz,
    )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
