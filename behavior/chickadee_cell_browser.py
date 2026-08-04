#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chickadee_cell_browser.py
=========================

Interactive browser for single-cell responses during the chickadee caching task
(Chettih et al., 2024).

Given
    aligned_spikes.npy   (n_cells, n_video_frames)  spikes per video frame
    annotatedSeeds.mat   annotated behaviour struct
the GUI shows, for one cell at a time, rasters + tuning curves (PSTHs) aligned to

    caches | retrievals | eating bouts
    checks | site-perch visits | feeder-perch visits

Checks and visits are split into occupied vs. empty sites.
Feeder visits are split by feeder ID, or by open/closed if feeder open times
are supplied.

Shared tuning-curve y-axes:
    caches  <-> retrievals
    checks  <-> visits            (all four occupied/empty traces)
    eating bouts                  (own scale)
    feeder visits                 (own scale)

Usage
-----
    python chickadee_cell_browser.py

Optional command line pre-fill:
    python chickadee_cell_browser.py --spikes path/to/aligned_spikes.npy \
                                     --seeds  path/to/annotatedSeeds.mat \
                                     --bird LMN88 --session 260727
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms as mtransforms
from matplotlib.figure import Figure

# tkinter is only needed to actually run the GUI.  Importing it lazily keeps the
# analysis/plotting layer of this module usable in headless scripts and tests.
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except Exception:                                            # pragma: no cover
    _HAS_TK = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Perch numbers that correspond to feeder perches.  Perches whose index falls
# outside the range of cache sites (i.e. >= seedChanges.shape[1]) are also
# treated as feeder perches, so this list is only a fallback.
FEEDER_PERCHES = np.asarray([84, 85, 86, 87])

DEFAULT_FPS = 50.0

# Colour for muted tkinter label text.  Tk only accepts X11 colour names or
# #rrggbb hex, so matplotlib "xkcd:" names must never be used on widgets.
TK_MUTED_FG = "#555555"

# Postural keypoint indices for the top and bottom of the beak; the two are
# averaged together, and the z (vertical) component is column 2.
BEAK_KEYPOINTS = (0, 1)
Z_AXIS = 2

# Waveforms in waveformStruct.mat are assumed to already be in microvolts.
# If your pipeline stores raw ADC counts instead, set the conversion here
# (e.g. 0.195 for an Intan RHD headstage).
WAVEFORM_UV_PER_UNIT = 1.0

# Waveform inset: x/width are fractions, the vertical extent is in points so
# it keeps its size and its clearance whatever the canvas height turns out to
# be.  x0 matches GRID_LEFT so the inset lines up with the panels underneath.
WAVEFORM_X0, WAVEFORM_W = 0.075, 0.135
WAVEFORM_TITLE_TOP_PTS = 14      # from the top of the figure
WAVEFORM_H_PTS = 52

# Sampling rate of the raw ephys, used only to sanity-check how many samples a
# stored waveform should have (spkDur seconds x this rate).
EXPECTED_SAMPLE_RATE = 30000.0
WAVEFORM_LW = 1.7
# Displayed slice of the waveform, in ms relative to the aligned peak
# (spkOffset).  Most units settle within ~0.3 ms of the peak, so a 1.5 ms
# window is enough to show the repolarisation without a long noisy tail.
WAVEFORM_WINDOW_MS = (-0.5, 1.0)
# Candidate voltage scale-bar heights (uV); the smallest sensible one is used
VOLTAGE_BAR_STEPS = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]
TIME_BAR_STEPS = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0]      # ms

# Panel order on the figure: (key, display label).  Laid out 3 across, 2 down.
PANEL_ORDER: List[Tuple[str, str]] = [
    ("cache",     "caches"),
    ("retrieval", "retrievals"),
    ("eat",       "eating bouts"),
    ("check",     "checks"),
    ("visit",     "site perch visits"),
    ("feeder",    "feeder perch visits"),
]

# Which panels share a tuning-curve y-axis.
Y_SHARE_GROUPS: List[List[str]] = [
    ["cache", "retrieval"],
    ["check", "visit"],
    ["eat"],
    ["feeder"],
]

# Default time windows (seconds), taken from the existing analysis scripts.
DEFAULT_WINDOWS: Dict[str, Dict[str, float]] = {
    "cache":     {"raster": 10.0, "psth": 1.5},
    "retrieval": {"raster": 10.0, "psth": 1.5},
    "eat":       {"raster": 20.0, "psth": 3.0},
    "check":     {"raster":  2.0, "psth": 1.0},
    "visit":     {"raster":  8.0, "psth": 1.0},
    "feeder":    {"raster": 20.0, "psth": 2.0},
}

# Default event boundary each panel is aligned to ('onset' or 'offset').
DEFAULT_ALIGNMENT: Dict[str, str] = {
    "cache":     "offset",
    "retrieval": "offset",
    "eat":       "onset",
    "check":     "onset",
    "visit":     "onset",
    "feeder":    "offset",
}

# Colours (cache / retrieval / check follow Chettih et al. 2024)
COL_CACHE = "xkcd:orange"
COL_RET = "xkcd:purple"
COL_EAT = "xkcd:cerulean"
COL_CHECK_OCC = "xkcd:forest green"
COL_CHECK_EMPT = "xkcd:light green"
COL_VISIT_OCC = "xkcd:charcoal"
COL_VISIT_EMPT = "xkcd:grey"
FEEDER_COLORS = ["xkcd:saffron", "xkcd:green", "xkcd:scarlet", "xkcd:blue"]

# Panel geometry.  Heights are in points rather than gridspec ratios so that
# labels keep their clearance whatever size the Tk canvas ends up being.
GRID_LEFT, GRID_RIGHT = 0.075, 0.985
TOP_BAND_PTS = 112         # header + waveform inset + row-0 panel titles
BOTTOM_PTS = 40            # bottom row's tuning-curve ticks + x label
COL_GAP_FRAC = 0.060
# Preferred heights, and the smallest each may shrink to on a short window.
# When space is tight all three shrink together toward their minimums rather
# than one of them absorbing the whole squeeze.
PANEL_BEAK_PTS, PANEL_BEAK_MIN_PTS = 68, 26        # beak trajectory strip
PANEL_RASTER_PTS, PANEL_RASTER_MIN_PTS = 110, 45   # raster
PANEL_PSTH_PTS, PANEL_PSTH_MIN_PTS = 74, 34        # tuning curve
RASTER_XLABEL_PTS = 38     # under the raster: tick labels + x label (fixed)
ROW_GAP_PTS = 48           # between the two rows of panels

# Rotation for the occupied/empty group labels on the raster y axis
GROUP_LABEL_ROTATION = 90

# Figure cosmetics
FIG_SIZE = (14.0, 10.5)          # inches; taller so labels are not cramped
TITLE_SIZE = 15
PANEL_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 10.5
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 8.5
WAVEFORM_TITLE_SIZE = 11
SCALEBAR_LABEL_SIZE = 8.5
PSTH_LW = 2.0
EVENT_LW = 1.0

# beak-trajectory panel
BEAK_TRIAL_LW = 0.25
BEAK_TRIAL_ALPHA = 0.18
BEAK_MEAN_LW = 1.0
BEAK_MAX_TRIALS = 300        # cap on individual traces drawn, for speed
# colours for group-split mean trajectories, indexed by group label
BEAK_GROUP_COLORS = ["xkcd:grey", "k"]      # empty, occupied

# upper bound on raster tick area (points^2); large enough that panels with
# only a handful of events still draw gap-free rows
RASTER_MAX_TICK_AREA = 10000.0


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB struct loading
# ─────────────────────────────────────────────────────────────────────────────

def _mat_to_dict(obj):
    """Recursively convert scipy mat_struct objects into nested dicts."""
    if isinstance(obj, sio.matlab.mat_struct):
        return {name: _mat_to_dict(getattr(obj, name))
                for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # Allocate the result and fill it element-wise. Building it with
        # np.array([...], dtype=object) lets numpy infer dimensions, which
        # silently collapses a cell array of equal-length entries into a 2-D
        # array -- for all-empty cells that gives shape (n, 0), and the
        # subsequent reshape fails with "cannot reshape array of size 0".
        out = np.empty(obj.shape, dtype=object)
        flat_out, flat_in = out.reshape(-1), obj.reshape(-1)
        for i in range(flat_in.size):
            flat_out[i] = _mat_to_dict(flat_in[i])
        return out
    return obj


def loadmat_struct(path: str) -> dict:
    """
    Load a .mat file into nested python dicts (equivalent to loadmat_sbx).

    Falls back to mat73 for MATLAB v7.3 (HDF5) files if that package is
    installed.
    """
    try:
        raw = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError as exc:                            # pragma: no cover
            raise RuntimeError(
                f"{os.path.basename(path)} is a MATLAB v7.3 (HDF5) file. "
                "Install the 'mat73' package to read it, or re-save the file "
                "in MATLAB with '-v7'."
            ) from exc
        return mat73.loadmat(path)

    return {k: _mat_to_dict(v) for k, v in raw.items()
            if not k.startswith("__")}


def _scalar(d: dict, key: str) -> float:
    """First element of a field, as a float (0.0 when absent)."""
    if key not in d:
        return 0.0
    v = np.atleast_1d(np.asarray(d[key], dtype=float)).ravel()
    return float(v[0]) if v.size else 0.0


def _first_len(d: dict, keys: Sequence[str]) -> Optional[int]:
    """Length of the first per-unit field present, i.e. the number of units."""
    for k in keys:
        if k in d:
            v = np.atleast_1d(np.asarray(d[k])).ravel()
            if v.size:
                return int(v.size)
    return None


def _orient_waveforms_3d(a: np.ndarray, n_units: int,
                         n_samp: Optional[int], max_site: int
                         ) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Reorder waveFormsMean to (n_cells, n_channels, n_timepoints).

    MATLAB stores it as (n_timepoints, n_channels, n_cells), so the usual
    answer is transpose(2, 1, 0) -- the same thing the lab's
    get_waveform_params() does.  The axes are resolved explicitly rather than
    assumed, because a silently transposed array produces a plot that looks
    like noise instead of raising anything.
    """
    shp = a.shape
    unit_axes = [i for i in range(3) if shp[i] == n_units]
    if not unit_axes:
        return None, (f"no axis has length n_units={n_units}")
    # MATLAB puts cells last, so prefer axis 2 when the length is ambiguous
    unit_ax = 2 if 2 in unit_axes else unit_axes[0]

    rest = [i for i in range(3) if i != unit_ax]
    samp_ax = None
    if n_samp:
        hit = [i for i in rest if shp[i] == n_samp]
        if hit:
            samp_ax = hit[0]
    if samp_ax is None:
        # the channel axis has to be able to hold max_site
        ok = [i for i in rest if shp[i] >= max_site]
        if len(ok) == 1:
            samp_ax = [i for i in rest if i != ok[0]][0]
        else:
            # fall back to MATLAB's order: timepoints first
            samp_ax = rest[0]
    chan_ax = [i for i in rest if i != samp_ax][0]

    if shp[chan_ax] < max_site:
        return None, (f"channel axis of length {shp[chan_ax]} cannot hold "
                      f"max_site={max_site}")
    return np.transpose(a, (unit_ax, chan_ax, samp_ax)), None


def _orient_mxwf(m: np.ndarray, n_units: Optional[int],
                 n_samp: Optional[int]) -> Tuple[np.ndarray, bool]:
    """Return mxWF as (n_units, n_samples), flipping it if it arrived rotated."""
    if n_units:
        if m.shape[0] == n_units and m.shape[1] != n_units:
            return m, False
        if m.shape[1] == n_units and m.shape[0] != n_units:
            return m.T, True
    if n_samp:
        if m.shape[1] == n_samp:
            return m, False
        if m.shape[0] == n_samp:
            return m.T, True
    return m, False


def inspect_waveform_file(path: str) -> str:
    """Human-readable dump of a waveformStruct.mat, for debugging layouts."""
    mat = loadmat_struct(path)
    out = [f"{path}", ""]
    for name, v in mat.items():
        if not isinstance(v, dict):
            continue
        out.append(f"struct '{name}':")
        for f, val in v.items():
            arr = np.asarray(val)
            out.append(f"    {f:16s} dtype={str(arr.dtype):10s} shape={arr.shape}")
        wv = v
        if "mxWF" not in wv and "waveFormsMean" not in wv:
            continue
        n_units = _first_len(wv, ("goodIDs", "meanRate", "max_site", "nSpikes"))
        dur, off = _scalar(wv, "spkDur"), _scalar(wv, "spkOffset")
        hint = int(round(dur * EXPECTED_SAMPLE_RATE)) if dur else None
        out += ["",
                f"    inferred n_units      : {n_units}",
                f"    spkDur / spkOffset    : {dur} s / {off} s",
                f"    expected n_samples    : {hint} "
                f"(at {EXPECTED_SAMPLE_RATE:g} Hz)"]
        sites = np.atleast_1d(np.asarray(wv.get("max_site", []))).ravel()
        if sites.size:
            out.append(f"    max_site range        : {sites.min()}-{sites.max()}")
        if "waveFormsMean" in wv and sites.size and n_units:
            a = np.asarray(wv["waveFormsMean"], dtype=float)
            if a.ndim == 3:
                arr, err = _orient_waveforms_3d(a, n_units, hint,
                                                int(sites.astype(int).max()))
                if err:
                    out.append(f"    waveFormsMean         : UNRESOLVED ({err})")
                else:
                    out.append(f"    waveFormsMean {a.shape} -> "
                               f"(cells, channels, samples) {arr.shape}")
                    idx = np.clip(sites.astype(int) - 1, 0, arr.shape[1] - 1)
                    wf = np.stack([arr[u, idx[u]] for u in range(n_units)])
                    out += _peak_report(wf, dur, off, "    from waveFormsMean")
        if "mxWF" in wv:
            m = np.atleast_2d(np.asarray(wv["mxWF"], dtype=float))
            wf, flip = _orient_mxwf(m, n_units, hint)
            out.append(f"    mxWF {m.shape} -> (units, samples) {wf.shape}"
                       f"{'  [transposed]' if flip else ''}")
            out += _peak_report(wf, dur, off, "    from mxWF")
        out.append("")
    return "\n".join(out)


def _peak_report(wf: np.ndarray, dur: float, off: float, tag: str) -> List[str]:
    n_samp = wf.shape[1]
    lines = [f"{tag}: amplitude p2p median "
             f"{np.median(wf.max(axis=1) - wf.min(axis=1)):.1f}"]
    if dur > 0 and off:
        expect = int(round(off / dur * n_samp))
        peaks = np.argmax(np.abs(wf), axis=1)
        tol = max(3, int(0.03 * n_samp))
        frac = float(np.mean(np.abs(peaks - expect) <= tol))
        verdict = "LOOKS CORRECT" if frac >= 0.5 else "LOOKS WRONG"
        lines.append(f"{tag}: {frac * 100:.0f}% peak at sample {expect}"
                     f"/{n_samp}  -> {verdict}")
    return lines


def _int_arr(d: dict, key: str) -> np.ndarray:
    """Pull a field out of a struct-dict as a 1-D int array."""
    if key not in d:
        return np.zeros(0, dtype=int)
    v = np.atleast_1d(np.asarray(d[key]))
    if v.size == 0:
        return np.zeros(0, dtype=int)
    return np.round(v.astype(float)).astype(int).ravel()


def _2d_arr(d: dict, key: str) -> np.ndarray:
    """Pull a field out of a struct-dict as a 2-D float array."""
    if key not in d:
        return np.zeros((0, 0))
    return np.atleast_2d(np.asarray(d[key], dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
# Event containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventSet:
    """A set of behavioural events of one type, optionally split into groups."""
    key: str
    label: str
    onsets: np.ndarray                       # (n_events,) frame index
    offsets: np.ndarray                      # (n_events,) frame index
    groups: np.ndarray                       # (n_events,) int group label
    group_names: List[str] = field(default_factory=list)
    group_colors: List[str] = field(default_factory=list)
    # draw a separate mean beak trajectory per group (used for checks)
    split_beak_mean: bool = False
    # feeder groups are already named on the raster y axis, so no legend
    psth_legend: bool = True
    # name the groups alongside the raster y axis (vs. no group labels)
    label_groups_on_yaxis: bool = True
    # subsample the raster when a group has very many events (checks/visits)
    cap_raster: bool = False

    @property
    def n(self) -> int:
        return int(self.onsets.shape[0])

    @property
    def durations(self) -> np.ndarray:
        return self.offsets - self.onsets

    def align_frames(self, alignment: str) -> np.ndarray:
        return self.onsets if alignment == "onset" else self.offsets


# ─────────────────────────────────────────────────────────────────────────────
# Session data
# ─────────────────────────────────────────────────────────────────────────────

class SessionData:
    """
    Loads aligned_spikes.npy + annotatedSeeds.mat and derives every event set
    the browser can plot.
    """

    def __init__(self,
                 spikes_path: str,
                 seeds_path: str,
                 bird: str = "",
                 session_id: str = "",
                 fps: float = DEFAULT_FPS,
                 feeder_open_times: Optional[Sequence[float]] = None,
                 feeder_close_times: Optional[Sequence[float]] = None,
                 posture_path: str = "",
                 waveform_path: str = ""):

        self.bird = bird
        self.session_id = session_id
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.notes: List[str] = []

        # ── neural data ──────────────────────────────────────────────────
        spike_fr = np.load(spikes_path)
        if spike_fr.ndim != 2:
            raise ValueError(
                f"aligned_spikes.npy should be 2-D (n_cells, n_frames); "
                f"got shape {spike_fr.shape}."
            )
        self.spike_fr = spike_fr.astype(np.float32)
        self.spike_bool = spike_fr.astype(bool)
        self.n_cells, self.n_frames = self.spike_fr.shape

        # session-average firing rate per cell, used as the PSTH baseline
        self.mean_fr = self.spike_fr.mean(axis=1) / self.dt

        # Cell IDs are Kilosort/phy cluster IDs.  Row i of aligned_spikes is
        # the i-th *good* cluster, so the row index is NOT the cluster ID --
        # goodIDs from waveformStruct.mat provides the mapping.  Without that
        # file we can only fall back to row numbers.
        self.cluster_ids = np.arange(self.n_cells)
        self.ids_are_clusters = False

        # ── behaviour data ───────────────────────────────────────────────
        mat = loadmat_struct(seeds_path)
        if "annotatedSeeds" not in mat:
            raise ValueError(
                f"{os.path.basename(seeds_path)} does not contain an "
                "'annotatedSeeds' struct."
            )
        self.seed_struct = mat["annotatedSeeds"]
        self.count_data = self.seed_struct["countData"]

        self.seed_changes = _2d_arr(self.seed_struct, "seedChanges")
        self.init_counts = np.atleast_1d(
            np.asarray(self.seed_struct.get("initSeedCounts", []), dtype=float)
        ).ravel()
        self.n_sites = int(self.seed_changes.shape[1])

        # feeder open/close periods (minutes -> frames)
        self.feeder_open_frames = (
            np.asarray(feeder_open_times, dtype=float) * 60.0 * self.fps
            if feeder_open_times is not None and len(feeder_open_times) else None
        )
        self.feeder_close_frames = (
            np.asarray(feeder_close_times, dtype=float) * 60.0 * self.fps
            if feeder_close_times is not None and len(feeder_close_times) else None
        )

        # ── posture (optional) ───────────────────────────────────────────
        self.beak_z: Optional[np.ndarray] = None
        if posture_path:
            self._load_beak_z(posture_path)

        # ── waveforms (optional) ─────────────────────────────────────────
        self.wf_mean: Optional[np.ndarray] = None      # (n_units, n_samples)
        self.wf_time_ms: Optional[np.ndarray] = None
        self.wf_cluster_ids: Optional[np.ndarray] = None
        self.wf_mean_rate: Optional[np.ndarray] = None
        if waveform_path:
            self._load_waveforms(waveform_path)
            self._adopt_cluster_ids()
        else:
            self.notes.append(
                "No waveformStruct.mat loaded, so cell IDs are row numbers in "
                "aligned_spikes.npy, not phy cluster IDs. Load the waveform "
                "file to index cells by their phy cluster ID.")

        self._parse_behavior()
        self.events: Dict[str, EventSet] = {}
        self._build_event_sets()

    def _load_beak_z(self, posture_path: str) -> None:
        """Vertical position of the beak, averaged over the top and bottom."""
        posture = np.load(posture_path)           # frames x keypoints x xyz
        if posture.ndim != 3 or posture.shape[2] < 3:
            raise ValueError(
                f"{os.path.basename(posture_path)} should be "
                f"(n_frames, n_keypoints, 3); got shape {posture.shape}.")
        need = max(BEAK_KEYPOINTS) + 1
        if posture.shape[1] < need:
            raise ValueError(
                f"{os.path.basename(posture_path)} has only "
                f"{posture.shape[1]} keypoints; beak indices "
                f"{BEAK_KEYPOINTS} require at least {need}.")

        beak = posture[:, list(BEAK_KEYPOINTS), Z_AXIS].astype(float)
        with np.errstate(invalid="ignore"):
            z = np.nanmean(beak, axis=1)

        # keep the trace the same length as the spike array
        if z.shape[0] != self.n_frames:
            self.notes.append(
                f"posture file has {z.shape[0]} frames but aligned_spikes has "
                f"{self.n_frames}; the beak trace was padded/truncated to match.")
            fixed = np.full(self.n_frames, np.nan)
            m = min(z.shape[0], self.n_frames)
            fixed[:m] = z[:m]
            z = fixed
        self.beak_z = z

    # ── behaviour parsing ────────────────────────────────────────────────


    def _load_waveforms(self, waveform_path: str) -> None:
        """
        Mean waveform per unit from waveformStruct.mat.

        MATLAB writes `waveFormsMean` as (n_timepoints, n_channels, n_cells)
        and scipy preserves that order, which is why the lab's own
        get_waveform_params() does np.transpose(..., (2, 1, 0)) to reach
        (n_cells, n_channels, n_timepoints).  We follow the same path and take
        each unit's peak channel via the 1-indexed `max_site`.

        `mxWF` is only used as a fallback, because its orientation varies
        between pipeline versions.  Whenever both are present they are
        cross-checked against each other.
        """
        mat = loadmat_struct(waveform_path)

        wv = None
        for v in mat.values():
            if isinstance(v, dict) and ("mxWF" in v or "waveFormsMean" in v):
                wv = v
                break
        if wv is None:
            raise ValueError(
                f"{os.path.basename(waveform_path)} contains no struct with "
                "'mxWF' or 'waveFormsMean'.")
        self._wv = wv

        # ── how many units, and how long should a waveform be? ───────────
        n_units = _first_len(wv, ("goodIDs", "meanRate", "max_site", "nSpikes"))
        dur = _scalar(wv, "spkDur")
        off = _scalar(wv, "spkOffset")
        n_samp_hint = int(round(dur * EXPECTED_SAMPLE_RATE)) if dur else None

        sites = np.atleast_1d(np.asarray(wv.get("max_site", []))).ravel()
        sites = sites.astype(int) if sites.size else None
        if n_units is None and sites is not None:
            n_units = sites.size

        wf_from_all, wf_from_mx = None, None

        # ── preferred: waveFormsMean at each unit's peak channel ─────────
        if "waveFormsMean" in wv and sites is not None and n_units:
            a = np.asarray(wv["waveFormsMean"], dtype=float)
            if a.ndim == 3:
                arr, err = _orient_waveforms_3d(a, n_units, n_samp_hint,
                                                int(sites.max()))
                if err:
                    self.notes.append(f"waveFormsMean {a.shape}: {err}")
                else:
                    idx = np.clip(sites - 1, 0, arr.shape[1] - 1)
                    wf_from_all = np.stack([arr[u, idx[u]]
                                            for u in range(n_units)])

        # ── fallback / cross-check: mxWF ─────────────────────────────────
        if "mxWF" in wv:
            m = np.atleast_2d(np.asarray(wv["mxWF"], dtype=float))
            wf_from_mx, flipped = _orient_mxwf(m, n_units, n_samp_hint)
            if flipped:
                self.notes.append(
                    f"mxWF was stored as {m.shape} (samples x units) and has "
                    "been transposed to units x samples.")

        wf = wf_from_all if wf_from_all is not None else wf_from_mx
        if wf is None:
            raise ValueError(
                "Could not work out the layout of the waveform arrays in "
                f"{os.path.basename(waveform_path)}. Run\n"
                f"    python {os.path.basename(__file__)} "
                "--inspect-waveforms <file>\nand send the output.")

        # disagreement means one of the two was read with the wrong layout
        if wf_from_all is not None and wf_from_mx is not None:
            if wf_from_all.shape == wf_from_mx.shape:
                num = np.linalg.norm(wf_from_all - wf_from_mx)
                den = np.linalg.norm(wf_from_all) + 1e-12
                if num / den > 0.05:
                    self.notes.append(
                        "mxWF and waveFormsMean disagree; using waveFormsMean "
                        "(n_timepoints x n_channels x n_cells), which matches "
                        "the lab's get_waveform_params().")
            else:
                self.notes.append(
                    f"mxWF {wf_from_mx.shape} and the waveform extracted from "
                    f"waveFormsMean {wf_from_all.shape} have different shapes; "
                    "using waveFormsMean.")

        wf = wf * WAVEFORM_UV_PER_UNIT
        n_units, n_samp = wf.shape

        # ── time base ────────────────────────────────────────────────────
        if dur and dur > 0:
            self.wf_time_ms = (np.arange(n_samp) / (n_samp / dur) - off) * 1000.0
        else:
            self.notes.append("waveformStruct has no spkDur; the waveform "
                              "time scale bar is in samples, not ms.")
            self.wf_time_ms = np.arange(n_samp) - n_samp // 2

        self.wf_mean = wf
        if "goodIDs" in wv:
            self.wf_cluster_ids = np.atleast_1d(
                np.asarray(wv["goodIDs"])).astype(int).ravel()
        if "meanRate" in wv:
            self.wf_mean_rate = np.atleast_1d(
                np.asarray(wv["meanRate"], dtype=float)).ravel()

        # ── does this actually look like a set of spike waveforms? ───────
        if n_samp_hint and abs(n_samp - n_samp_hint) > 2:
            self.notes.append(
                f"Waveforms have {n_samp} samples but spkDur implies about "
                f"{n_samp_hint} at {EXPECTED_SAMPLE_RATE:g} Hz. The sample "
                "axis may have been picked up wrongly.")
        if dur and dur > 0 and off:
            expect = int(round(off / dur * n_samp))
            peaks = np.argmax(np.abs(wf), axis=1)
            tol = max(3, int(0.03 * n_samp))
            frac = float(np.mean(np.abs(peaks - expect) <= tol))
            if frac < 0.5:
                self.notes.append(
                    f"Only {frac * 100:.0f}% of waveforms peak at spkOffset "
                    f"(sample {expect} of {n_samp}). They are probably being "
                    "read with the wrong array layout - run "
                    "--inspect-waveforms on the file and send the output.")

        if n_units != self.n_cells:
            self.notes.append(
                f"waveformStruct has {n_units} units but aligned_spikes has "
                f"{self.n_cells} cells. Waveforms are matched by row order, so "
                "cells beyond the shorter list will have no waveform shown.")

    def _adopt_cluster_ids(self) -> None:
        """Use goodIDs as the cell IDs, and sanity-check the row alignment."""
        if self.wf_cluster_ids is None:
            return
        if self.wf_cluster_ids.size != self.n_cells:
            self.notes.append(
                f"waveformStruct lists {self.wf_cluster_ids.size} cluster IDs "
                f"but aligned_spikes has {self.n_cells} rows, so cells are "
                "still addressed by row number. The two files probably come "
                "from different sorts.")
            return

        self.cluster_ids = self.wf_cluster_ids.astype(int)
        self.ids_are_clusters = True

        # Cross-check: the mean rate stored per cluster should track the rate
        # computed from that row of aligned_spikes.  If it does not, the rows
        # and the cluster list are out of order and every plot would be
        # showing the wrong cell.
        if self.wf_mean_rate is not None and self.wf_mean_rate.size == self.n_cells:
            a = np.asarray(self.wf_mean_rate, dtype=float)
            b = self.mean_fr
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            if ok.sum() >= 3:
                ratio = b[ok] / a[ok]
                frac_off = float(np.mean((ratio < 0.4) | (ratio > 2.5)))
                if frac_off > 0.5:
                    self.notes.append(
                        f"Firing rates from aligned_spikes disagree with "
                        f"waveformStruct.meanRate for {frac_off * 100:.0f}% of "
                        "cells. The spike rows and the cluster list may be in "
                        "different orders - check that both files come from "
                        "the same sort before trusting the cell IDs.")

    def row_of(self, cell_id: int) -> Optional[int]:
        """Row of aligned_spikes for a cell ID (a phy cluster ID)."""
        hit = np.where(self.cluster_ids == int(cell_id))[0]
        return int(hit[0]) if hit.size else None

    def cluster_of(self, row: int) -> int:
        return int(self.cluster_ids[row])

    def waveform(self, row: int) -> Optional[np.ndarray]:
        """Mean waveform (uV) for one row of aligned_spikes, or None."""
        if self.wf_mean is None or not (0 <= row < self.wf_mean.shape[0]):
            return None
        return self.wf_mean[row]

    def _parse_behavior(self) -> None:
        cd = self.count_data

        # site interactions
        self.int_start = _int_arr(cd, "newSite")
        self.int_end = _int_arr(cd, "endSite")
        self.site_num = _int_arr(cd, "siteNum")
        self.n_interactions = self.int_start.shape[0]

        if self.seed_changes.size:
            self.int_changes = np.sum(self.seed_changes, axis=1)
        else:
            self.int_changes = np.zeros(self.n_interactions)

        # perch interactions
        self.perch_start = _int_arr(cd, "newPerch")
        self.perch_end = _int_arr(cd, "endPerch")
        self.perch_num = _int_arr(cd, "perchNum")
        self.n_perches = self.perch_start.shape[0]

        # eating bouts (beak at a perch)
        self.eat_start = _int_arr(cd, "newBeakPerch")
        self.eat_end = _int_arr(cd, "endBeakPerch")

        # feeder interactions (beak near a feeder) - used only for reference
        self.feeder_beak_start = _int_arr(cd, "newFeeder")
        self.feeder_beak_end = _int_arr(cd, "endFeeder")
        self.feeder_beak_num = _int_arr(cd, "feederNum")

        # which perches are feeder perches
        self.is_feeder_perch = (
            np.isin(self.perch_num, FEEDER_PERCHES) |
            (self.perch_num >= self.n_sites)
        )

        # visits: perch bouts containing no eating bout and no site interaction
        non_visit_start = np.sort(np.concatenate([self.eat_start,
                                                  self.int_start]))
        if non_visit_start.size:
            lo = np.searchsorted(non_visit_start, self.perch_start, side="right")
            hi = np.searchsorted(non_visit_start, self.perch_end, side="left")
            self.is_visit = ~(hi > lo)
        else:
            self.is_visit = np.ones(self.n_perches, dtype=bool)

        self._classify_occupancy()

    def _classify_occupancy(self) -> None:
        """Occupied vs. empty sites for checks and for site-perch visits."""
        # ---- checks: initial seed counts are known -----------------------
        self.is_check = self.int_changes == 0
        if self.seed_changes.size and self.init_counts.size == self.n_sites:
            seeds_in_sites = np.cumsum(self.seed_changes, axis=0) + self.init_counts
        elif self.seed_changes.size:
            seeds_in_sites = np.cumsum(self.seed_changes, axis=0)
            self.notes.append(
                "initSeedCounts missing/mismatched - check occupancy uses "
                "cumulative seed changes only."
            )
        else:
            seeds_in_sites = np.zeros((self.n_interactions, max(self.n_sites, 1)))

        site_idx = self.site_num - 1                     # siteNum is 1-indexed
        self.check_occupied = np.zeros(int(np.sum(self.is_check)), dtype=bool)
        c = -1
        for n_int in range(self.n_interactions):
            if not self.is_check[n_int]:
                continue
            c += 1
            s = site_idx[n_int] if n_int < site_idx.shape[0] else -1
            if 0 <= s < seeds_in_sites.shape[1]:
                self.check_occupied[c] = seeds_in_sites[n_int, s] > 0

        # ---- visits: initial seed counts are unknown ---------------------
        # NOTE: this differs slightly from summary_fig.py, which advanced the
        # interaction pointer by at most one per visit.  Here the most recent
        # completed interaction is found directly, and a site counts as
        # occupied only when the cumulative change is strictly positive.
        if self.seed_changes.size:
            seeds_visits = np.cumsum(self.seed_changes, axis=0)
        else:
            seeds_visits = np.zeros((self.n_interactions, max(self.n_sites, 1)))

        self.perch_occupied = np.zeros(self.n_perches, dtype=bool)
        if self.n_interactions:
            order = np.argsort(self.int_start)
            int_start_sorted = self.int_start[order]
            for i, ps in enumerate(self.perch_start):
                if not self.is_visit[i] or self.is_feeder_perch[i]:
                    continue
                site = self.perch_num[i]                 # perchNum is 0-indexed
                if not (0 <= site < seeds_visits.shape[1]):
                    continue
                k = int(np.searchsorted(int_start_sorted, ps, side="right")) - 1
                if k < 0:
                    continue                             # nothing known yet
                self.perch_occupied[i] = seeds_visits[order[k], site] > 0

    # ── event set construction ───────────────────────────────────────────

    def _build_event_sets(self) -> None:
        self.events["cache"] = self._simple_set(
            "cache", "caches",
            self.int_start[self.int_changes > 0],
            self.int_end[self.int_changes > 0],
            COL_CACHE)

        self.events["retrieval"] = self._simple_set(
            "retrieval", "retrievals",
            self.int_start[self.int_changes < 0],
            self.int_end[self.int_changes < 0],
            COL_RET)

        self.events["eat"] = self._simple_set(
            "eat", "eating bouts",
            self.eat_start, self.eat_end, COL_EAT)

        # checks, split occupied / empty
        chk_on = self.int_start[self.is_check]
        chk_off = self.int_end[self.is_check]
        occ = self.check_occupied
        self.events["check"] = EventSet(
            key="check", label="checks",
            onsets=chk_on, offsets=chk_off,
            groups=occ.astype(int),
            group_names=["empty", "occupied"],
            group_colors=[COL_CHECK_EMPT, COL_CHECK_OCC],
            split_beak_mean=True, cap_raster=True)

        # site-perch visits, split occupied / empty
        vis_mask = self.is_visit & ~self.is_feeder_perch
        self.events["visit"] = EventSet(
            key="visit", label="site perch visits",
            onsets=self.perch_start[vis_mask],
            offsets=self.perch_end[vis_mask],
            groups=self.perch_occupied[vis_mask].astype(int),
            group_names=["empty", "occupied"],
            group_colors=[COL_VISIT_EMPT, COL_VISIT_OCC], cap_raster=True)

        self._build_feeder_set(status="open")

    def _simple_set(self, key, label, onsets, offsets, color) -> EventSet:
        onsets = np.asarray(onsets, dtype=int)
        offsets = np.asarray(offsets, dtype=int)
        return EventSet(key=key, label=label, onsets=onsets, offsets=offsets,
                        groups=np.zeros(onsets.shape[0], dtype=int),
                        group_names=[""], group_colors=[color])

    def _build_feeder_set(self, status: str = "open") -> str:
        """
        Feeder-perch visits, always split by feeder identity.

        `status` selects which subset to show: 'open' or 'closed'.  Visits
        during which the feeder opened or closed part-way through (partial
        visits) are always excluded.  When no feeder open times were supplied
        the status cannot be determined, so every visit is shown and 'all'
        is returned.
        """
        mask = self.is_feeder_perch
        onsets = self.perch_start[mask]
        offsets = self.perch_end[mask]
        pnum = self.perch_num[mask]

        # feeder ID 0..3, in the order given by FEEDER_PERCHES where possible
        feeder_id = np.zeros(pnum.shape[0], dtype=int)
        for i, p in enumerate(pnum):
            hit = np.where(FEEDER_PERCHES == p)[0]
            feeder_id[i] = hit[0] if hit.size else max(p - self.n_sites, 0)
        feeder_id = np.clip(feeder_id, 0, len(FEEDER_COLORS) - 1)

        st = self._feeder_status(onsets, offsets)
        if st is None:
            keep = np.ones(onsets.shape[0], dtype=bool)
            applied = "all"
            self.n_feeder_partial = 0
        else:
            self.n_feeder_partial = int(np.sum(st == 0.5))
            want = 1.0 if status == "open" else 0.0
            keep = st == want            # partial visits (0.5) dropped here
            applied = "open" if status == "open" else "closed"

        suffix = {"open": ", open", "closed": ", closed", "all": ""}[applied]

        self.feeder_status_applied = applied
        self.events["feeder"] = EventSet(
            key="feeder", label="feeder perch visits" + suffix,
            onsets=onsets[keep], offsets=offsets[keep],
            groups=feeder_id[keep],
            group_names=[f"feeder {i + 1}" for i in range(len(FEEDER_COLORS))],
            group_colors=list(FEEDER_COLORS),
            psth_legend=False, label_groups_on_yaxis=False)
        return applied

    def _feeder_status(self, onsets, offsets) -> Optional[np.ndarray]:
        """0 = closed throughout, 1 = open throughout, 0.5 = changed mid-visit."""
        if self.feeder_open_frames is None or self.feeder_close_frames is None:
            return None
        n = onsets.shape[0]
        status = np.zeros(n, dtype=float)
        for i, (fs, fe) in enumerate(zip(onsets, offsets)):
            s_open = e_open = 0
            for fo, fc in zip(self.feeder_open_frames, self.feeder_close_frames):
                if fo < fs < fc:
                    s_open = 1
                if fo < fe < fc:
                    e_open = 1
            status[i] = np.mean((s_open, e_open))
        return status

    # ── convenience ──────────────────────────────────────────────────────

    def set_feeder_status(self, status: str) -> str:
        """Show only 'open' or only 'closed' feeder visits."""
        return self._build_feeder_set(status)

    def summary(self) -> str:
        parts = [f"{self.n_cells} cells, {self.n_frames} frames "
                 f"({self.n_frames * self.dt / 60:.1f} min @ {self.fps:g} Hz)"]
        for key, _ in PANEL_ORDER:
            ev = self.events[key]
            parts.append(f"{ev.label}: {ev.n}")
        if getattr(self, "n_feeder_partial", 0):
            parts.append(f"{self.n_feeder_partial} partial feeder visits excluded")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_raster(spike_counts_1d: np.ndarray,
                 align_frames: np.ndarray,
                 half_width: int) -> np.ndarray:
    """
    (n_events, 2*half_width+1) matrix of spike COUNTS per video frame,
    zero-padded at the edges of the session.
    """
    n_frames = spike_counts_1d.shape[0]
    n_t = 2 * half_width + 1
    out = np.zeros((align_frames.shape[0], n_t), dtype=np.int32)
    for i, a in enumerate(align_frames):
        s, e = a - half_width, a + half_width + 1
        s_c, e_c = max(s, 0), min(e, n_frames)
        if e_c <= s_c:
            continue
        out[i, s_c - s: s_c - s + (e_c - s_c)] = spike_counts_1d[s_c:e_c]
    return out


def raster_points(counts: np.ndarray, t_edges: np.ndarray, dt: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a count matrix into scatter coordinates.

    Spikes are recorded at 30 kHz but binned into 50 Hz video frames, so every
    spike in a frame would otherwise land on exactly the same x value and the
    raster shows vertical stripes.  Each spike is given a random offset inside
    its own frame, which removes the stripes and lets frames holding more than
    one spike show all of them instead of a single tick.
    """
    rows, cols = np.nonzero(counts)
    if rows.size == 0:
        return np.zeros(0), np.zeros(0)
    reps = counts[rows, cols].astype(int)
    rows = np.repeat(rows, reps)
    cols = np.repeat(cols, reps)
    t = t_edges[cols] + rng.uniform(0.0, dt, size=cols.size)
    return t, rows


def _nice_ceiling(v: float) -> float:
    """Round a firing rate up to a readable axis maximum (2, 5, 20, 25 ...)."""
    if not np.isfinite(v) or v <= 0:
        return 1.0
    exp = np.floor(np.log10(v))
    base = 10.0 ** exp
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * base * (1 + 1e-9):
            return float(m * base)
    return float(10 * base)


def _nice_xticks(ax, half_span: float, max_ticks: int = 5) -> None:
    """Round, evenly spaced x ticks instead of matplotlib's default clutter."""
    steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0]
    target = (2.0 * half_span) / max(max_ticks, 1)
    step = next((v for v in steps if v >= target), steps[-1])
    n = int(np.floor(half_span / step + 1e-9))
    ticks = np.round(np.arange(-n, n + 1) * step, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])


def build_trace_matrix(signal_1d: np.ndarray,
                       align_frames: np.ndarray,
                       half_width: int) -> np.ndarray:
    """
    (n_events, 2*half_width+1) matrix of a continuous signal cut around each
    event.  Frames outside the session are filled with NaN so they neither
    plot nor contribute to the mean.
    """
    n_frames = signal_1d.shape[0]
    n_t = 2 * half_width + 1
    out = np.full((align_frames.shape[0], n_t), np.nan)
    for i, a in enumerate(align_frames):
        s0, e0 = a - half_width, a + half_width + 1
        s_c, e_c = max(s0, 0), min(e0, n_frames)
        if e_c <= s_c:
            continue
        out[i, s_c - s0: s_c - s0 + (e_c - s_c)] = signal_1d[s_c:e_c]
    return out


def compute_psth(spike_fr_1d: np.ndarray,
                 align_frames: np.ndarray,
                 half_width: int,
                 dt: float,
                 sigma_frames: float) -> Tuple[Optional[np.ndarray], int]:
    """
    Mean firing rate (Hz) in a symmetric window around `align_frames`.
    Events whose full window leaves the session are dropped.
    """
    n_frames = spike_fr_1d.shape[0]
    keep = (align_frames - half_width >= 0) & \
           (align_frames + half_width + 1 <= n_frames)
    used = align_frames[keep]
    if used.shape[0] == 0:
        return None, 0
    snips = np.stack([spike_fr_1d[a - half_width: a + half_width + 1]
                      for a in used])
    psth = snips.mean(axis=0) / dt
    if sigma_frames > 0:
        psth = gaussian_filter1d(psth, sigma_frames, mode="nearest")
    return psth, int(used.shape[0])


@dataclass
class PlotParams:
    # per-event alignment: {'cache': 'onset'|'offset', ...}
    alignment: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ALIGNMENT))
    smooth_ms: float = 100.0
    sort_by_duration: bool = True
    # groups with fewer than this many usable events get no tuning curve
    # (0 disables the check)
    min_events: int = 0
    # most raster rows to draw per group on capped panels (0 = no cap).
    # Tuning curves always use every event.
    raster_cap: int = 100
    windows: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_WINDOWS.items()})

    def align_of(self, key: str) -> str:
        return self.alignment.get(key, DEFAULT_ALIGNMENT.get(key, "offset"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure drawing
# ─────────────────────────────────────────────────────────────────────────────

def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)


def _empty_panel(ax_b, ax_r, ax_p, label):
    for ax in (ax_b, ax_r, ax_p):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax_b.set_title(label, fontsize=PANEL_TITLE_SIZE, pad=4)
    ax_r.text(0.5, 0.5, f"no {label}", ha="center", va="center",
              transform=ax_r.transAxes, fontsize=AXIS_LABEL_SIZE,
              color="xkcd:grey")


def _draw_beak(ax_b, sess: SessionData, ev: EventSet, align: np.ndarray,
               half_r: int, t_raster: np.ndarray,
               show_label: bool) -> Optional[Tuple[float, float]]:
    """
    Vertical beak trajectory above a panel: one faint line per trial plus a
    heavy mean.  No axes are drawn; only the first panel is labelled.
    Returns the (min, max) of the plotted data, or None.
    """
    for side in ax_b.spines.values():
        side.set_visible(False)
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    if sess.beak_z is None or ev.n == 0:
        return None

    traces = build_trace_matrix(sess.beak_z, align, half_r)
    if not np.any(np.isfinite(traces)):
        return None

    # thin out the individual traces if there are very many events
    n_ev = traces.shape[0]
    if n_ev > BEAK_MAX_TRIALS:
        step = int(np.ceil(n_ev / BEAK_MAX_TRIALS))
        shown = traces[::step]
    else:
        shown = traces

    ax_b.plot(t_raster, shown.T, color="k",
              lw=BEAK_TRIAL_LW, alpha=BEAK_TRIAL_ALPHA, solid_capstyle="butt")

    groups = np.unique(ev.groups)
    if ev.split_beak_mean and groups.size > 1:
        # one mean per group, e.g. occupied vs. empty checks
        for g in groups:
            with np.errstate(invalid="ignore"):
                mean_trace = np.nanmean(traces[ev.groups == g], axis=0)
            ax_b.plot(t_raster, mean_trace, lw=BEAK_MEAN_LW, zorder=3,
                      color=BEAK_GROUP_COLORS[g % len(BEAK_GROUP_COLORS)])
    else:
        with np.errstate(invalid="ignore"):
            mean_trace = np.nanmean(traces, axis=0)
        ax_b.plot(t_raster, mean_trace, color="k", lw=BEAK_MEAN_LW, zorder=3)

    ax_b.axvline(0, color="xkcd:grey", ls="dashed", lw=0.7, zorder=1)
    ax_b.set_xlim(t_raster[0], t_raster[-1])

    if show_label:
        # horizontal, flush with the left edge of the trace
        ax_b.text(0.0, 1.04, "beak z", transform=ax_b.transAxes,
                  ha="left", va="bottom", fontsize=AXIS_LABEL_SIZE)

    finite = traces[np.isfinite(traces)]
    return float(np.min(finite)), float(np.max(finite))


def _draw_panel(fig, ax_b, ax_r, ax_p, sess: SessionData, ev: EventSet,
                row: int, p: PlotParams,
                show_beak_label: bool = False) -> Tuple[Optional[float],
                                                        Optional[Tuple[float, float]]]:
    """
    Draw one panel: beak trajectory, raster, tuning curve (top to bottom).
    Returns (PSTH y-max, beak z range), either of which may be None.
    """
    label = ev.label
    if ev.n == 0:
        _empty_panel(ax_b, ax_r, ax_p, label)
        return None, None

    dt = sess.dt
    spk_counts = sess.spike_fr[row]
    spk_fr = sess.spike_fr[row]
    # deterministic jitter and subsampling: redrawing a cell gives the same
    # raster rather than a different random one each time
    rng = np.random.default_rng((row + 1) * 7919 + sum(map(ord, ev.key)))

    align_mode = p.align_of(ev.key)
    align = ev.align_frames(align_mode)
    durations = ev.durations

    # ── raster ───────────────────────────────────────────────────────────
    raster_win = float(p.windows[ev.key]["raster"])
    half_r = max(int(round((raster_win / 2.0) / dt)), 1)
    t_raster = (np.arange(-half_r, half_r + 1)) * dt

    # sort: group first, then duration within group
    if p.sort_by_duration:
        order = np.lexsort((durations, ev.groups))
    else:
        order = np.lexsort((align, ev.groups))

    # Cap how many rows each group contributes, so a panel with a thousand
    # checks stays legible.  Tuning curves below still use every event.
    n_total = ev.n
    capped = False
    if ev.cap_raster and p.raster_cap and p.raster_cap > 0:
        kept = []
        for g in np.unique(ev.groups[order]):
            idx = order[ev.groups[order] == g]
            if idx.size > p.raster_cap:
                pick = np.sort(rng.choice(idx.size, p.raster_cap,
                                          replace=False))
                idx = idx[pick]
                capped = True
            kept.append(idx)
        order = np.concatenate(kept) if kept else order

    raster = build_raster(spk_counts, align[order], half_r)
    grp_sorted = ev.groups[order]
    dur_sorted = durations[order]

    n_ev = int(order.size)
    # Scale tick height so that consecutive rows touch: for the '|' marker the
    # drawn height is ~sqrt(s) points, and one row spans (axis height / n_ev)
    # points, so s = (row height)^2.  Only a lower bound is imposed, otherwise
    # panels with few events leave white gaps between the hash marks.
    ax_h_pts = ax_r.get_position().height * fig.get_figheight() * 72
    row_pts = ax_h_pts / max(n_ev, 1)
    spk_s = float(np.clip(row_pts ** 2, 0.6, RASTER_MAX_TICK_AREA))

    spk_t, spk_row = raster_points(raster, t_raster, dt, rng)
    if spk_row.size:
        row_group = grp_sorted[spk_row]
        for g in np.unique(grp_sorted):
            sel = row_group == g
            if not np.any(sel):
                continue
            color = ev.group_colors[g % len(ev.group_colors)]
            ax_r.scatter(spk_t[sel], spk_row[sel], marker="|",
                         color=color, lw=0.6, s=spk_s, alpha=1.0)

    # mark the other event boundary
    sign = 1.0 if align_mode == "onset" else -1.0
    other_t = sign * dur_sorted * dt
    vis = np.abs(other_t) <= (raster_win / 2.0)
    if np.any(vis):
        ax_r.scatter(other_t[vis], np.arange(n_ev)[vis], marker="|",
                     color="k", lw=EVENT_LW, s=spk_s / 2, zorder=3)

    ax_r.axvline(0, color="k", ls="dashed", lw=EVENT_LW, zorder=2)

    # group separators, and name the groups on the y axis
    uniq = np.unique(grp_sorted)
    if uniq.shape[0] > 1:
        for b in np.where(np.diff(grp_sorted) != 0)[0] + 1:
            # sit halfway between the last row of one group and the first of
            # the next, so the line never bisects a tick mark
            ax_r.axhline(b - 0.5, color="xkcd:grey", lw=0.8, zorder=4)

    # every raster counts events the same way: integer ticks, even spacing
    ax_r.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=5, integer=True))

    if uniq.shape[0] > 1 and ev.label_groups_on_yaxis:
        # Group names sit outside the tick labels, where the axis label would
        # normally go, so the ticks stay free for the event count.
        trans = mtransforms.blended_transform_factory(ax_r.transAxes,
                                                      ax_r.transData)
        # clear the tick labels: they grow with the number of digits shown
        pad = 16.0 + 6.6 * len(str(max(n_ev, 1)))
        for g in uniq:
            idx = np.where(grp_sorted == g)[0]
            name = ev.group_names[g] if g < len(ev.group_names) else str(g)
            ax_r.annotate(name, xy=(0.0, float(idx.mean())), xycoords=trans,
                          xytext=(-pad, 0), textcoords="offset points",
                          rotation=GROUP_LABEL_ROTATION,
                          ha="center", va="center", fontsize=AXIS_LABEL_SIZE)
    else:
        ax_r.set_ylabel("event #", fontsize=AXIS_LABEL_SIZE)

    if capped:
        ax_r.text(0.995, 0.985, f"{p.raster_cap}/group shown",
                  transform=ax_r.transAxes, ha="right", va="top",
                  fontsize=LEGEND_SIZE, color="xkcd:dark grey", zorder=6)

    ax_r.set_xlim(t_raster[0], t_raster[-1])
    ax_r.set_ylim(-0.5, max(n_ev, 1) - 0.5)
    ax_r.tick_params(labelbottom=True, labelsize=TICK_LABEL_SIZE)
    ax_r.set_xlabel(f"time from {align_mode} (s)", fontsize=AXIS_LABEL_SIZE)
    _nice_xticks(ax_r, raster_win / 2.0)
    _style_axis(ax_r)

    # ── beak trajectory, sharing the raster's time axis ──────────────────
    beak_range = _draw_beak(ax_b, sess, ev, align, half_r, t_raster,
                            show_beak_label)
    # the title belongs at the very top of the panel stack
    ax_b.set_title(f"{label} (n={n_total})", fontsize=PANEL_TITLE_SIZE, pad=4)

    # ── tuning curve ─────────────────────────────────────────────────────
    psth_win = float(p.windows[ev.key]["psth"])
    half_p = max(int(round(psth_win / dt)), 1)
    t_psth = np.arange(-half_p, half_p + 1) * dt
    sigma_frames = (p.smooth_ms / 1000.0) / dt

    y_max = 0.0
    any_trace = False
    n_skipped = 0
    for g in np.unique(ev.groups):
        sel = ev.groups == g
        psth, n_used = compute_psth(spk_fr, align[sel], half_p, dt, sigma_frames)
        if psth is None:
            continue
        if p.min_events and n_used < p.min_events:
            n_skipped += 1          # too few trials to average meaningfully
            continue
        any_trace = True
        color = ev.group_colors[g % len(ev.group_colors)]
        name = ev.group_names[g] if g < len(ev.group_names) else str(g)
        lbl = f"{name} (n={n_used})" if name else f"n={n_used}"
        ax_p.plot(t_psth, psth, lw=PSTH_LW, color=color, label=lbl)
        y_max = max(y_max, float(np.max(psth)))

    baseline = float(sess.mean_fr[row])
    ax_p.axhline(baseline, color="xkcd:grey", ls="dashed", lw=EVENT_LW)
    ax_p.axvline(0, color="k", ls="dashed", lw=EVENT_LW)
    ax_p.set_xlim(t_psth[0], t_psth[-1])
    _nice_xticks(ax_p, psth_win)
    ax_p.set_xlabel(f"time from {align_mode} (s)", fontsize=AXIS_LABEL_SIZE)
    ax_p.set_ylabel("firing rate (Hz)", fontsize=AXIS_LABEL_SIZE)
    _style_axis(ax_p)

    if ev.psth_legend and (len(ev.group_names) > 1
                           or (ev.group_names and ev.group_names[0])):
        ax_p.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper right",
                    handlelength=1.3, labelspacing=0.25, borderaxespad=0.3)

    if not any_trace:
        msg = (f"< {p.min_events} events" if n_skipped else "no complete windows")
        ax_p.text(0.5, 0.5, msg, ha="center", va="center",
                  transform=ax_p.transAxes, fontsize=AXIS_LABEL_SIZE,
                  color="xkcd:grey")
        ax_p.set_yticks([])
        return None, beak_range

    return max(y_max, baseline), beak_range



def _nice_step(target: float, steps: Sequence[float]) -> float:
    """Largest tabulated step that does not exceed `target` (else the smallest)."""
    ok = [v for v in steps if v <= target]
    return float(max(ok)) if ok else float(min(steps))


def _draw_waveform(fig: Figure, sess: SessionData, row: int) -> None:
    """
    Mean waveform for the selected cell, inset at the top-left of the figure.
    No axes or ticks: scale is conveyed by an L-shaped bar giving ms and uV.
    """
    wf = sess.waveform(row)
    if wf is None or sess.wf_time_ms is None or not np.any(np.isfinite(wf)):
        return

    fig_pts = fig.get_figheight() * 72.0
    y1 = 1.0 - (WAVEFORM_TITLE_TOP_PTS + WAVEFORM_TITLE_SIZE + 5) / fig_pts
    y0 = y1 - WAVEFORM_H_PTS / fig_pts
    ax = fig.add_axes([WAVEFORM_X0, y0, WAVEFORM_W,
                       max(y1 - y0, 0.01)])
    ax.set_facecolor("none")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # crop to a short window around the aligned peak (t = 0)
    t = sess.wf_time_ms
    lo, hi = WAVEFORM_WINDOW_MS
    keep = (t >= lo) & (t <= hi)
    if keep.sum() >= 3:
        t, wf = t[keep], wf[keep]

    ax.plot(t, wf, color="k", lw=WAVEFORM_LW, solid_joinstyle="round")
    ax.set_title("avg. waveform", fontsize=WAVEFORM_TITLE_SIZE, pad=3)

    t_span = float(t[-1] - t[0])
    v_lo, v_hi = float(np.nanmin(wf)), float(np.nanmax(wf))
    v_span = max(v_hi - v_lo, 1e-6)

    # room to the left and below for the scale bar
    ax.set_xlim(t[0] - 0.30 * t_span, t[-1] + 0.04 * t_span)
    ax.set_ylim(v_lo - 0.62 * v_span, v_hi + 0.10 * v_span)

    # bar sizes: roughly a quarter of the trace extent, rounded to a nice value
    t_bar = _nice_step(0.30 * t_span, TIME_BAR_STEPS)
    v_bar = _nice_step(0.35 * v_span, VOLTAGE_BAR_STEPS)

    x0 = t[0] - 0.23 * t_span
    y0 = v_lo - 0.26 * v_span
    ax.plot([x0, x0 + t_bar], [y0, y0], color="k", lw=1.4,
            solid_capstyle="butt", clip_on=False)
    ax.plot([x0, x0], [y0, y0 + v_bar], color="k", lw=1.4,
            solid_capstyle="butt", clip_on=False)

    ax.text(x0 + t_bar / 2, y0 - 0.04 * v_span,
            f"{t_bar:g} ms", ha="center", va="top",
            fontsize=SCALEBAR_LABEL_SIZE)
    ax.text(x0 - 0.025 * t_span, y0 + v_bar / 2,
            f"{v_bar:g} \u00b5V", ha="right", va="center",
            fontsize=SCALEBAR_LABEL_SIZE, rotation=90)


def draw_cell(fig: Figure, sess: SessionData, cell_id: int,
              p: PlotParams) -> None:
    """
    Draw the full six-panel summary for one cell onto `fig`.

    `cell_id` is a phy cluster ID when waveformStruct.mat has been loaded,
    otherwise a row index into aligned_spikes.npy.
    """
    row = sess.row_of(cell_id)
    if row is None:
        kind = "cluster" if sess.ids_are_clusters else "row"
        raise KeyError(f"cell {cell_id} is not in this session "
                       f"({kind} IDs range from {sess.cluster_ids.min()} to "
                       f"{sess.cluster_ids.max()}).")
    fig.clear()

    # Lay the panels out by hand in figure coordinates.  Fixed point heights
    # for the beak strip, the tuning curve and the gap under the raster mean
    # those labels keep their clearance no matter how tall the canvas is; the
    # raster simply absorbs whatever height is left over.
    fig_pts = fig.get_figheight() * 72.0
    grid_top = 1.0 - TOP_BAND_PTS / fig_pts
    grid_bottom = BOTTOM_PTS / fig_pts
    row_gap = ROW_GAP_PTS / fig_pts
    row_h = ((grid_top - grid_bottom) - row_gap) / 2.0
    col_w = (GRID_RIGHT - GRID_LEFT - 2 * COL_GAP_FRAC) / 3.0

    # The gap under the raster is fixed -- it holds tick labels and an axis
    # label, so shrinking it is what caused clipping.  Everything else is
    # interpolated between its preferred and minimum height according to how
    # much room the canvas actually has.
    gap_h = RASTER_XLABEL_PTS / fig_pts
    avail_pts = (row_h - gap_h) * fig_pts

    want = np.array([PANEL_BEAK_PTS, PANEL_RASTER_PTS, PANEL_PSTH_PTS],
                    dtype=float)
    least = np.array([PANEL_BEAK_MIN_PTS, PANEL_RASTER_MIN_PTS,
                      PANEL_PSTH_MIN_PTS], dtype=float)

    if avail_pts >= want.sum():
        heights = want.copy()
        heights[1] += avail_pts - want.sum()      # surplus goes to the raster
    elif avail_pts > least.sum():
        frac = (avail_pts - least.sum()) / (want.sum() - least.sum())
        heights = least + frac * (want - least)
    else:                                          # desperately short canvas
        heights = least * max(avail_pts / least.sum(), 0.1)

    beak_h, raster_h, psth_h = (heights / fig_pts)
    raster_h = max(raster_h, 0.005)

    psth_axes: Dict[str, plt.Axes] = {}
    psth_max: Dict[str, Optional[float]] = {}

    beak_axes: Dict[str, plt.Axes] = {}
    beak_range: Dict[str, Optional[Tuple[float, float]]] = {}

    for i, (key, _label) in enumerate(PANEL_ORDER):
        # NB: grid_row/grid_col, not row/col -- `row` is the aligned_spikes
        # row for the selected cell and must not be shadowed here
        grid_row, grid_col = divmod(i, 3)
        x0 = GRID_LEFT + grid_col * (col_w + COL_GAP_FRAC)
        cell_top = grid_top - grid_row * (row_h + row_gap)
        cell_bot = cell_top - row_h

        ax_b = fig.add_axes([x0, cell_top - beak_h, col_w, beak_h])
        ax_r = fig.add_axes([x0, cell_bot + psth_h + gap_h, col_w, raster_h])
        ax_p = fig.add_axes([x0, cell_bot, col_w, psth_h])

        psth_axes[key] = ax_p
        beak_axes[key] = ax_b
        psth_max[key], beak_range[key] = _draw_panel(
            fig, ax_b, ax_r, ax_p, sess, sess.events[key], row, p,
            show_beak_label=(i == 0))

    # shared y-axes for the tuning curves
    for group in Y_SHARE_GROUPS:
        vals = [psth_max[k] for k in group if psth_max.get(k) is not None]
        if not vals:
            continue
        top = _nice_ceiling(max(vals) * 1.18)
        for k in group:
            ax = psth_axes.get(k)
            if ax is None:
                continue
            ax.set_ylim(0, top)
            ax.set_yticks([0, top])
            ax.set_yticklabels([f"{v:g}" for v in (0, top)])

    # one shared y-range for every beak trace, so heights are comparable
    ranges = [r for r in beak_range.values() if r is not None]
    if ranges:
        lo = min(r[0] for r in ranges)
        hi = max(r[1] for r in ranges)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        for ax in beak_axes.values():
            ax.set_ylim(lo - pad, hi + pad)

    _draw_waveform(fig, sess, row)

    # header: bird / session in bold, cell details on the line below
    session_line = f"{sess.bird} {sess.session_id}".strip()
    if session_line:
        fig.text(0.5, 1.0 - 6.0 / fig_pts, session_line,
                 ha="center", va="top",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
        cell_y = 1.0 - (10.0 + TITLE_SIZE + 1) / fig_pts
    else:
        cell_y = 1.0 - 8.0 / fig_pts

    cell_line = (f"cell {cell_id}   \u2014   "
                 f"session mean {sess.mean_fr[row]:.2f} Hz")
    fig.text(0.5, cell_y, cell_line, ha="center", va="top",
             fontsize=TITLE_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

def sibling_path(path: str, filename: str) -> str:
    """
    Path to `filename` next to `path`, keeping the separator style already in
    use.  os.path.join on Windows appends a backslash, which gives ugly mixed
    paths like 'Z:/data/session/behavior_data\\posture_pos_smooth.npy'.
    """
    idx = max(path.rfind("/"), path.rfind("\\"))
    if idx < 0:
        return filename                      # bare filename, no folder part
    sep = path[idx]                          # reuse whichever separator is used
    folder = path[:idx].rstrip("/\\")
    if not folder:
        return sep + filename                # file sits at the filesystem root
    return folder + sep + filename


def parse_feeder_times(text: str) -> Tuple[Optional[List[float]],
                                           Optional[List[float]]]:
    """
    Parse feeder open periods, e.g. '10-20, 65-75' (minutes).
    Returns (open_times, close_times) or (None, None) if the field is empty.
    """
    text = (text or "").strip()
    if not text:
        return None, None
    opens, closes = [], []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"could not parse feeder period '{chunk}'")
        a, b = chunk.split("-", 1)
        opens.append(float(a))
        closes.append(float(b))
    if not opens:
        return None, None
    return opens, closes


class CellBrowser:
    """tkinter front end."""

    def __init__(self, root, prefill: Optional[dict] = None):
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self._NavigationToolbar2Tk = NavigationToolbar2Tk

        self.root = root
        self.root.title("Chickadee cell browser")
        # Size to the screen rather than a fixed 1750x1180: on a 1080p display
        # a taller window runs off the bottom and the plot looks cut off.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = min(1750, int(screen_w * 0.95))
        win_h = min(1180, int(screen_h * 0.90))
        self.root.geometry(f"{win_w}x{win_h}+"
                           f"{max((screen_w - win_w) // 2, 0)}+"
                           f"{max((screen_h - win_h) // 3, 0)}")
        self.root.minsize(min(1150, win_w), min(720, win_h))

        self.sess: Optional[SessionData] = None
        prefill = prefill or {}

        # ── tk variables ─────────────────────────────────────────────────
        self.var_spikes = tk.StringVar(value=prefill.get("spikes", ""))
        self.var_seeds = tk.StringVar(value=prefill.get("seeds", ""))
        self.var_bird = tk.StringVar(value=prefill.get("bird", ""))
        self.var_session = tk.StringVar(value=prefill.get("session", ""))
        self.var_fps = tk.StringVar(value=str(prefill.get("fps", DEFAULT_FPS)))
        self.var_feeder_times = tk.StringVar(value=prefill.get("feeder_times", ""))
        self.var_posture = tk.StringVar(value=prefill.get("posture", ""))
        self.var_waveforms = tk.StringVar(value=prefill.get("waveforms", ""))

        self.var_cell = tk.StringVar(value="0")
        self.var_smooth = tk.StringVar(value="100")
        self.var_sort = tk.BooleanVar(value=True)
        _min = int(prefill.get("min_events", 0) or 0)
        self.var_min_on = tk.BooleanVar(value=_min > 0)
        self.var_min_events = tk.StringVar(value=str(_min if _min > 0 else 5))
        self.var_raster_cap = tk.StringVar(value="100")
        self.var_feeder_status = tk.StringVar(value="open")
        self.var_status = tk.StringVar(value="Load a session to begin.")
        self.var_cellinfo = tk.StringVar(value="")
        self.var_loaded = tk.StringVar(value="no session loaded")

        # per-event time windows and alignment
        self.win_vars: Dict[str, Dict[str, tk.StringVar]] = {}
        self.align_vars: Dict[str, tk.BooleanVar] = {}
        for key, _ in PANEL_ORDER:
            self.win_vars[key] = {
                "raster": tk.StringVar(value=str(DEFAULT_WINDOWS[key]["raster"])),
                "psth": tk.StringVar(value=str(DEFAULT_WINDOWS[key]["psth"])),
            }
            self.align_vars[key] = tk.BooleanVar(
                value=DEFAULT_ALIGNMENT[key] == "offset")

        self._build_layout()

    # ── layout ───────────────────────────────────────────────────────────

    def _build_layout(self):
        root = self.root

        # The file panel costs ~110 px of canvas height, which matters a lot on
        # a 1080p screen, so it can be folded away once a session is loaded.
        bar = ttk.Frame(root)
        bar.pack(side="top", fill="x", padx=8, pady=(6, 0))
        self.var_files_btn = tk.StringVar(value="\u25bc  Session files")
        ttk.Button(bar, textvariable=self.var_files_btn, width=20,
                   command=self._toggle_file_panel).pack(side="left")
        ttk.Label(bar, textvariable=self.var_loaded, foreground=TK_MUTED_FG
                  ).pack(side="left", padx=10)

        self.file_panel = ttk.LabelFrame(root, text="Session files", padding=6)
        self.file_panel.pack(side="top", fill="x", padx=8, pady=(2, 4))
        self._files_visible = True
        self._build_file_bar(self.file_panel)

        status = ttk.Frame(root, padding=(10, 3))
        status.pack(side="bottom", fill="x")
        ttk.Label(status, textvariable=self.var_status,
                  anchor="w").pack(side="left", fill="x", expand=True)

        self.main_area = ttk.Frame(root)
        self.main_area.pack(side="top", fill="both", expand=True)

        side = ttk.Frame(self.main_area, padding=(8, 4))
        side.pack(side="left", fill="y")
        self._build_sidebar(side)

        plot_frame = ttk.Frame(self.main_area, padding=(4, 4))
        plot_frame.pack(side="right", fill="both", expand=True)
        self._build_canvas(plot_frame)

    def _toggle_file_panel(self):
        """Fold the file panel away to give the figure more height."""
        if self._files_visible:
            self.file_panel.pack_forget()
            self.var_files_btn.set("\u25b6  Session files")
        else:
            self.file_panel.pack(side="top", fill="x", padx=8, pady=(2, 4),
                                 before=self.main_area)
            self.var_files_btn.set("\u25bc  Session files")
        self._files_visible = not self._files_visible
        self.root.update_idletasks()
        self._last_canvas_size = None
        self.refresh()

    def _build_file_bar(self, parent):
        for c in (1, 4):
            parent.columnconfigure(c, weight=1)

        ttk.Label(parent, text="aligned_spikes.npy").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(parent, textvariable=self.var_spikes).grid(
            row=0, column=1, sticky="ew", padx=2)
        ttk.Button(parent, text="Browse\u2026", width=10,
                   command=self._browse_spikes).grid(row=0, column=2, padx=(2, 12))

        ttk.Label(parent, text="annotatedSeeds.mat").grid(
            row=0, column=3, sticky="w", padx=(0, 4))
        ttk.Entry(parent, textvariable=self.var_seeds).grid(
            row=0, column=4, sticky="ew", padx=2)
        ttk.Button(parent, text="Browse\u2026", width=10,
                   command=self._browse_seeds).grid(row=0, column=5, padx=2)

        ttk.Label(parent, text="posture_pos_smooth.npy").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.var_posture).grid(
            row=1, column=1, sticky="ew", padx=2, pady=(4, 0))
        ttk.Button(parent, text="Browse\u2026", width=10,
                   command=self._browse_posture).grid(row=1, column=2,
                                                      padx=(2, 12), pady=(4, 0))
        ttk.Label(parent, text="(optional \u2014 for the beak trajectory)",
                  foreground=TK_MUTED_FG).grid(
            row=1, column=3, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="waveformStruct.mat").grid(
            row=2, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        ttk.Entry(parent, textvariable=self.var_waveforms).grid(
            row=2, column=1, sticky="ew", padx=2, pady=(4, 0))
        ttk.Button(parent, text="Browse\u2026", width=10,
                   command=self._browse_waveforms).grid(row=2, column=2,
                                                        padx=(2, 12), pady=(4, 0))
        ttk.Label(parent, text="(optional \u2014 for the mean waveform inset)",
                  foreground=TK_MUTED_FG).grid(
            row=2, column=3, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="bird ID").grid(row=3, column=0, sticky="w",
                                               pady=(6, 0))
        ttk.Entry(parent, textvariable=self.var_bird, width=14).grid(
            row=3, column=1, sticky="w", padx=2, pady=(6, 0))

        meta = ttk.Frame(parent)
        meta.grid(row=3, column=3, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(meta, text="session ID").pack(side="left")
        ttk.Entry(meta, textvariable=self.var_session, width=12).pack(
            side="left", padx=(4, 12))
        ttk.Label(meta, text="fps").pack(side="left")
        ttk.Entry(meta, textvariable=self.var_fps, width=6).pack(
            side="left", padx=(4, 12))
        ttk.Label(meta, text="feeder open (min, e.g. 10-20, 65-75)").pack(side="left")
        ttk.Entry(meta, textvariable=self.var_feeder_times, width=20).pack(
            side="left", padx=(4, 12))
        ttk.Button(meta, text="Load session",
                   command=self.load_session).pack(side="left")

    def _build_sidebar(self, parent):
        # cell navigation
        nav = ttk.LabelFrame(parent, text="Cell", padding=6)
        nav.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(nav)
        row.pack(fill="x")
        ttk.Label(row, text="cell ID").pack(side="left")
        self.spin_cell = ttk.Spinbox(row, values=(0,), width=7,
                                     textvariable=self.var_cell,
                                     command=self.refresh)
        self.spin_cell.pack(side="left", padx=6)
        self.spin_cell.bind("<Return>", lambda e: self.refresh())
        ttk.Button(row, text="\u25c0", width=3,
                   command=lambda: self.step_cell(-1)).pack(side="left")
        ttk.Button(row, text="\u25b6", width=3,
                   command=lambda: self.step_cell(1)).pack(side="left", padx=(2, 0))

        ttk.Label(nav, textvariable=self.var_cellinfo,
                  foreground=TK_MUTED_FG).pack(anchor="w", pady=(6, 0))

        # display options
        disp = ttk.LabelFrame(parent, text="Display", padding=6)
        disp.pack(fill="x", pady=(0, 8))

        ttk.Label(disp, text="smoothing (ms)").grid(row=0, column=0, sticky="w")
        e = ttk.Entry(disp, textvariable=self.var_smooth, width=8)
        e.grid(row=0, column=1, sticky="w", pady=2)
        e.bind("<Return>", lambda ev: self.refresh())

        ttk.Checkbutton(disp, text="sort raster by duration",
                        variable=self.var_sort,
                        command=self.refresh).grid(row=1, column=0, columnspan=2,
                                                   sticky="w", pady=2)

        # hide tuning curves built from too few trials
        mbox = ttk.Frame(disp)
        mbox.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(mbox, text="hide curves with <",
                        variable=self.var_min_on,
                        command=self.refresh).pack(side="left")
        ent_min = ttk.Entry(mbox, textvariable=self.var_min_events, width=4)
        ent_min.pack(side="left", padx=3)
        ent_min.bind("<Return>", lambda ev: self.refresh())
        ttk.Label(mbox, text="events").pack(side="left")

        cbox = ttk.Frame(disp)
        cbox.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(cbox, text="max raster rows / group").pack(side="left")
        ent_cap = ttk.Entry(cbox, textvariable=self.var_raster_cap, width=5)
        ent_cap.pack(side="left", padx=3)
        ent_cap.bind("<Return>", lambda ev: self.refresh())

        ttk.Label(disp, text="feeder visits").grid(row=4, column=0, sticky="w")
        fbox = ttk.Frame(disp)
        fbox.grid(row=4, column=1, sticky="w", pady=2)
        self.feeder_radios = []
        for txt in ("open", "closed"):
            rb = ttk.Radiobutton(fbox, text=txt, value=txt,
                                 variable=self.var_feeder_status,
                                 command=self._change_feeder_status)
            rb.pack(side="left")
            self.feeder_radios.append(rb)

        # time windows
        win = ttk.LabelFrame(parent, text="Per-event settings", padding=6)
        win.pack(fill="x", pady=(0, 8))
        ttk.Label(win, text="event", width=15).grid(row=0, column=0, sticky="w")
        ttk.Label(win, text="raster\n(s)", width=7,
                  justify="center").grid(row=0, column=1)
        ttk.Label(win, text="curve\n\u00b1 (s)", width=7,
                  justify="center").grid(row=0, column=2)
        ttk.Label(win, text="align to\noffset", width=8,
                  justify="center").grid(row=0, column=3)
        for r, (key, label) in enumerate(PANEL_ORDER, start=1):
            ttk.Label(win, text=label).grid(row=r, column=0, sticky="w")
            for c, which in enumerate(("raster", "psth"), start=1):
                ent = ttk.Entry(win, textvariable=self.win_vars[key][which],
                                width=7)
                ent.grid(row=r, column=c, padx=2, pady=1)
                ent.bind("<Return>", lambda ev: self.refresh())
            ttk.Checkbutton(win, variable=self.align_vars[key],
                            command=self.refresh).grid(row=r, column=3)

        # actions
        act = ttk.Frame(parent)
        act.pack(fill="x", pady=(4, 0))
        ttk.Button(act, text="Update plot",
                   command=self.refresh).pack(fill="x", pady=2)
        ttk.Button(act, text="Save figure\u2026",
                   command=self.save_figure).pack(fill="x", pady=2)
        ttk.Button(act, text="Reset per-event settings",
                   command=self.reset_windows).pack(fill="x", pady=2)

        ttk.Label(parent, wraplength=230, foreground=TK_MUTED_FG,
                  text=("Tip: \u2190 / \u2192 step through cells when the plot "
                        "has focus.")).pack(anchor="w", pady=(10, 0))

    def _build_canvas(self, parent):
        self.fig = Figure(figsize=FIG_SIZE, dpi=100)
        self.canvas = self._FigureCanvasTkAgg(self.fig, master=parent)
        toolbar = self._NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Axes positions are stored as figure fractions, so when Tk resizes
        # the canvas every point-based clearance shrinks with it and the axis
        # labels start colliding.  Recompute the layout after each resize.
        self._resize_job = None
        self._last_canvas_size = None
        self._drawing = False
        # add="+" is essential: FigureCanvasTkAgg already binds <Configure> to
        # its own resize handler, and binding without "+" replaces it, leaving
        # the figure stuck at its construction size while the widget changes.
        self.canvas.get_tk_widget().bind("<Configure>", self._on_canvas_resize,
                                         add="+")

        self.root.bind("<Left>", lambda e: self.step_cell(-1))
        self.root.bind("<Right>", lambda e: self.step_cell(1))

        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Choose aligned_spikes.npy and annotatedSeeds.mat,\n"
                "then press 'Load session'.",
                ha="center", va="center", fontsize=13, color="xkcd:grey")
        self.canvas.draw()

    # ── callbacks ────────────────────────────────────────────────────────

    def _browse_spikes(self):
        path = filedialog.askopenfilename(
            title="Select aligned_spikes.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")])
        if path:
            self.var_spikes.set(path)
            for var, fname in ((self.var_seeds, "annotatedSeeds.mat"),
                               (self.var_posture, "posture_pos_smooth.npy"),
                               (self.var_waveforms, "waveformStruct.mat")):
                if not var.get():
                    guess = sibling_path(path, fname)
                    if os.path.exists(guess):
                        var.set(guess)

    def _on_canvas_resize(self, event):
        if self.sess is None or self._drawing:
            return
        size = (event.width, event.height)
        if size == self._last_canvas_size:
            return
        self._last_canvas_size = size
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        # debounce: only redraw once the drag has settled
        self._resize_job = self.root.after(150, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_job = None
        self.refresh()

    def _browse_waveforms(self):
        path = filedialog.askopenfilename(
            title="Select waveformStruct.mat",
            filetypes=[("MATLAB file", "*.mat"), ("All files", "*.*")])
        if path:
            self.var_waveforms.set(path)

    def _browse_posture(self):
        path = filedialog.askopenfilename(
            title="Select posture_pos_smooth.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")])
        if path:
            self.var_posture.set(path)

    def _browse_seeds(self):
        path = filedialog.askopenfilename(
            title="Select annotatedSeeds.mat",
            filetypes=[("MATLAB file", "*.mat"), ("All files", "*.*")])
        if path:
            self.var_seeds.set(path)

    def load_session(self):
        spikes = self.var_spikes.get().strip()
        seeds = self.var_seeds.get().strip()
        if not spikes or not os.path.exists(spikes):
            messagebox.showerror("Missing file",
                                 "Please select a valid aligned_spikes.npy.")
            return
        if not seeds or not os.path.exists(seeds):
            messagebox.showerror("Missing file",
                                 "Please select a valid annotatedSeeds.mat.")
            return
        try:
            fps = float(self.var_fps.get())
            if fps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad value", "fps must be a positive number.")
            return
        try:
            opens, closes = parse_feeder_times(self.var_feeder_times.get())
        except ValueError as exc:
            messagebox.showerror("Bad value", str(exc))
            return
        posture = self.var_posture.get().strip()
        if posture and not os.path.exists(posture):
            messagebox.showerror("Missing file",
                                 f"Posture file not found:\n{posture}")
            return
        waveforms = self.var_waveforms.get().strip()
        if waveforms and not os.path.exists(waveforms):
            messagebox.showerror("Missing file",
                                 f"Waveform file not found:\n{waveforms}")
            return

        self.var_status.set("Loading\u2026")
        self.root.update_idletasks()
        try:
            self.sess = SessionData(spikes, seeds,
                                    bird=self.var_bird.get().strip(),
                                    session_id=self.var_session.get().strip(),
                                    fps=fps,
                                    feeder_open_times=opens,
                                    feeder_close_times=closes,
                                    posture_path=posture,
                                    waveform_path=waveforms)
        except Exception as exc:
            self.sess = None
            self.var_status.set("Load failed.")
            messagebox.showerror("Could not load session",
                                 f"{type(exc).__name__}: {exc}")
            return

        if self.sess.notes:
            messagebox.showwarning("Loaded with warnings",
                                   "\n\n".join(self.sess.notes))
        self.var_loaded.set(
            f"{self.sess.bird} {self.sess.session_id}  \u2014  "
            f"{self.sess.n_cells} cells".strip())
        if self._files_visible:
            self._toggle_file_panel()
        ids = [int(v) for v in self.sess.cluster_ids]
        self.spin_cell.configure(values=ids)
        if self._cell_id() not in ids:
            self.var_cell.set(str(ids[0]))
        self._change_feeder_status(silent=True)
        self.refresh()

    def _change_feeder_status(self, silent: bool = False):
        """Switch the feeder panel between open and closed visits."""
        if self.sess is None:
            return
        wanted = self.var_feeder_status.get()
        applied = self.sess.set_feeder_status(wanted)
        no_times = (applied == "all")
        for rb in getattr(self, "feeder_radios", []):
            rb.state(["disabled"] if no_times else ["!disabled"])
        if no_times and not silent:
            messagebox.showinfo(
                "Feeder times needed",
                "Enter feeder open periods (e.g. '10-20, 65-75') in the "
                "session bar to separate open from closed feeder visits. "
                "All feeder-perch visits are shown for now.")
        if not silent:
            self.refresh()

    def _cell_id(self) -> Optional[int]:
        try:
            return int(float(self.var_cell.get()))
        except ValueError:
            return None

    def step_cell(self, delta: int):
        """Move to the next/previous cell in the session's ID list."""
        if self.sess is None:
            return
        self.var_loaded.set(
            f"{self.sess.bird} {self.sess.session_id}  \u2014  "
            f"{self.sess.n_cells} cells".strip())
        if self._files_visible:
            self._toggle_file_panel()
        ids = [int(v) for v in self.sess.cluster_ids]
        cur = self._cell_id()
        if cur in ids:
            pos = ids.index(cur)
        else:
            pos = 0
        pos = int(np.clip(pos + delta, 0, len(ids) - 1))
        self.var_cell.set(str(ids[pos]))
        self.refresh()

    def _collect_params(self) -> Optional[PlotParams]:
        try:
            smooth = float(self.var_smooth.get())
            if smooth < 0:
                raise ValueError("smoothing must be >= 0")
            min_events = 0
            if self.var_min_on.get():
                min_events = int(float(self.var_min_events.get()))
                if min_events < 0:
                    raise ValueError("minimum event count must be >= 0")
            raster_cap = int(float(self.var_raster_cap.get()))
            if raster_cap < 0:
                raise ValueError("max raster rows must be >= 0")
            windows = {}
            for key, _ in PANEL_ORDER:
                r = float(self.win_vars[key]["raster"].get())
                q = float(self.win_vars[key]["psth"].get())
                if r <= 0 or q <= 0:
                    raise ValueError("time windows must be > 0")
                windows[key] = {"raster": r, "psth": q}
        except ValueError as exc:
            messagebox.showerror("Bad value", str(exc))
            return None
        alignment = {key: ("offset" if self.align_vars[key].get() else "onset")
                     for key, _ in PANEL_ORDER}
        return PlotParams(alignment=alignment,
                          smooth_ms=smooth,
                          sort_by_duration=bool(self.var_sort.get()),
                          min_events=min_events,
                          raster_cap=raster_cap,
                          windows=windows)

    def refresh(self):
        if self.sess is None:
            return
        params = self._collect_params()
        if params is None:
            return

        cid = self._cell_id()
        if cid is None or self.sess.row_of(cid) is None:
            # showing a different cell than the one asked for is exactly the
            # bug this guards against, so say so instead of substituting
            kind = "cluster" if self.sess.ids_are_clusters else "row"
            self.var_status.set(
                f"cell {self.var_cell.get()!r} is not in this session "
                f"({kind} IDs {self.sess.cluster_ids.min()}"
                f"-{self.sess.cluster_ids.max()})")
            return
        row = self.sess.row_of(cid)

        self.var_status.set("Plotting\u2026")
        self.root.update_idletasks()
        self._drawing = True
        try:
            draw_cell(self.fig, self.sess, cid, params)
            self.canvas.draw()
        except Exception as exc:
            self.var_status.set("Plot failed.")
            messagebox.showerror("Plotting error",
                                 f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._drawing = False

        n_spikes = int(self.sess.spike_fr[row].sum())
        label = "phy cluster" if self.sess.ids_are_clusters else "row"
        info = [f"{label} {cid}  (row {row} of {self.sess.n_cells - 1})",
                f"{n_spikes} spikes, {self.sess.mean_fr[row]:.2f} Hz in session"]
        if (self.sess.wf_mean_rate is not None
                and row < self.sess.wf_mean_rate.size):
            info.append(f"{self.sess.wf_mean_rate[row]:.2f} Hz over recording")
        self.var_cellinfo.set("\n".join(info))
        self.var_status.set(self.sess.summary())

    def reset_windows(self):
        for key, _ in PANEL_ORDER:
            self.win_vars[key]["raster"].set(str(DEFAULT_WINDOWS[key]["raster"]))
            self.win_vars[key]["psth"].set(str(DEFAULT_WINDOWS[key]["psth"]))
            self.align_vars[key].set(DEFAULT_ALIGNMENT[key] == "offset")
        self.refresh()

    def save_figure(self):
        if self.sess is None:
            return
        cid = self._cell_id()
        if cid is None or self.sess.row_of(cid) is None:
            return
        default = f"{self.sess.bird}_{self.sess.session_id}_cell{cid}.png".lstrip("_")
        path = filedialog.asksaveasfilename(
            title="Save figure", defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        params = self._collect_params()
        if params is None:
            return
        # Render at the full figure size regardless of how small the window
        # currently is, so a cramped canvas never ends up in a saved figure.
        current = tuple(self.fig.get_size_inches())
        self._drawing = True
        try:
            self.fig.set_size_inches(*FIG_SIZE, forward=False)
            draw_cell(self.fig, self.sess, cid, params)
            self.fig.savefig(path, dpi=300, bbox_inches="tight",
                             facecolor="white")
        except Exception as exc:
            messagebox.showerror("Could not save",
                                 f"{type(exc).__name__}: {exc}")
        finally:
            self.fig.set_size_inches(*current, forward=False)
            try:
                draw_cell(self.fig, self.sess, cid, params)
                self.canvas.draw()
            except Exception:
                pass
            self._drawing = False
        self.var_status.set(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spikes", default="", help="path to aligned_spikes.npy")
    ap.add_argument("--seeds", default="", help="path to annotatedSeeds.mat")
    ap.add_argument("--bird", default="", help="bird ID, e.g. LMN88")
    ap.add_argument("--session", default="", help="session ID, e.g. 260727")
    ap.add_argument("--fps", default=DEFAULT_FPS, type=float,
                    help="video frame rate (Hz)")
    ap.add_argument("--posture", default="",
                    help="path to posture_pos_smooth.npy (optional)")
    ap.add_argument("--min-events", default=0, type=int,
                    help="hide tuning curves built from fewer events than this")
    ap.add_argument("--waveforms", default="",
                    help="path to waveformStruct.mat (optional)")
    ap.add_argument("--feeder-times", default="",
                    help="feeder open periods in minutes, e.g. '10-20, 65-75'")
    ap.add_argument("--inspect-waveforms", default="", metavar="FILE",
                    help="print the layout of a waveformStruct.mat and exit")
    args = ap.parse_args(argv)

    if args.inspect_waveforms:
        print(inspect_waveform_file(args.inspect_waveforms))
        return

    if not _HAS_TK:
        sys.exit("tkinter is not available in this Python installation. "
                 "Install it (e.g. 'sudo apt install python3-tk') and retry.")

    matplotlib.use("TkAgg")
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    app = CellBrowser(root, prefill=dict(
        spikes=args.spikes, seeds=args.seeds, bird=args.bird,
        session=args.session, fps=args.fps, feeder_times=args.feeder_times,
        posture=args.posture, waveforms=args.waveforms,
        min_events=args.min_events))

    if args.spikes and args.seeds:
        root.after(200, app.load_session)

    root.mainloop()


if __name__ == "__main__":
    main()
