#!/usr/bin/env python
"""
spike_video_gui.py
==================

Interactive viewer for behavior video with simultaneously-recorded spiking
activity, for the chickadee caching task (Chettih et al. 2024 paradigm).

Layout
------
    top     : one camera view (pan / zoom)
    middle  : one-row ethogram -- feeder / eating / cache / retrieve / check
              spans, color-coded, with a colored-text legend above it
    bottom  : spike raster (default) or firing-rate heatmap, cells on y, time
              on x, centered on the current frame (thin red line).  In the
              raster each spike is a tick the height of its cell's row,
              jittered within its video frame the same way event_psth
              .raster_scatter does it.  In the heatmap black = max, white =
              min.  'd' switches modes.  Click a row to highlight it (red)
              and show its phy cell ID in the status bar; click the same row
              again to clear

Launch it from a thin wrapper script (see run_spike_video_gui.py) rather than
importing it interactively -- Qt wants to own the main thread.

Dependencies
------------
    numpy, scipy, pandas, opencv-python, pyqtgraph, and one Qt binding
    (PyQt5 / PyQt6 / PySide2 / PySide6 -- pyqtgraph picks whichever it finds).

        pip install pyqtgraph opencv-python PyQt5

Session layout (as built by build_data_dict.py)
-----------------------------------------------
    <session_dir>/behavior_data/aligned_spikes.npy      (n_cells, n_frames)
    <session_dir>/behavior_data/annotatedSeeds.mat      behavior annotation
    <session_dir>/<bird>_<ephys_id>/kilosort4*/cluster_group.tsv

`data_dir` is the behavior_data folder.  Events are computed live with
behavior/format_behavior_data.py, and cluster IDs are read from the phy tsv the
same way format_waveform_data.get_good_cluster_ids does, so nothing here
duplicates the pipeline -- point `repo_root` at your lhy_hpc checkout.

Normalization
-------------
Firing rates are normalized as in build_data_dict.collect_population_vectors /
format_waveform_data.pop_normalize:

    inst_fr = aligned_spikes / dt
    sd      = std(inst_fr, axis=1, ddof=1) + std_reg
    norm_fr = (inst_fr - moving_avg(inst_fr, baseline_window)) / sd

(the two versions in the repo differ only in the order of the divide and the
subtraction, which commute, and in std_reg: 0.6 vs 1e-2)

`norm_across_cells` adds a display-only second step on top of that: each row is
rescaled by its own robust range, so a cell whose SD sits well below std_reg is
not washed out by a high-rate neighbour.
"""

import os
import sys
import glob
import warnings
from collections import OrderedDict

import numpy as np
from matplotlib.colors import to_hex

try:
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
except ImportError:
    gaussian_filter1d = uniform_filter1d = None

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# --------------------------------------------------------------------------- #
#  REPO IMPORTS                                                                #
# --------------------------------------------------------------------------- #

def _default_repo_root():
    """If this file lives inside lhy_hpc, find the repo root from it."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(candidate, 'utils')):
            return candidate
    return None


def add_repo_to_path(repo_root=None):
    """Put lhy_hpc's utils/behavior/neural folders on sys.path."""
    repo_root = repo_root or _default_repo_root()
    if repo_root is None:
        return None
    repo_root = os.path.abspath(repo_root)
    for sub in ('utils', 'behavior', 'neural', 'stim', 'anatomy', ''):
        path = os.path.join(repo_root, sub)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
    return repo_root


# name -> (hotkey, color, legend label).  Order sets the legend order and the
# ethogram draw order: later entries are drawn on top, so the rare events go
# last in case two annotations ever do overlap.
EVENT_SPEC = OrderedDict([
    ('feeder',   dict(key='f', color=to_hex('xkcd:scarlet'),     label='feeder')),
    ('eat',      dict(key='e', color='#006666',                  label='eating')),
    ('cache',    dict(key='c', color=to_hex('xkcd:orange'),      label='cache')),
    ('retrieve', dict(key='r', color=to_hex('xkcd:purple'),      label='retrieve')),
    ('check',    dict(key='k', color=to_hex('xkcd:deep green'),  label='check')),
])

# left-axis width, shared by the ethogram and the heatmap so they line up
AXIS_WIDTH = 96

# raster tick height, as a fraction of a cell's row (1.0 = rows just touch)
RASTER_TICK_HEIGHT = 1.0

# per-cell jitter seed, matching plot_baited_cached_activity.py so a cell's
# raster looks the same here as in the static figures
RASTER_SEED_MULT = 7919


def _css_color(color):
    """EVENT_SPEC colors as CSS, whether they are hex strings or RGB tuples."""
    if isinstance(color, str):
        return color
    return 'rgb(%d, %d, %d)' % tuple(int(c) for c in color[:3])


def _clean_event(onsets, offsets, n_frames, min_dur=1):
    """Sort, clip to the session, and guarantee offset > onset."""
    on = np.round(np.asarray(onsets, dtype=float).ravel()).astype(np.int64)
    off = np.round(np.asarray(offsets, dtype=float).ravel()).astype(np.int64)
    if off.size != on.size:
        off = on.copy()
    keep = (on >= 0) & (on < n_frames)
    on, off = on[keep], off[keep]
    off = np.clip(np.maximum(off, on + min_dur), 0, n_frames - 1)
    order = np.argsort(on, kind='stable')
    return on[order], off[order]


def load_events(data_dir, n_frames, repo_root=None, refined=False,
                use_beak=True):
    """
    Compute cache / retrieve / check / eat / feeder events straight from
    annotatedSeeds.mat with behavior/format_behavior_data.py.

    refined : bool
        False (default) -- raw event bounds (get_cache_ints, get_retrieve_ints,
        get_checks_raw, get_eating_bouts, get_feeder_ints).  These are the true
        event boundaries, which is what you want when looking at video.
        True -- the SC/EM 2024 analysis windows (get_*_refined): padded by
        250 ms and truncated against neighbouring interactions.

    Returns {name: (onsets, offsets)} in frames.
    """
    add_repo_to_path(repo_root)
    try:
        import format_behavior_data as fbd
    except ImportError as err:
        warnings.warn('could not import format_behavior_data (%s) -- no events '
                      'will be available.  Set repo_root, or pass events= '
                      'yourself.' % err)
        return OrderedDict((n, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
                           for n in EVENT_SPEC)

    data_dir = os.path.join(data_dir, '')      # load_behavior_data f-strings
    seed_struct, count_data = fbd.load_behavior_data(data_dir)

    if refined:
        c_on, c_off, _ = fbd.get_caches_refined(count_data, seed_struct, n_frames)
        r_on, r_off, _ = fbd.get_retrievals_refined(count_data, seed_struct, n_frames)
        k_on, k_off, _ = fbd.get_checks_refined(count_data, seed_struct, n_frames)
    else:
        c_on, c_off = fbd.get_cache_ints(count_data, seed_struct)
        r_on, r_off = fbd.get_retrieve_ints(count_data, seed_struct)
        k_on, k_off, _ = fbd.get_checks_raw(count_data, seed_struct)
    e_on, e_off = fbd.get_eating_bouts(count_data)
    f_on, f_off, _ = fbd.get_feeder_ints(count_data, use_beak=use_beak)

    out = OrderedDict()
    for name, (on, off) in zip(
            ('cache', 'retrieve', 'check', 'eat', 'feeder'),
            ((c_on, c_off), (r_on, r_off), (k_on, k_off),
             (e_on, e_off), (f_on, f_off))):
        out[name] = _clean_event(on, off, n_frames)
    return out


def load_cluster_ids(n_cells, data_dir=None, ks_dir=None, only_good=True):
    """
    phy cluster ID for each row of aligned_spikes.npy.

    align_spikes_behavior() fills those rows in the order get_spike_times()
    returns good_clusters -- cluster_group.tsv row order filtered to
    group == 'good' -- which is what this reproduces.  If ks_dir is not given,
    look for a kilosort folder next to data_dir.
    """
    if ks_dir is None and data_dir is not None:
        hits = sorted(glob.glob(os.path.join(
            data_dir, '..', '*', '*kilosort*', 'cluster_group.tsv')))
        if hits:
            ks_dir = os.path.dirname(hits[0])
            if len(hits) > 1:
                warnings.warn('several kilosort folders found; using %s' % ks_dir)
    if ks_dir is None:
        warnings.warn('no cluster_group.tsv found -- falling back to row '
                      'indices, so cell_ids= will not work.  Pass ks_dir=.')
        return np.arange(n_cells, dtype=np.int64)

    tsv = ks_dir if ks_dir.endswith('.tsv') else os.path.join(
        ks_dir, 'cluster_group.tsv')
    import pandas as pd
    phy_info = pd.read_csv(tsv, sep='\t')
    cluster_id = phy_info['cluster_id'].values
    if only_good:
        cluster_id = cluster_id[(phy_info['group'].values == 'good').astype(bool)]
    cluster_id = np.asarray(cluster_id).astype(np.int64)

    if cluster_id.size != n_cells:
        warnings.warn('%s has %d %sclusters but aligned_spikes has %d rows -- '
                      'falling back to row indices.  Check that only_good '
                      'matches how aligned_spikes was built.'
                      % (tsv, cluster_id.size, 'good ' if only_good else '',
                         n_cells))
        return np.arange(n_cells, dtype=np.int64)
    return cluster_id


# --------------------------------------------------------------------------- #
#  FIRING RATES                                                                #
# --------------------------------------------------------------------------- #

def _moving_avg(x, window_frames):
    """Running mean along the last axis, edge-padded."""
    if uniform_filter1d is None:
        raise ImportError('scipy is required for the running baseline')
    size = int(max(1, window_frames))
    if size % 2 == 0:
        size += 1
    return uniform_filter1d(x, size=size, axis=-1, mode='nearest')


def prepare_rates(aligned_spikes, fps=50, smoothing=None, std_reg=0.6,
                  baseline_window=30, baseline_units='minutes',
                  pop_norm=True, use_repo_moving_avg=True, clip_pct=99.5,
                  chunk=16):
    """
    aligned_spikes -> the same normalized rates the population-vector code uses.

        inst_fr = aligned_spikes / dt
        sd      = std(inst_fr, ddof=1) + std_reg    (from UNSMOOTHED rates, as
                                                     in the repo)
        norm_fr = (inst_fr - moving_avg(inst_fr)) / sd

    Smoothing, if any, is applied last and is display-only, so it does not
    change the units.

    baseline_units : 'minutes' | 'frames'
        helpers.moving_avg's window units are not obvious from the call site
        (build_data_dict passes 30 for what the docstring calls a 30 min
        baseline).  With use_repo_moving_avg=True and helpers importable this
        calls helpers.moving_avg(row, window=baseline_window) verbatim, so the
        display matches your population vectors exactly; otherwise the window
        is interpreted with these units.

    Returns (rates, scale).
    """
    counts = np.asarray(aligned_spikes)
    n_cells, n_frames = counts.shape
    dt = 1.0 / float(fps)

    repo_moving_avg = None
    if pop_norm and use_repo_moving_avg:
        try:
            import helpers
            repo_moving_avg = helpers.moving_avg
        except Exception:
            pass
    window_frames = (int(round(baseline_window * 60.0 * fps))
                     if baseline_units == 'minutes'
                     else int(round(baseline_window)))
    if pop_norm:
        print('  baseline: %s' % (
            'helpers.moving_avg(window=%s)' % baseline_window
            if repo_moving_avg is not None else
            'uniform_filter1d, %d frames (%.1f min)'
            % (window_frames, window_frames / fps / 60.0)))

    rates = np.empty((n_cells, n_frames), dtype=np.float32)
    sd = np.zeros(n_cells, dtype=np.float32)

    for i in range(0, n_cells, chunk):
        block = counts[i:i + chunk].astype(np.float32) / dt      # inst. rate
        if pop_norm:
            # SD from the raw instantaneous rate, as in the repo
            block_sd = np.std(block, axis=1, ddof=1) + std_reg
            if repo_moving_avg is not None:
                base = np.empty_like(block)
                for j in range(block.shape[0]):
                    base[j] = repo_moving_avg(block[j], window=baseline_window)
            else:
                base = _moving_avg(block, window_frames)
            block = (block - base) / block_sd[:, None]
            sd[i:i + chunk] = block_sd
        if smoothing:
            if gaussian_filter1d is None:
                raise ImportError('scipy is required for smoothing')
            block = gaussian_filter1d(block, sigma=float(smoothing), axis=1,
                                      mode='nearest')
        rates[i:i + chunk] = block

    # per-cell robust range, for the across-cell display normalization
    lo = np.empty(n_cells, dtype=np.float32)
    hi = np.empty(n_cells, dtype=np.float32)
    for i in range(n_cells):
        lo[i] = np.percentile(rates[i], 100.0 - clip_pct)
        hi[i] = np.percentile(rates[i], clip_pct)
    hi = np.maximum(hi, lo + 1e-6)

    scale = dict(lo=lo, hi=hi, sd=sd, pop_norm=pop_norm,
                 shared_levels=(float(np.percentile(lo, 5)),
                                float(np.percentile(hi, 95))),
                 units='SD' if pop_norm else 'Hz')
    return rates, scale


def build_spike_times(aligned_spikes, cluster_ids, fps=50):
    """
    Spike counts per frame -> one jittered spike-time array (seconds) per cell.

    Spikes are sorted at 30 kHz but binned into video frames, so without a
    within-frame offset every spike in a frame lands on the same x and a frame
    with three spikes looks like a frame with one -- the same reason
    event_psth.raster_scatter jitters.  The jitter is baked in once here, not
    regenerated per displayed frame, so ticks stay put during playback instead
    of shimmering.  Each cell is seeded from its phy ID, as in
    plot_baited_cached_activity.py, so a cell matches its static raster.

    Returns a list of float64 arrays, one per row of aligned_spikes.
    """
    dt = 1.0 / float(fps)
    spike_times = []
    for i in range(aligned_spikes.shape[0]):
        counts = np.asarray(aligned_spikes[i]).astype(np.int64)
        frames = np.flatnonzero(counts)
        if frames.size:
            frames = np.repeat(frames, counts[frames])
            rng = np.random.default_rng(int(cluster_ids[i]) * RASTER_SEED_MULT)
            times = (frames + rng.uniform(0.0, 1.0, size=frames.size)) * dt
            # jitter within a multi-spike frame is not drawn in order, so the
            # array picks up local inversions -- resort, since the windowing
            # below is searchsorted
            times.sort()
        else:
            times = np.zeros(0)
        spike_times.append(times)
    return spike_times


def normalize_window(win, scale, rows, norm_across_cells):
    """Display scaling for one heatmap window."""
    if not norm_across_cells:
        return win, scale['shared_levels']
    lo = scale['lo'][rows, None]
    hi = scale['hi'][rows, None]
    return (win - lo) / (hi - lo), (0.0, 1.0)


def sort_cells_by_peak_time(rates, cell_rows=None):
    """
    Order cells by when their rate peaks -- indices into the rows of
    aligned_spikes.npy, i.e. something to hand to `cell_sort`.  Most useful on
    an event-aligned average; over a whole 3 h session it is dominated by a
    single excursion per cell.
    """
    rows = np.arange(rates.shape[0]) if cell_rows is None else np.asarray(cell_rows)
    return rows[np.argsort(np.argmax(rates[rows], axis=1))]


# --------------------------------------------------------------------------- #
#  VIDEO                                                                       #
# --------------------------------------------------------------------------- #

class VideoReader(object):
    """
    Random-access frame reader with a small decoded-frame cache.

    Sequential reads are free, short forward jumps grab() their way there, and
    anything else seeks.  For MJPG .avi (every frame a keyframe) seeks land
    exactly; with a long-GOP codec they may snap to the nearest keyframe.
    """

    MAX_GRAB = 60
    CACHE_SIZE = 96

    def __init__(self, path, downsample=1):
        self.path = path
        self.downsample = max(1, int(downsample))
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError('could not open video: %s' % path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0
        self._cache = OrderedDict()

    def get(self, idx):
        """Frame `idx` as an (H, W, 3) uint8 RGB array, or None."""
        idx = int(idx)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        if idx < self._next or idx > self._next + self.MAX_GRAB:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        while self._next < idx:
            if not self.cap.grab():
                return None
            self._next += 1

        ok, bgr = self.cap.read()
        self._next = idx + 1
        if not ok or bgr is None:
            return None
        if self.downsample > 1:
            bgr = bgr[::self.downsample, ::self.downsample]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._cache[idx] = rgb
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return rgb

    def close(self):
        self.cap.release()


# --------------------------------------------------------------------------- #
#  QT ENUM SHIM  (PyQt5/PySide2 flat enums vs PyQt6/PySide6 scoped ones)       #
# --------------------------------------------------------------------------- #

def _qt(name):
    if hasattr(QtCore.Qt, name):
        return getattr(QtCore.Qt, name)
    for holder in vars(QtCore.Qt).values():
        if isinstance(holder, type) and hasattr(holder, name):
            return getattr(holder, name)
    raise AttributeError('Qt enum not found: %s' % name)


K_SPACE = _qt('Key_Space')
K_LEFT = _qt('Key_Left')
K_RIGHT = _qt('Key_Right')
K_UP = _qt('Key_Up')
K_DOWN = _qt('Key_Down')
K_ESC = _qt('Key_Escape')
M_SHIFT = _qt('ShiftModifier')


# --------------------------------------------------------------------------- #
#  MAIN WINDOW                                                                 #
# --------------------------------------------------------------------------- #

class SpikeVideoGUI(QtWidgets.QMainWindow):

    def __init__(self, video_path, rates, scale, rows, cluster_ids, events,
                 fps, spike_t_window=250, norm_across_cells=True,
                 playback_speed=1.0, video_downsample=1, title=None,
                 spike_times=None, n_frames=None, display_mode='raster',
                 rates_fn=None):
        super(SpikeVideoGUI, self).__init__()

        self.video = VideoReader(video_path, downsample=video_downsample)
        self.rates = rates
        self.scale = scale
        self.spike_times = spike_times
        self.rates_fn = rates_fn          # deferred prepare_rates(), so raster
        self.display_mode = display_mode  # mode does not pay for normalization
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cluster_ids = np.asarray(cluster_ids)
        self.events = events                       # {name: (onsets, offsets)}
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.window = int(spike_t_window)
        self.norm_across_cells = bool(norm_across_cells)
        self.speed = float(playback_speed)

        n_spike_frames = (n_frames if n_frames is not None
                          else self.rates.shape[1])
        self.n_frames = (min(n_spike_frames, self.video.n_frames)
                         if self.video.n_frames > 0 else n_spike_frames)
        if 0 < self.video.n_frames != n_spike_frames:
            warnings.warn('video has %d frames but aligned_spikes has %d '
                          'columns -- using the shorter (%d)'
                          % (self.video.n_frames, n_spike_frames,
                             self.n_frames))

        self.frame = 0
        self.playing = False
        self.grab_mode = False
        self._grab_anchor = None
        self._play_origin = 0
        self._clock = QtCore.QElapsedTimer()
        self.selected_row = None      # display-row index into self.rows, or None
        self._win = None              # last heatmap array + levels, for the
        self._win_levels = None       # highlight overlay to reuse without
                                      # recomputing on every click

        self._build_ui(title or os.path.basename(video_path))
        self.set_frame(0)

    # ---------------------------------------------------------------- UI ----

    def _build_ui(self, title):
        self.setWindowTitle('spike video  |  %s' % title)

        splitter = QtWidgets.QSplitter(_qt('Vertical'))
        self.setCentralWidget(splitter)

        # --- video ----------------------------------------------------------
        self.vid_widget = pg.GraphicsLayoutWidget()
        self.vid_widget.setFocusPolicy(_qt('NoFocus'))
        self.vid_vb = self.vid_widget.addViewBox()
        self.vid_vb.setAspectLocked(True)
        self.vid_vb.invertY(True)
        self.vid_vb.setMouseEnabled(False, False)
        self.vid_vb.setMenuEnabled(False)
        self.vid_img = pg.ImageItem(axisOrder='row-major')
        self.vid_vb.addItem(self.vid_img)
        splitter.addWidget(self.vid_widget)

        self.vid_widget.setMouseTracking(True)
        self.vid_widget.viewport().setMouseTracking(True)
        self.vid_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # --- legend + ethogram + heatmap ------------------------------------
        bottom_pane = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_pane)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(2)

        # colored-text legend, indented to sit over the ethogram's plot area
        self.legend = QtWidgets.QLabel('&nbsp;&nbsp;&nbsp;&nbsp;'.join(
            "<span style='color:%s'>%s</span>"
            % (_css_color(spec['color']), spec['label'])
            for spec in EVENT_SPEC.values()))
        self.legend.setAlignment(_qt('AlignCenter'))
        self.legend.setStyleSheet('font-weight: bold; font-size: 11pt;')
        bottom_layout.addWidget(self.legend)

        self.bottom = pg.GraphicsLayoutWidget()
        self.bottom.setFocusPolicy(_qt('NoFocus'))
        bottom_layout.addWidget(self.bottom)

        # one shared ethogram row: events never overlap in time, so color
        # alone identifies them and the heatmap gets the rest of the height
        self.eth_plot = self.bottom.addPlot(row=0, col=0)
        self.eth_plot.setMouseEnabled(False, False)
        self.eth_plot.setMenuEnabled(False)
        self.eth_plot.hideAxis('bottom')
        self.eth_plot.setYRange(0, 1, padding=0)
        self.eth_items = OrderedDict()
        for name, spec in EVENT_SPEC.items():
            item = pg.PlotCurveItem(pen=pg.mkPen(spec['color'], width=8),
                                    connect='pairs')
            self.eth_plot.addItem(item)
            self.eth_items[name] = item
        self.eth_plot.getAxis('left').setTicks([[]])     # keep the width, drop
        self.eth_plot.getAxis('left').setWidth(AXIS_WIDTH)   # the labels
        self.eth_plot.addItem(pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen((220, 20, 20), width=3)))
        self.bottom.ci.layout.setRowFixedHeight(0, 9)

        self.hm_plot = self.bottom.addPlot(row=1, col=0)
        self.hm_plot.setMouseEnabled(False, False)
        self.hm_plot.setMenuEnabled(False)
        self.hm_plot.invertY(True)
        self.hm_plot.setLabel('bottom', 'time from current frame (s)')
        self.hm_plot.setLabel('left', 'cell (phy ID)')
        self.hm_plot.getAxis('left').setWidth(AXIS_WIDTH)
        self.eth_plot.setXLink(self.hm_plot)

        self.hm_img = pg.ImageItem(axisOrder='row-major')
        ramp = np.linspace(255, 0, 256).astype(np.uint8)   # white min, black max
        self.hm_img.setLookupTable(np.repeat(ramp[:, None], 3, axis=1))
        self.hm_plot.addItem(self.hm_img)
        self.hm_plot.setYRange(0, len(self.rows), padding=0)

        # single-row overlay for the selected cell: same data, white -> red
        self.highlight_img = pg.ImageItem(axisOrder='row-major')
        white_to_red = np.stack([
            np.full(256, 255, dtype=np.uint8),      # R stays at 255
            np.linspace(255, 0, 256).astype(np.uint8),  # G falls to 0
            np.linspace(255, 0, 256).astype(np.uint8),  # B falls to 0
        ], axis=1)
        self.highlight_img.setLookupTable(white_to_red)
        self.highlight_img.setZValue(10)
        self.highlight_img.setVisible(False)
        self.hm_plot.addItem(self.highlight_img)

        # raster mode: every visible spike in one curve item (one draw call),
        # with the selected cell's spikes split into a second, red one
        self.raster_item = pg.PlotCurveItem(
            pen=pg.mkPen('k', width=1), connect='pairs')
        self.raster_item.setZValue(5)
        self.hm_plot.addItem(self.raster_item)
        self.raster_sel_item = pg.PlotCurveItem(
            pen=pg.mkPen('r', width=1), connect='pairs')
        self.raster_sel_item.setZValue(11)
        self.hm_plot.addItem(self.raster_sel_item)

        self._apply_window_geometry()

        self.center_line = pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen((220, 20, 20), width=3))
        self.center_line.setZValue(20)
        self.hm_plot.addItem(self.center_line)

        self.bottom.scene().sigMouseClicked.connect(self._on_heatmap_click)

        self._set_cell_ticks()
        splitter.addWidget(bottom_pane)
        splitter.setSizes([560, 520])

        # --- status ---------------------------------------------------------
        self.status = QtWidgets.QLabel('')
        self.status.setStyleSheet('font-family: monospace;')
        self.statusBar().addWidget(self.status)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.resize(1500, 1040)
        raster = (self.display_mode == 'raster')
        self.raster_item.setVisible(raster)
        self.raster_sel_item.setVisible(raster)
        self.hm_img.setVisible(not raster)
        self.setFocusPolicy(_qt('StrongFocus'))
        self.setFocus()

    def _row_transform(self, y_offset):
        half = (self.window + 0.5) * self.dt
        tr = QtGui.QTransform()
        tr.translate(-half, y_offset)
        tr.scale(self.dt, 1.0)
        return tr

    def _apply_window_geometry(self):
        half = (self.window + 0.5) * self.dt
        self.hm_img.setTransform(self._row_transform(0.0))
        if self.selected_row is not None:
            self.highlight_img.setTransform(
                self._row_transform(self.selected_row))
        self.hm_plot.setXRange(-half, half, padding=0)

    def _set_cell_ticks(self):
        axis = self.hm_plot.getAxis('left')
        n = len(self.rows)
        step = 1 if n <= 60 else int(np.ceil(n / 30.0))
        axis.setTicks([[(i + 0.5, str(int(self.cluster_ids[self.rows[i]])))
                        for i in range(0, n, step)]])

    # ------------------------------------------------------------ display ---

    def _heatmap_window(self, center):
        lo, hi = center - self.window, center + self.window + 1
        a, b = max(0, lo), min(self.n_frames, hi)
        win = self.rates[:, a:b][self.rows]
        win, levels = normalize_window(win, self.scale, self.rows,
                                       self.norm_across_cells)
        if a > lo or b < hi:                  # pad session edges with "min"
            pad = np.full((len(self.rows), hi - lo), levels[0], dtype=np.float32)
            pad[:, a - lo:a - lo + (b - a)] = win
            win = pad
        return np.ascontiguousarray(win, dtype=np.float32), levels

    def _update_ethogram(self, center):
        lo, hi = center - self.window, center + self.window
        half = (self.window + 0.5) * self.dt
        for name, item in self.eth_items.items():
            onsets, offsets = self.events[name]
            if onsets.size == 0:
                item.setData([], [])
                continue
            i0 = np.searchsorted(offsets, lo, side='left')
            i1 = np.searchsorted(onsets, hi, side='right')
            if i1 <= i0:
                item.setData([], [])
                continue
            x0 = np.clip((onsets[i0:i1] - center) * self.dt, -half, half)
            x1 = np.clip((offsets[i0:i1] - center) * self.dt, -half, half)
            x1 = np.maximum(x1, x0 + 1.5 * self.dt)   # keep 1-frame events visible
            xs = np.empty(2 * x0.size)
            xs[0::2], xs[1::2] = x0, x1
            item.setData(xs, np.full(2 * x0.size, 0.5), connect='pairs')

    def set_frame(self, idx):
        self.frame = int(np.clip(idx, 0, self.n_frames - 1))

        rgb = self.video.get(self.frame)
        if rgb is not None:
            first = self.vid_img.image is None
            self.vid_img.setImage(rgb, autoLevels=False, levels=(0, 255))
            if first:
                self.vid_vb.autoRange(padding=0)

        if self.display_mode == 'raster':
            self._update_raster(self.frame)
        else:
            win, levels = self._heatmap_window(self.frame)
            self.hm_img.setImage(win, autoLevels=False, levels=levels)
            self._win, self._win_levels = win, levels
            self._update_highlight()
        self._update_ethogram(self.frame)
        self._update_status()

    def _update_raster(self, center):
        """Draw every spike in the window as a tick the height of its row."""
        t0 = center * self.dt
        half = (self.window + 0.5) * self.dt
        xs, ys, sel_xs, sel_ys = [], [], [], []
        for i, row in enumerate(self.rows):
            st = self.spike_times[row]
            a = np.searchsorted(st, t0 - half)
            b = np.searchsorted(st, t0 + half)
            if b <= a:
                continue
            t = st[a:b] - t0
            x = np.repeat(t, 2)
            y = np.tile([float(i), i + RASTER_TICK_HEIGHT], t.size)
            if i == self.selected_row:
                sel_xs.append(x)
                sel_ys.append(y)
            else:
                xs.append(x)
                ys.append(y)

        for item, xl, yl in ((self.raster_item, xs, ys),
                             (self.raster_sel_item, sel_xs, sel_ys)):
            if xl:
                item.setData(np.concatenate(xl), np.concatenate(yl),
                             connect='pairs')
            else:
                item.setData([], [])

    def _ensure_rates(self):
        """Normalized rates are only computed if the heatmap is actually used."""
        if self.rates is None and self.rates_fn is not None:
            self.status.setText('computing normalized firing rates ...')
            QtWidgets.QApplication.processEvents()
            self.rates, self.scale = self.rates_fn()
        return self.rates is not None

    def set_display_mode(self, mode):
        if mode == 'heatmap' and not self._ensure_rates():
            self.status.setText('no firing rates available for the heatmap')
            return
        self.display_mode = mode
        raster = (mode == 'raster')
        self.raster_item.setVisible(raster)
        self.raster_sel_item.setVisible(raster)
        self.hm_img.setVisible(not raster)
        if raster:
            self.highlight_img.setVisible(False)
        self.set_frame(self.frame)

    def _update_highlight(self):
        """Redraw the selected row's overlay from the current heatmap window,
        without recomputing it."""
        if self.selected_row is None or self._win is None:
            self.highlight_img.setVisible(False)
            return
        row = self._win[self.selected_row:self.selected_row + 1]
        self.highlight_img.setImage(row, autoLevels=False,
                                    levels=self._win_levels)
        self.highlight_img.setVisible(True)

    def _on_heatmap_click(self, ev):
        if ev.button() != _qt('LeftButton'):
            return
        if not self.hm_plot.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = self.hm_plot.vb.mapSceneToView(ev.scenePos())
        row = int(np.floor(pt.y()))
        if not (0 <= row < len(self.rows)):
            return
        if self.selected_row == row:
            self.selected_row = None
        else:
            self.selected_row = row
            self.highlight_img.setTransform(self._row_transform(row))
        if self.display_mode == 'raster':
            self._update_raster(self.frame)
        else:
            self._update_highlight()
        self._update_status()

    def _update_status(self):
        t = self.frame / self.fps
        if self.display_mode == 'raster':
            norm = 'raster'
        elif self.norm_across_cells:
            norm = 'per-cell, rescaled'
        else:
            norm = 'shared scale (%s)' % self.scale['units']
        here = [EVENT_SPEC[n]['label'] for n, (on, off) in self.events.items()
                if on.size and np.any((on <= self.frame) & (off >= self.frame))]
        if self.selected_row is not None:
            cell = '   cell %d selected' % int(
                self.cluster_ids[self.rows[self.selected_row]])
        else:
            cell = ''
        self.status.setText(
            'frame %7d / %d   %02d:%02d:%05.2f   %s   view: %s   '
            'window: \u00b1%d fr (%.2f s)   %.2fx%s%s%s'
            % (self.frame, self.n_frames - 1,
               int(t // 3600), int(t % 3600 // 60), t % 60,
               'PLAY ' if self.playing else 'PAUSE', norm,
               self.window, self.window * self.dt, self.speed,
               '   GRAB' if self.grab_mode else '',
               ('   << %s >>' % ', '.join(here)) if here else '',
               cell))

    # ----------------------------------------------------------- playback ---

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._play_origin = self.frame
            self._clock.restart()
            self.timer.start(max(1, int(1000.0 / (self.fps * self.speed))))
        else:
            self.timer.stop()
        self._update_status()

    def _on_tick(self):
        if not self.playing:
            return
        target = self._play_origin + int(
            self._clock.elapsed() / 1000.0 * self.fps * self.speed)
        if target >= self.n_frames:
            self.playing = False
            self.timer.stop()
            target = self.n_frames - 1
        if target != self.frame:
            self.set_frame(target)

    def step(self, n):
        if self.playing:
            self.toggle_play()
        self.set_frame(self.frame + n)

    def jump_to_event(self, name, forward=True):
        onsets = self.events[name][0]
        if onsets.size == 0:
            self.status.setText("no '%s' events loaded" % name)
            return
        if self.playing:
            self.toggle_play()
        if forward:
            i = np.searchsorted(onsets, self.frame + 1, side='left')
            if i >= onsets.size:
                self.status.setText('last %s reached' % name)
                return
        else:
            i = np.searchsorted(onsets, self.frame, side='left') - 1
            if i < 0:
                self.status.setText('first %s reached' % name)
                return
        self.set_frame(int(onsets[i]))

    # --------------------------------------------------------- video view ---

    def zoom(self, factor):
        self.vid_vb.scaleBy((factor, factor),
                            center=self.vid_vb.viewRect().center())

    def reset_view(self):
        self.vid_vb.autoRange(padding=0)

    def toggle_grab(self):
        self.grab_mode = not self.grab_mode
        self._grab_anchor = None
        self.vid_widget.setCursor(QtGui.QCursor(
            _qt('OpenHandCursor') if self.grab_mode else _qt('ArrowCursor')))
        self._update_status()

    def _on_mouse_moved(self, scene_pos):
        if not self.grab_mode:
            self._grab_anchor = None
            return
        if not self.vid_vb.sceneBoundingRect().contains(scene_pos):
            self._grab_anchor = None
            return
        pt = self.vid_vb.mapSceneToView(scene_pos)
        if self._grab_anchor is not None:
            d = pt - self._grab_anchor
            self.vid_vb.translateBy(x=-d.x(), y=-d.y())
        self._grab_anchor = self.vid_vb.mapSceneToView(scene_pos)

    # ---------------------------------------------------------- keyboard ---

    def keyPressEvent(self, ev):
        key = ev.key()
        shift = bool(ev.modifiers() & M_SHIFT)
        big = 10 if shift else 1

        if key == K_SPACE:
            return self.toggle_play()
        if key == K_RIGHT:
            return self.step(big)
        if key == K_LEFT:
            return self.step(-big)
        if key in (K_UP, K_DOWN):
            factor = 0.8 if key == K_UP else 1.25
            return self.set_window(max(5, int(round(self.window * factor))))
        if key == K_ESC:
            return self.close()

        low = ev.text().lower()
        if low in ('+', '='):
            self.zoom(0.8)
        elif low in ('-', '_'):
            self.zoom(1.25)
        elif low == 'h':
            self.toggle_grab()
        elif low == 'z':
            self.reset_view()
        elif low == 'n':
            if self.display_mode == 'raster':
                self.status.setText(
                    "normalization applies to the heatmap -- press 'd'")
            else:
                self.norm_across_cells = not self.norm_across_cells
                self.set_frame(self.frame)
        elif low == 'd':
            self.set_display_mode(
                'heatmap' if self.display_mode == 'raster' else 'raster')
        elif low == '[':
            self.set_speed(self.speed / 1.5)
        elif low == ']':
            self.set_speed(self.speed * 1.5)
        elif low == 'q':
            self.close()
        elif low in [s['key'] for s in EVENT_SPEC.values()]:
            name = next(n for n, s in EVENT_SPEC.items() if s['key'] == low)
            self.jump_to_event(name, forward=not shift)
        else:
            super(SpikeVideoGUI, self).keyPressEvent(ev)

    def set_speed(self, speed):
        self.speed = float(np.clip(speed, 0.05, 20.0))
        if self.playing:
            self._play_origin = self.frame
            self._clock.restart()
            self.timer.start(max(1, int(1000.0 / (self.fps * self.speed))))
        self._update_status()

    def set_window(self, window):
        self.window = int(window)
        self._apply_window_geometry()
        self.set_frame(self.frame)

    def closeEvent(self, ev):
        self.timer.stop()
        self.video.close()
        super(SpikeVideoGUI, self).closeEvent(ev)


# --------------------------------------------------------------------------- #
#  ENTRY POINT                                                                 #
# --------------------------------------------------------------------------- #

def launch(vid_dir, data_dir, cam_id,
           norm_across_cells=True,
           spike_t_window=250,
           smoothing=None,
           cell_ids=None,
           cell_sort=None,
           # --- optional ---
           repo_root=None,
           ks_dir=None,
           only_good=True,
           fps=50,
           pop_norm=True,
           std_reg=0.6,
           baseline_window=30,
           baseline_units='minutes',
           use_repo_moving_avg=True,
           refined_events=False,
           display_mode='raster',
           playback_speed=1.0,
           video_downsample=1,
           events=None,
           cluster_ids=None,
           clip_pct=99.5):
    """Build the session and open the viewer.  See run_spike_video_gui.py."""
    add_repo_to_path(repo_root)

    video_path = os.path.join(vid_dir, '%s.avi' % cam_id)
    if not os.path.exists(video_path):
        raise IOError('no video at %s' % video_path)
    spikes_path = os.path.join(data_dir, 'aligned_spikes.npy')
    if not os.path.exists(spikes_path):
        raise IOError('no aligned_spikes.npy in %s' % data_dir)

    aligned_spikes = np.load(spikes_path)
    n_cells, n_frames = aligned_spikes.shape
    print('session: %d cells x %d frames @ %g Hz  (%.1f min)'
          % (n_cells, n_frames, fps, n_frames / fps / 60.0))

    if cluster_ids is None:
        cluster_ids = load_cluster_ids(n_cells, data_dir=data_dir,
                                       ks_dir=ks_dir, only_good=only_good)
    cluster_ids = np.asarray(cluster_ids).ravel()

    if events is None:
        events = load_events(data_dir, n_frames, repo_root=repo_root,
                             refined=refined_events)
    events = OrderedDict(
        (name, _clean_event(events.get(name, (np.zeros(0), np.zeros(0)))[0],
                            events.get(name, (np.zeros(0), np.zeros(0)))[1],
                            n_frames))
        for name in EVENT_SPEC)
    print('  events: %s' % ', '.join('%s=%d' % (k, v[0].size)
                                     for k, v in events.items()))

    # --- which rows to show, and in what order ------------------------------
    if cell_sort is not None:
        order = np.asarray(cell_sort, dtype=np.int64).ravel()
        if order.size != n_cells or np.unique(order).size != order.size:
            warnings.warn('cell_sort should be a permutation of the %d rows of '
                          'aligned_spikes.npy; using it as given' % n_cells)
    else:
        order = np.arange(n_cells, dtype=np.int64)

    if cell_ids is not None:
        wanted = np.asarray(cell_ids).ravel()
        have = set(cluster_ids.tolist())
        missing = [c for c in wanted if c not in have]
        if missing:
            raise ValueError('cluster IDs not in this session: %s' % missing)
        if cell_sort is None:
            order = np.array([int(np.flatnonzero(cluster_ids == c)[0])
                              for c in wanted], dtype=np.int64)
        else:
            order = order[np.isin(cluster_ids[order], wanted)]
    rows = order
    print('  showing %d cells' % len(rows))

    spike_times = build_spike_times(aligned_spikes, cluster_ids, fps=fps)
    print('  %d spikes total (%.1f Hz mean)'
          % (sum(t.size for t in spike_times),
             sum(t.size for t in spike_times) / n_cells / (n_frames / fps)))

    def make_rates():
        return prepare_rates(aligned_spikes, fps=fps, smoothing=smoothing,
                             std_reg=std_reg,
                             baseline_window=baseline_window,
                             baseline_units=baseline_units,
                             pop_norm=pop_norm,
                             use_repo_moving_avg=use_repo_moving_avg,
                             clip_pct=clip_pct)

    # the raster does not need normalized rates, so only pay for them if the
    # heatmap is the starting view; 'd' computes them on first use otherwise
    if display_mode == 'heatmap':
        rates, scale = make_rates()
    else:
        rates, scale = None, None

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=False)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = SpikeVideoGUI(video_path, rates, scale, rows, cluster_ids, events,
                        fps=fps, spike_t_window=spike_t_window,
                        norm_across_cells=norm_across_cells,
                        playback_speed=playback_speed,
                        video_downsample=video_downsample,
                        spike_times=spike_times, n_frames=n_frames,
                        display_mode=display_mode, rates_fn=make_rates,
                        title='%s  |  %s' % (os.path.basename(
                            os.path.dirname(os.path.join(data_dir, ''))), cam_id))
    gui.show()
    gui.raise_()
    gui.activateWindow()
    return (getattr(app, 'exec', None) or app.exec_)()
