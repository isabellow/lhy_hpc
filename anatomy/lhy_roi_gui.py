#!/usr/bin/env python
"""
lhy_roi_gui.py
==============

Flip through coronal histology sections for one bird and mark, per section:

    * whether the lateral hypothalamus is present         (y / n / ?)
    * the midline, as a dorsal -> ventral segment         (ML = 0, DV direction)
    * the dorsal brain surface, as a polyline             (DV zero)
    * one or more elliptical LHy ROIs
    * the probe scar, traced as a polyline per shank      (tip + track slope)
    * the section's position on its slide, lost sections before it, and
      whether it was mounted flipped

Annotations are saved as raw pixel geometry in a JSON file; brain coordinates,
the scar track fit and LHy distances come from lhy_roi_tools.load_lhy_rois().
Launch it from run_lhy_roi_gui.py.

Dependencies
------------
    numpy, pyqtgraph, a Qt binding, nd2 (pip install nd2), pandas
"""

import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import lhy_roi_tools as lrt


# --------------------------------------------------------------------------- #
#  QT ENUM SHIM  (same as spike_video_gui)                                     #
# --------------------------------------------------------------------------- #

def _qt(name):
    if hasattr(QtCore.Qt, name):
        return getattr(QtCore.Qt, name)
    for holder in vars(QtCore.Qt).values():
        if isinstance(holder, type) and hasattr(holder, name):
            return getattr(holder, name)
    raise AttributeError('Qt enum not found: %s' % name)


def _event_type(name):
    if hasattr(QtCore.QEvent, name):
        return getattr(QtCore.QEvent, name)
    return getattr(QtCore.QEvent.Type, name)


K = {n: _qt('Key_' + n) for n in
     ('Left', 'Right', 'PageUp', 'PageDown', 'Escape', 'Return', 'Enter',
      'Delete', 'Backspace', 'S')}
M_CTRL = _qt('ControlModifier')
LEFT_BUTTON = _qt('LeftButton')
EV_KEYPRESS = _event_type('KeyPress')

STATUS_COLORS = {None: (90, 90, 90), 'yes': (40, 130, 60),
                 'no': (130, 40, 40), 'unsure': (170, 110, 20)}

SHANK_COLORS = [(255, 70, 70), (70, 170, 255), (255, 150, 0), (200, 90, 255),
                (0, 220, 200), (255, 255, 255), (160, 255, 60), (255, 120, 200)]

PEN = dict(
    ellipse=pg.mkPen((255, 220, 0), width=2),
    ellipse_sel=pg.mkPen((0, 220, 255), width=3),
    midline=pg.mkPen((255, 80, 255), width=2),
    surface=pg.mkPen((80, 255, 120), width=2),
)

HELP = """<b>keys</b> (click the image first)<br>
&larr; / &rarr;, PgUp/PgDn &nbsp; prev / next section &nbsp;&nbsp; u &nbsp; next unreviewed<br>
y / n / ? &nbsp; LHy present / absent / unsure<br>
m &nbsp; midline at cursor (<b>D</b> handle = dorsal)<br>
s &nbsp; trace dorsal surface &nbsp;&nbsp; t &nbsp; trace scar for the current shank<br>
&nbsp;&nbsp;&nbsp;click points, Enter (or the same key) finishes, Esc cancels<br>
&nbsp;&nbsp;&nbsp;click a line segment to add a vertex, right-click a vertex to remove<br>
1-8 &nbsp; current shank<br>
e &nbsp; ellipse at cursor; click to select<br>
Del &nbsp; selected ellipse, else the scar under the cursor<br>
[ / ] &nbsp; move section earlier / later on its slide<br>
x &nbsp; flipped &nbsp;&nbsp; a &nbsp; AC reference = this section<br>
v &nbsp; copy midline/surface/ellipses from previous section<br>
h &nbsp; hide overlays &nbsp;&nbsp; f &nbsp; fit view &nbsp;&nbsp; Ctrl+S save &nbsp;&nbsp; q &nbsp; quit"""


# --------------------------------------------------------------------------- #
#  IMAGE IO                                                                    #
# --------------------------------------------------------------------------- #

def _reduce_to_yx(arr, dims):
    """Max-project every axis that is not Y, X or S (RGB samples)."""
    dims = [d.upper() for d in dims]
    keep = [d for d in dims if d in ('Y', 'X', 'S')]
    for d in reversed(list(dims)):
        if d not in ('Y', 'X', 'S'):
            arr = arr.max(axis=dims.index(d))
            dims.remove(d)
    order = [dims.index(d) for d in ('Y', 'X', 'S') if d in keep]
    return np.transpose(arr, order)


def read_image_file(path):
    """Returns (array Y x X [x 3], pixel_size_um (x, y) or None)."""
    ext = os.path.splitext(path)[1].lower()
    px = None
    if ext == '.nd2':
        import nd2
        with nd2.ND2File(path) as f:
            arr = _reduce_to_yx(np.asarray(f.asarray()), list(f.sizes.keys()))
            try:
                vs = f.voxel_size()
                if vs.x > 0 and vs.y > 0:
                    px = (float(vs.x), float(vs.y))
            except Exception as err:            # metadata varies by scope
                warnings.warn('%s: no pixel size in metadata (%s)' % (path, err))
    elif ext in ('.tif', '.tiff'):
        import tifffile
        arr = np.squeeze(tifffile.imread(path))
    else:
        arr = np.squeeze(np.load(path))
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr.max(axis=0)
    return arr, px


def _block_mean(arr, ds, rows_per_chunk=256):
    """Downsample by ds with block means, a band of rows at a time (low RAM)."""
    if ds <= 1:
        return arr.astype(np.float32)
    h, w = (arr.shape[0] // ds) * ds, (arr.shape[1] // ds) * ds
    out = np.empty((h // ds, w // ds) + arr.shape[2:], dtype=np.float32)
    step = max(1, rows_per_chunk // ds) * ds
    for r in range(0, h, step):
        band = arr[r:r + step, :w].astype(np.float32)
        shape = (band.shape[0] // ds, ds, w // ds, ds) + band.shape[2:]
        out[r // ds:(r + band.shape[0]) // ds] = band.reshape(shape).mean(axis=(1, 3))
    return out


def display_levels(img):
    """Contrast from tissue pixels -- stitched scans pad with zeros."""
    sample = img[::max(1, img.shape[0] // 512), ::max(1, img.shape[1] // 512)]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, [1, 99.7])
    return float(lo), float(max(hi, lo + 1))


class ImageCache(object):
    """Reads section images, downsamples for display, caches in RAM and disk."""

    def __init__(self, hist_dir, downsample=4, cache_dir=None, max_items=3):
        self.hist_dir = hist_dir
        self.ds = max(1, int(downsample))
        self.cache_dir = cache_dir
        self.max_items = max_items
        self._mem = OrderedDict()

    def get(self, fname):
        if fname in self._mem:
            self._mem.move_to_end(fname)
            return self._mem[fname]
        out = self._from_disk_cache(fname)
        if out is None:
            path = os.path.join(self.hist_dir, fname)
            if not os.path.exists(path):
                raise IOError('missing image %s' % path)
            arr, px = read_image_file(path)
            out = (_block_mean(arr, self.ds), tuple(arr.shape[:2]), px)
            del arr
            self._to_disk_cache(fname, out)
        self._mem[fname] = out
        while len(self._mem) > self.max_items:
            self._mem.popitem(last=False)
        return out

    def _cache_path(self, fname):
        return os.path.join(self.cache_dir, '%s.ds%d.npz' % (fname, self.ds))

    def _from_disk_cache(self, fname):
        if not self.cache_dir:
            return None
        p = self._cache_path(fname)
        if not os.path.exists(p):
            return None
        z = np.load(p)
        px = tuple(z['px']) if np.all(np.isfinite(z['px'])) else None
        return z['img'], tuple(int(v) for v in z['full_shape']), px

    def _to_disk_cache(self, fname, out):
        if not self.cache_dir:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            img, shape, px = out
            np.savez(self._cache_path(fname), img=img, full_shape=shape,
                     px=np.asarray(px if px else (np.nan, np.nan)))
        except OSError as err:
            warnings.warn('could not write image cache: %s' % err)


def _dist_to_polyline(p, pts):
    """Distance from point p to a polyline (both in the same units)."""
    pts = np.asarray(pts, float)
    p = np.asarray(p, float)
    if len(pts) == 1:
        return float(np.hypot(*(pts[0] - p)))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((proj - p) ** 2).sum(1)).min())


# --------------------------------------------------------------------------- #
#  MAIN WINDOW                                                                 #
# --------------------------------------------------------------------------- #

class LHyROIGUI(QtWidgets.QMainWindow):

    def __init__(self, ann, ann_path, images, pixel_size_override=None,
                 default_ellipse_axes_um=(250., 200.), dv_mode='local',
                 ref_ml_um=None, keep_levels=True, export_csv=True,
                 shank_labels=('A',)):
        super(LHyROIGUI, self).__init__()
        self.ann = ann
        self.ann_path = ann_path
        self.images = images
        self.pixel_size_override = pixel_size_override
        self.default_axes_um = default_ellipse_axes_um
        self.dv_mode = dv_mode
        self.ref_ml_um = ref_ml_um
        self.export_csv = export_csv
        self.shank_labels = list(shank_labels)
        for s in ann['sections'].values():           # labels already in the file
            for scar in s.get('scars', []):
                if scar.get('shank') not in self.shank_labels:
                    self.shank_labels.append(scar.get('shank'))

        self.keys = lrt.assign_section_indices(ann)
        self.cur = 0
        self.dirty = False
        self.mouse_xy = None
        self.overlays_visible = True
        self.drawing = None                # None | 'surface' | 'scar'
        self.draft = []
        self._geom = None
        self._implant = None

        self.midline_roi = None
        self.midline_labels = []
        self.surface_roi = None
        self.ellipse_rois = []
        self.selected = None
        self.scar_rois = []

        self._build_ui(keep_levels)
        QtWidgets.QApplication.instance().installEventFilter(self)
        first = next((i for i, k in enumerate(self.keys)
                      if self.sec(k)['lhy'] is None), 0)
        self.goto(first)

    # ------------------------------------------------------------ helpers --
    def sec(self, key=None):
        return self.ann['sections'][key if key is not None else self.keys[self.cur]]

    def _mark_dirty(self, implant=False):
        self.dirty = True
        self._geom = None
        if implant:
            self._implant = None

    def _reorder(self):
        key = self.keys[self.cur]
        self.keys = lrt.assign_section_indices(self.ann)
        self.cur = self.keys.index(key)
        self._mark_dirty()
        self._refresh_list()
        self._refresh_panel()

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self, keep_levels):
        self.setWindowTitle('LHy ROIs  |  %s' % self.ann.get('bird'))
        splitter = QtWidgets.QSplitter(_qt('Horizontal'))
        self.setCentralWidget(splitter)

        # --- image ----------------------------------------------------------
        self.view = pg.GraphicsLayoutWidget()
        self.view.setFocusPolicy(_qt('StrongFocus'))
        self.vb = self.view.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.vb.addItem(self.img)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.view.addItem(self.hist, row=0, col=1)
        self.draft_item = pg.PlotCurveItem()
        self.draft_item.setZValue(30)
        self.vb.addItem(self.draft_item)
        self.flip_label = pg.TextItem('FLIPPED', color=(255, 60, 60), anchor=(0, 0))
        self.flip_label.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold
                                            if hasattr(QtGui.QFont, 'Bold')
                                            else QtGui.QFont.Weight.Bold))
        self.flip_label.setZValue(40)
        self.vb.addItem(self.flip_label)
        self.view.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.view.scene().sigMouseClicked.connect(self._on_scene_click)
        splitter.addWidget(self.view)

        # --- side panel -----------------------------------------------------
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        self.section_list = QtWidgets.QListWidget()
        self.section_list.setFocusPolicy(_qt('NoFocus'))
        self.section_list.itemClicked.connect(
            lambda item: self.goto(self.section_list.row(item)))
        lay.addWidget(QtWidgets.QLabel(
            '<b>sections</b>  (index  slide #order  region  flags)'))
        lay.addWidget(self.section_list, stretch=3)

        # series parameters
        series = self.ann['series']
        box = QtWidgets.QGroupBox('series (whole bird)')
        form = QtWidgets.QFormLayout(box)
        self.thick = QtWidgets.QDoubleSpinBox()
        self.thick.setRange(0, 1000)
        self.thick.setDecimals(1)
        self.thick.setSuffix(' um')
        self.thick.setSpecialValueText('unset')
        self.thick.setValue(series['section_thickness_um'] or 0)
        self.thick.valueChanged.connect(
            lambda v: self._set_series('section_thickness_um', v or None))
        form.addRow('thickness', self.thick)

        self.interval = QtWidgets.QSpinBox()
        self.interval.setRange(1, 50)
        self.interval.setValue(int(series['section_interval']))
        self.interval.valueChanged.connect(
            lambda v: self._set_series('section_interval', int(v)))
        form.addRow('interval', self.interval)

        self.ap_sign = QtWidgets.QComboBox()
        self.ap_sign.addItems(['slides run posterior -> anterior',
                               'slides run anterior -> posterior'])
        self.ap_sign.setCurrentIndex(0 if series['ap_sign'] >= 0 else 1)
        self.ap_sign.currentIndexChanged.connect(
            lambda i: self._set_series('ap_sign', 1 if i == 0 else -1))
        form.addRow('AP order', self.ap_sign)

        self.ac_label = QtWidgets.QLabel()
        ac_btn = QtWidgets.QPushButton('set current (a)')
        ac_btn.setFocusPolicy(_qt('NoFocus'))
        ac_btn.clicked.connect(self.set_ac_here)
        ac_row = QtWidgets.QHBoxLayout()
        ac_row.addWidget(self.ac_label)
        ac_row.addWidget(ac_btn)
        form.addRow('AC section', ac_row)

        self.ac_ap = QtWidgets.QLineEdit(
            '' if series['ac_ap_um'] is None else '%g' % series['ac_ap_um'])
        self.ac_ap.setPlaceholderText('blank = unset')
        self.ac_ap.setValidator(QtGui.QDoubleValidator())
        self.ac_ap.editingFinished.connect(self._on_ac_ap_edited)
        form.addRow('AC AP from lambda (um)', self.ac_ap)

        self.implant = QtWidgets.QComboBox()
        self.implant.addItems(['auto (from scar traces)',
                               'image right (unflipped)', 'image left (unflipped)'])
        self.implant.setCurrentIndex({None: 0, 'right': 1, 'left': 2}.get(
            series.get('implant_side'), 0))
        self.implant.currentIndexChanged.connect(
            lambda i: self._set_series('implant_side', [None, 'right', 'left'][i],
                                       implant=True))
        form.addRow('implanted side', self.implant)

        self.inplane = QtWidgets.QDoubleSpinBox()
        self.inplane.setRange(0.5, 2.0)
        self.inplane.setDecimals(3)
        self.inplane.setSingleStep(0.01)
        self.inplane.setValue(float(series['inplane_scale']))
        self.inplane.valueChanged.connect(
            lambda v: self._set_series('inplane_scale', float(v)))
        form.addRow('in-plane scale', self.inplane)
        lay.addWidget(box)

        # current section
        box = QtWidgets.QGroupBox('current section')
        form = QtWidgets.QFormLayout(box)
        self.file_label = QtWidgets.QLabel()
        self.file_label.setWordWrap(True)
        form.addRow(self.file_label)

        order_row = QtWidgets.QHBoxLayout()
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 999)
        self.order_spin.valueChanged.connect(self._on_slide_order)
        self.gap_spin = QtWidgets.QSpinBox()
        self.gap_spin.setRange(0, 500)
        self.gap_spin.valueChanged.connect(self._on_gap)
        self.flip_box = QtWidgets.QCheckBox('flipped (x)')
        self.flip_box.setFocusPolicy(_qt('NoFocus'))
        self.flip_box.clicked.connect(lambda checked: self.set_flipped(checked))
        order_row.addWidget(QtWidgets.QLabel('order on slide'))
        order_row.addWidget(self.order_spin)
        order_row.addWidget(QtWidgets.QLabel('lost before'))
        order_row.addWidget(self.gap_spin)
        order_row.addWidget(self.flip_box)
        form.addRow(order_row)

        state_row = QtWidgets.QHBoxLayout()
        self.state_group = QtWidgets.QButtonGroup(self)
        for i, label in enumerate(['LHy (y)', 'no (n)', 'unsure (?)']):
            b = QtWidgets.QRadioButton(label)
            b.setFocusPolicy(_qt('NoFocus'))
            self.state_group.addButton(b, i)
            state_row.addWidget(b)
        self.state_group.buttonClicked.connect(
            lambda b: self.set_state(['yes', 'no', 'unsure'][self.state_group.id(b)]))
        form.addRow('status', state_row)

        self.shank_combo = QtWidgets.QComboBox()
        self.shank_combo.addItems(self.shank_labels)
        self.shank_combo.setFocusPolicy(_qt('NoFocus'))
        form.addRow('scar shank (1-8)', self.shank_combo)

        self.notes = QtWidgets.QLineEdit()
        self.notes.editingFinished.connect(self._on_notes_edited)
        form.addRow('notes', self.notes)

        dv_row = QtWidgets.QHBoxLayout()
        self.dv_combo = QtWidgets.QComboBox()
        self.dv_combo.addItems(['local', 'ref_ml', 'midline'])
        self.dv_combo.setCurrentText(self.dv_mode)
        self.dv_combo.currentTextChanged.connect(self._on_dv_mode)
        self.ref_ml = QtWidgets.QDoubleSpinBox()
        self.ref_ml.setRange(0, 5000)
        self.ref_ml.setSuffix(' um')
        self.ref_ml.setValue(self.ref_ml_um or 0)
        self.ref_ml.valueChanged.connect(self._on_dv_mode)
        dv_row.addWidget(self.dv_combo)
        dv_row.addWidget(self.ref_ml)
        form.addRow('DV readout', dv_row)
        lay.addWidget(box)

        help_label = QtWidgets.QLabel(HELP)
        help_label.setWordWrap(True)
        help_label.setStyleSheet('font-size: 9pt;')
        lay.addWidget(help_label)

        self.keep_levels = QtWidgets.QCheckBox('keep contrast between sections')
        self.keep_levels.setChecked(keep_levels)
        self.keep_levels.setFocusPolicy(_qt('NoFocus'))
        lay.addWidget(self.keep_levels)

        splitter.addWidget(panel)
        splitter.setSizes([1100, 420])

        self.status = QtWidgets.QLabel('')
        self.status.setStyleSheet('font-family: monospace;')
        self.statusBar().addWidget(self.status)
        self.message = QtWidgets.QLabel('')
        self.statusBar().addPermanentWidget(self.message)

        self._refresh_list()
        self._refresh_ac_label()

    # --------------------------------------------------------- navigation --
    def goto(self, i):
        if not self.keys:
            return
        self._cancel_draw()
        if hasattr(self, '_loaded_key'):
            self._pull_items()
            if self.dirty:
                self.save()
        self.cur = int(np.clip(i, 0, len(self.keys) - 1))
        s = self.sec()
        self._loaded_key = self.keys[self.cur]

        QtWidgets.QApplication.setOverrideCursor(_qt('WaitCursor'))
        try:
            img, full_shape, px = self.images.get(s['file'])
            if self.pixel_size_override is not None:
                px = self.pixel_size_override
            if px is None:
                px = (1.0, 1.0)
                self.flash('no pixel size for %s -- set pixel_size_override; '
                           'using 1 um/px' % s['file'], error=True)
            if s['pixel_size_um'] is None or list(s['pixel_size_um']) != list(px):
                s['pixel_size_um'] = [float(px[0]), float(px[1])]
                s['image_shape'] = [int(full_shape[0]), int(full_shape[1])]
                self.dirty = True
            levels = self.hist.getLevels() if (self.keep_levels.isChecked()
                                               and self.img.image is not None) else None
            if levels is None:
                levels = display_levels(img)
            self.img.setImage(img, autoLevels=False, levels=levels)
            self.img.setRect(QtCore.QRectF(0, 0, full_shape[1], full_shape[0]))
            self.vb.autoRange(padding=0.02)
        except Exception as err:
            self.img.clear()
            self.flash('could not load %s: %s' % (s['file'], err), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._load_items()
        self._refresh_panel()
        self._refresh_list()
        self.view.setFocus()

    def next_unreviewed(self):
        n = len(self.keys)
        for step in range(1, n + 1):
            j = (self.cur + step) % n
            if self.sec(self.keys[j])['lhy'] is None:
                return self.goto(j)
        self.flash('every section has been reviewed')

    # ----------------------------------------------------- overlay items --
    def _all_items(self):
        items = [self.midline_roi, self.surface_roi] + self.midline_labels + self.ellipse_rois
        for roi in self.scar_rois:
            items += [roi, roi._text]
        return [it for it in items if it is not None]

    def _clear_items(self):
        for item in self._all_items():
            self.vb.removeItem(item)
        self.midline_roi = self.surface_roi = self.selected = None
        self.midline_labels, self.ellipse_rois, self.scar_rois = [], [], []

    def _load_items(self):
        self._clear_items()
        s = self.sec()
        if s.get('midline_px'):
            self._make_midline(*s['midline_px'])
        if s.get('surface_px'):
            self._make_surface(s['surface_px'])
        for e in s.get('ellipses', []):
            self._make_ellipse(e)
        for scar in s.get('scars', []):
            self._make_scar(scar['points_px'], scar.get('shank', 'A'))
        self._set_overlays_visible(self.overlays_visible)
        self._geom = None

    def _pull_items(self):
        """Copy the current geometry of every overlay into the section dict."""
        s = self.sec(self._loaded_key)
        s['midline_px'] = self._midline_points()
        s['surface_px'] = self._poly_points(self.surface_roi)
        s['ellipses'] = [self._ellipse_dict(r) for r in self.ellipse_rois]
        s['scars'] = [dict(shank=r._shank, points_px=self._poly_points(r))
                      for r in self.scar_rois]

    def _changed(self, *_, implant=False):
        self._pull_items()
        self._mark_dirty(implant=implant)
        self._refresh_list_item(self.cur)
        self._update_status()

    # midline
    def _make_midline(self, dorsal, ventral):
        roi = pg.LineSegmentROI([list(dorsal), list(ventral)], pen=PEN['midline'])
        roi.setZValue(20)
        self.vb.addItem(roi)
        self.midline_roi = roi
        self.midline_labels = []
        for text in ('D', 'V'):
            label = pg.TextItem(text, color=(255, 80, 255), anchor=(0.5, 1.2))
            label.setZValue(21)
            self.vb.addItem(label)
            self.midline_labels.append(label)
        roi.sigRegionChanged.connect(self._place_midline_labels)
        roi.sigRegionChangeFinished.connect(lambda *_: self._changed(implant=True))
        self._place_midline_labels()

    def _midline_points(self):
        if self.midline_roi is None:
            return None
        return [[float(p.x()), float(p.y())] for p in
                (self.midline_roi.mapToParent(pos) for _, pos in
                 self.midline_roi.getLocalHandlePositions())]

    def _place_midline_labels(self, *_):
        for label, (x, y) in zip(self.midline_labels, self._midline_points()):
            label.setPos(x, y)

    # polylines
    @staticmethod
    def _poly_points(roi):
        if roi is None:
            return None
        return [[float(p.x()), float(p.y())] for p in
                (roi.mapToParent(pos) for _, pos in roi.getLocalHandlePositions())]

    def _make_surface(self, points):
        roi = pg.PolyLineROI([list(p) for p in points], closed=False,
                             pen=PEN['surface'])
        roi.setZValue(19)
        self.vb.addItem(roi)
        roi.sigRegionChangeFinished.connect(self._changed)
        self.surface_roi = roi

    def _shank_color(self, shank):
        i = self.shank_labels.index(shank) if shank in self.shank_labels else 0
        return SHANK_COLORS[i % len(SHANK_COLORS)]

    def _make_scar(self, points, shank):
        color = self._shank_color(shank)
        roi = pg.PolyLineROI([list(p) for p in points], closed=False,
                             pen=pg.mkPen(color, width=2))
        roi.setZValue(24)
        roi._shank = shank
        roi._text = pg.TextItem(shank, color=color, anchor=(-0.3, 0.5))
        roi._text.setZValue(25)
        self.vb.addItem(roi)
        self.vb.addItem(roi._text)
        roi.sigRegionChanged.connect(lambda *_: self._place_scar_label(roi))
        roi.sigRegionChangeFinished.connect(lambda *_: self._changed(implant=True))
        self.scar_rois.append(roi)
        self._place_scar_label(roi)
        return roi

    def _place_scar_label(self, roi):
        pts = self._poly_points(roi)
        if pts:
            deepest = max(pts, key=lambda p: p[1])
            roi._text.setPos(*deepest)

    # ellipses
    def _make_ellipse(self, e):
        a, b = e['axes_px']
        c = np.asarray(e['center_px'], float)
        pos = c - lrt.qt_rotation(e['angle_deg']) @ np.array([a, b])
        roi = pg.EllipseROI(list(pos), [2 * a, 2 * b], angle=e['angle_deg'],
                            pen=PEN['ellipse'])
        roi.setZValue(22)
        roi.setAcceptedMouseButtons(LEFT_BUTTON)
        roi.sigClicked.connect(self._select_roi)
        roi.sigRegionChangeFinished.connect(self._changed)
        self.vb.addItem(roi)
        self.ellipse_rois.append(roi)
        return roi

    @staticmethod
    def _ellipse_dict(roi):
        w, h = roi.size()
        c = roi.mapToParent(QtCore.QPointF(w / 2.0, h / 2.0))
        return dict(center_px=[float(c.x()), float(c.y())],
                    axes_px=[abs(float(w)) / 2.0, abs(float(h)) / 2.0],
                    angle_deg=float(roi.angle()))

    def _select_roi(self, roi, *_):
        if self.selected is not None and self.selected in self.ellipse_rois:
            self.selected.setPen(PEN['ellipse'])
        self.selected = None if roi is self.selected else roi
        if self.selected is not None:
            self.selected.setPen(PEN['ellipse_sel'])

    def _set_overlays_visible(self, visible):
        for item in self._all_items():
            item.setVisible(visible)

    # -------------------------------------------------------------- actions --
    def _px_per_um(self):
        px = self.sec()['pixel_size_um'] or (1.0, 1.0)
        return 1.0 / float(np.mean(px))

    def _cursor_or_center(self):
        if self.mouse_xy is not None:
            return self.mouse_xy
        (x0, x1), (y0, y1) = self.vb.viewRange()
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def add_midline(self):
        x, y = self._cursor_or_center()
        (_, _), (y0, y1) = self.vb.viewRange()
        half = 0.3 * abs(y1 - y0)
        if self.midline_roi is not None:
            self.vb.removeItem(self.midline_roi)
            for label in self.midline_labels:
                self.vb.removeItem(label)
        self._make_midline((x, y - half), (x, y + half))
        self._set_overlays_visible(True)
        self._changed(implant=True)
        self.flash('drag the D handle onto the dorsal midline')

    def start_draw(self, mode):
        if self.drawing == mode:
            return self.finish_draw()
        self._cancel_draw()
        self.drawing = mode
        self.draft = []
        color = PEN['surface'].color() if mode == 'surface' else \
            QtGui.QColor(*self._shank_color(self.shank_combo.currentText()))
        self.draft_item.setPen(pg.mkPen(color, width=1, style=_qt('DashLine')))
        what = ('dorsal surface' if mode == 'surface'
                else 'scar, shank %s' % self.shank_combo.currentText())
        self.flash('tracing %s: click points, Enter to finish, Esc to cancel' % what)

    def finish_draw(self):
        mode, pts = self.drawing, self.draft
        self._cancel_draw()
        if mode is None:
            return
        if len(pts) < 2:
            self.flash('need at least 2 points', error=True)
            return
        if mode == 'surface':
            if self.surface_roi is not None:
                self.vb.removeItem(self.surface_roi)
            self._make_surface(pts)
            self._changed()
        else:
            self._make_scar(pts, self.shank_combo.currentText())
            self._changed(implant=True)
            self.flash('scar %s traced -- trace the deepest point of the scar '
                       'carefully on its last section' % self.shank_combo.currentText())

    def _cancel_draw(self):
        self.drawing = None
        self.draft = []
        self.draft_item.setData([], [])

    def add_ellipse(self):
        x, y = self._cursor_or_center()
        a, b = (v * self._px_per_um() for v in self.default_axes_um)
        roi = self._make_ellipse(dict(center_px=[x, y], axes_px=[a, b],
                                      angle_deg=0.0))
        roi.setVisible(True)
        self._select_roi(roi)
        if self.sec()['lhy'] is None:
            self.set_state('yes')
        self._changed()

    def delete_selected(self):
        if self.selected is not None and self.selected in self.ellipse_rois:
            self.ellipse_rois.remove(self.selected)
            self.vb.removeItem(self.selected)
            self.selected = None
            self._changed()
            return
        if self.scar_rois and self.mouse_xy is not None:
            (x0, x1), _ = self.vb.viewRange()
            d = [_dist_to_polyline(self.mouse_xy, self._poly_points(r))
                 for r in self.scar_rois]
            j = int(np.argmin(d))
            if d[j] < 0.02 * abs(x1 - x0):
                roi = self.scar_rois.pop(j)
                self.vb.removeItem(roi)
                self.vb.removeItem(roi._text)
                self._changed(implant=True)
                return
        self.flash('nothing to delete (click an ellipse, or hover a scar)')

    def copy_previous(self):
        for j in range(self.cur - 1, -1, -1):
            src = self.sec(self.keys[j])
            if src.get('midline_px') or src.get('ellipses'):
                break
        else:
            self.flash('no earlier section with annotations', error=True)
            return
        copied = []
        if self.midline_roi is None and src.get('midline_px'):
            self._make_midline(*src['midline_px'])
            copied.append('midline')
        if self.surface_roi is None and src.get('surface_px'):
            self._make_surface(src['surface_px'])
            copied.append('surface')
        if not self.ellipse_rois and src.get('ellipses'):
            for e in src['ellipses']:
                self._make_ellipse(e)
            copied.append('%d ellipse(s)' % len(src['ellipses']))
        self._set_overlays_visible(True)
        self._changed(implant=True)
        self.flash('copied %s from %s -- adjust to this section'
                   % (', '.join(copied) or 'nothing', self.keys[j]))

    def set_state(self, state):
        self.sec()['lhy'] = state
        self._mark_dirty()
        self._refresh_panel()
        self._refresh_list_item(self.cur)

    def set_flipped(self, flipped=None):
        s = self.sec()
        s['flipped'] = (not s['flipped']) if flipped is None else bool(flipped)
        self._mark_dirty(implant=True)
        self._refresh_panel()
        self._refresh_list_item(self.cur)

    def move_on_slide(self, step):
        """Swap slide_order with the neighbouring section on the same slide."""
        j = self.cur + step
        if not (0 <= j < len(self.keys)) or \
                self.sec(self.keys[j])['slide'] != self.sec()['slide']:
            self.flash('already at the %s of slide %d'
                       % ('start' if step < 0 else 'end', self.sec()['slide']))
            return
        a, b = self.sec(), self.sec(self.keys[j])
        a['slide_order'], b['slide_order'] = b['slide_order'], a['slide_order']
        self._reorder()

    def set_shank(self, i):
        if i < self.shank_combo.count():
            self.shank_combo.setCurrentIndex(i)
            self.flash('current shank %s' % self.shank_combo.currentText())

    def set_ac_here(self):
        self._set_series('ac_section', self.keys[self.cur])
        self._refresh_ac_label()

    def _set_series(self, name, value, implant=False):
        self.ann['series'][name] = value
        self._mark_dirty(implant=implant)
        self._update_status()

    def _on_ac_ap_edited(self):
        txt = self.ac_ap.text().strip()
        self._set_series('ac_ap_um', float(txt) if txt else None)
        self.view.setFocus()

    def _on_notes_edited(self):
        if self.sec()['notes'] != self.notes.text():
            self.sec()['notes'] = self.notes.text()
            self._mark_dirty()
        self.view.setFocus()

    def _on_dv_mode(self, *_):
        self.dv_mode = self.dv_combo.currentText()
        self.ref_ml_um = self.ref_ml.value() or None
        self._geom = None
        self._update_status()

    def _on_slide_order(self, value):
        s = self.sec()
        old = s['slide_order']
        if value == old:
            return
        for k, other in self.ann['sections'].items():     # keep orders unique
            if other is not s and other['slide'] == s['slide'] and other['slide_order'] == value:
                other['slide_order'] = old
        s['slide_order'] = int(value)
        self._reorder()

    def _on_gap(self, value):
        if self.sec()['gap_before'] != value:
            self.sec()['gap_before'] = int(value)
            self._reorder()

    # -------------------------------------------------------------- events --
    def eventFilter(self, obj, ev):
        if ev.type() != EV_KEYPRESS or not self.isActiveWindow():
            return False
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                              QtWidgets.QComboBox)):
            if ev.key() in (K['Return'], K['Enter'], K['Escape']):
                self.view.setFocus()
            return False
        return self.handle_key(ev)

    def handle_key(self, ev):
        key, txt = ev.key(), ev.text()
        ctrl = bool(ev.modifiers() & M_CTRL)
        if ctrl and key == K['S']:
            self.save()
        elif key in (K['Right'], K['PageDown']):
            self.goto(self.cur + 1)
        elif key in (K['Left'], K['PageUp']):
            self.goto(self.cur - 1)
        elif key in (K['Return'], K['Enter']) and self.drawing:
            self.finish_draw()
        elif key == K['Escape']:
            if self.drawing:
                self._cancel_draw()
                self.flash('tracing cancelled')
            else:
                self._select_roi(self.selected)   # clears
        elif key in (K['Delete'], K['Backspace']):
            self.delete_selected()
        elif txt == 'y':
            self.set_state('yes')
        elif txt == 'n':
            self.set_state('no')
        elif txt in ('?', '/'):
            self.set_state('unsure')
        elif txt == 'u':
            self.next_unreviewed()
        elif txt == 'm':
            self.add_midline()
        elif txt == 's':
            self.start_draw('surface')
        elif txt == 't':
            self.start_draw('scar')
        elif txt and txt in '12345678':
            self.set_shank(int(txt) - 1)
        elif txt == 'e':
            self.add_ellipse()
        elif txt == 'v':
            self.copy_previous()
        elif txt == 'x':
            self.set_flipped()
        elif txt == '[':
            self.move_on_slide(-1)
        elif txt == ']':
            self.move_on_slide(1)
        elif txt == 'a':
            self.set_ac_here()
        elif txt == 'h':
            self.overlays_visible = not self.overlays_visible
            self._set_overlays_visible(self.overlays_visible)
        elif txt == 'f':
            self.vb.autoRange(padding=0.02)
        elif txt == 'q':
            self.close()
        else:
            return False
        return True

    def _on_mouse_moved(self, pos):
        if not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        self.mouse_xy = (float(p.x()), float(p.y()))
        if self.drawing and self.draft:
            pts = np.asarray(self.draft + [list(self.mouse_xy)])
            self.draft_item.setData(pts[:, 0], pts[:, 1])
        self._update_status()

    def _on_scene_click(self, ev):
        if not self.drawing or ev.button() != LEFT_BUTTON:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        self.draft.append([float(p.x()), float(p.y())])
        pts = np.asarray(self.draft)
        self.draft_item.setData(pts[:, 0], pts[:, 1])
        ev.accept()

    def closeEvent(self, ev):
        if hasattr(self, '_loaded_key'):
            self._pull_items()
        if self.dirty:
            self.save()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        super(LHyROIGUI, self).closeEvent(ev)

    # ------------------------------------------------------------- display --
    def flash(self, text, error=False):
        self.message.setStyleSheet('color: %s;' % ('#e05050' if error else '#a0a0a0'))
        self.message.setText(text)
        if error:
            print('  ' + text)

    def _list_text(self, k):
        s = self.sec(k)
        mark = {None: '  .', 'yes': 'LHy', 'no': ' no', 'unsure': '  ?'}[s['lhy']]
        flags = ('F' if s.get('flipped') else ' ')
        extra = ''
        if s['ellipses']:
            extra += ' %de' % len(s['ellipses'])
        if s.get('scars'):
            extra += ' scar:' + ''.join(sorted({sc['shank'] for sc in s['scars']}))
        ac = '  [AC]' if self.ann['series'].get('ac_section') == k else ''
        gap = '  (+%d lost)' % s['gap_before'] if s.get('gap_before') else ''
        return '%4d  s%02d #%-2d r%03d %s %s%s%s%s' % (
            s['section_index'], s['slide'], s['slide_order'], s['region'],
            flags, mark, extra, ac, gap)

    def _refresh_list_item(self, i):
        item = self.section_list.item(i)
        if item is None:
            return
        k = self.keys[i]
        item.setText(self._list_text(k))
        item.setBackground(QtGui.QBrush(QtGui.QColor(*STATUS_COLORS[self.sec(k)['lhy']])))

    def _refresh_list(self):
        self.section_list.clear()
        fdb = QtGui.QFontDatabase
        fixed = getattr(fdb, 'FixedFont', None)
        if fixed is None:
            fixed = fdb.SystemFont.FixedFont
        font = fdb.systemFont(fixed)
        for i, k in enumerate(self.keys):
            item = QtWidgets.QListWidgetItem()
            item.setFont(font)
            item.setForeground(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
            self.section_list.addItem(item)
            self._refresh_list_item(i)
        if self.keys:
            self.section_list.setCurrentRow(self.cur)
        dup = lrt.duplicate_slide_orders(self.ann)
        if dup:
            self.flash('duplicate slide order: %s' % dup, error=True)

    def _refresh_ac_label(self):
        ac = self.ann['series'].get('ac_section')
        self.ac_label.setText(ac or 'unset')
        if self.keys:
            self._refresh_list()

    def _refresh_panel(self):
        s = self.sec()
        self.file_label.setText('<b>%s</b> &nbsp; index %d &nbsp; (%d / %d)<br>%s'
                                % (self.keys[self.cur], s['section_index'],
                                   self.cur + 1, len(self.keys), s['file']))
        for spin, value in ((self.order_spin, s['slide_order']),
                            (self.gap_spin, s['gap_before'])):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self.flip_box.setChecked(bool(s['flipped']))
        self.flip_label.setVisible(bool(s['flipped']))
        self.state_group.setExclusive(False)
        for b in self.state_group.buttons():
            b.setChecked(False)
        self.state_group.setExclusive(True)
        if s['lhy'] is not None:
            self.state_group.button(['yes', 'no', 'unsure'].index(s['lhy'])).setChecked(True)
        self.notes.setText(s.get('notes', ''))
        self._update_status()

    def _implant_sign(self):
        if self._implant is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._implant = lrt.resolve_implant_sign(
                    self.ann, pixel_size_override=self.pixel_size_override)
        return self._implant

    def _geometry(self):
        if self._geom is None:
            sign, _ = self._implant_sign()
            try:
                self._geom = lrt.SectionGeometry(
                    self.sec(), self.ann['series'].get('inplane_scale', 1.0),
                    dv_mode=self.dv_mode, ref_ml_um=self.ref_ml_um,
                    implant_sign=sign or 1)
            except ValueError:
                self._geom = False
        return self._geom

    def _update_status(self):
        if not self.keys:
            return
        s = self.sec()
        parts = []
        if self.mouse_xy is not None:
            parts.append('px (%6.0f, %6.0f)' % self.mouse_xy)
            geom = self._geometry()
            if geom:
                ml, dv = geom.px_to_brain(np.asarray(self.mouse_xy))
                sign, source = self._implant_sign()
                side = 'ML' if sign is not None else 'ML(img)'
                parts.append('%s %+6.0f  DV[%s] %6.0f' % (side, ml[0], self.dv_mode, dv[0]))
        ap_rel, ap_abs = lrt.section_ap(self.ann)[self.keys[self.cur]]
        parts.append('AP-AC %s' % ('%+.0f' % ap_rel if np.isfinite(ap_rel) else 'n/a'))
        if np.isfinite(ap_abs):
            parts.append('AP %+.0f' % ap_abs)
        if s['pixel_size_um']:
            parts.append('%.3g um/px' % np.mean(s['pixel_size_um']))
        if self.dirty:
            parts.append('*')
        self.status.setText('  '.join(parts))

    # ---------------------------------------------------------------- save --
    def save(self):
        if hasattr(self, '_loaded_key'):
            self._pull_items()
        try:
            lrt.save_annotation(self.ann, self.ann_path)
        except Exception as err:
            self.flash('SAVE FAILED: %s' % err, error=True)
            return
        self.dirty = False
        self.flash('saved %s' % os.path.basename(self.ann_path))
        if self.export_csv:
            self._export_csv()
        self._update_status()

    def _export_csv(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rois = lrt.LHyROIs(self.ann, dv_mode=self.dv_mode,
                                   ref_ml_um=self.ref_ml_um,
                                   pixel_size_override=self.pixel_size_override)
                tracks = rois.scar_tracks()
            stem = os.path.splitext(self.ann_path)[0]
            rois.sections.to_csv(stem + '_sections.csv', index=False)
            if len(rois.rois):
                rois.rois.drop(columns='boundary').to_csv(stem + '_ellipses.csv',
                                                          index=False)
            if len(tracks):
                tracks.to_csv(stem + '_scar_tracks.csv', index=False)
        except Exception as err:            # export is a convenience only
            print('  csv export skipped: %s' % err)


# --------------------------------------------------------------------------- #
#  ENTRY POINT                                                                 #
# --------------------------------------------------------------------------- #

def build_annotation(bird, hist_dir, annotation_path, file_pattern, channel,
                     series_params):
    found = lrt.discover_sections(hist_dir, file_pattern=file_pattern,
                                  channel=channel)
    if not found and not os.path.exists(annotation_path):
        raise IOError('no images matching the file pattern in %s' % hist_dir)
    print('%s: %d section images in %s' % (bird, len(found), hist_dir))

    if os.path.exists(annotation_path):
        ann = lrt.load_annotation(annotation_path)
        print('  loaded %s' % annotation_path)
        secs = ann['sections']
        for k, info in found.items():
            if k not in secs:
                order = lrt.next_slide_order(secs, info['slide'])
                secs[k] = lrt.empty_section(info, order)
                print('  new image %s -> slide %d, order %d (check its position)'
                      % (info['file'], info['slide'], order))
            elif secs[k]['file'] != info['file']:
                print('  %s: image file is now %s' % (k, info['file']))
                secs[k]['file'] = info['file']
        gone = [k for k in secs if k not in found]
        if gone:
            print('  %d annotated section(s) have no image on disk: %s'
                  % (len(gone), gone))
    else:
        ann = lrt.new_annotation(bird, hist_dir, found)
        print('  new annotation file %s' % annotation_path)

    # run-script values override the file whenever they are set
    for name, value in series_params.items():
        if value is None:
            continue
        old = ann['series'].get(name)
        if old is not None and old != value:
            print('  series %s: file has %r, run script sets %r -- using %r'
                  % (name, old, value, value))
        ann['series'][name] = value
    lrt.assign_section_indices(ann)
    return ann


def launch(bird, hist_dir,
           annotation_path=None,
           file_pattern=lrt.DEFAULT_FILE_PATTERN,
           channel=None,
           section_thickness_um=None,
           section_interval=None,
           ap_sign=None,
           ac_section=None,
           ac_ap_um=None,
           implant_side=None,
           inplane_scale=None,
           shank_labels=('A',),
           pixel_size_override=None,
           display_downsample=4,
           cache_dir=None,
           default_ellipse_axes_um=(250., 200.),
           dv_mode='local',
           ref_ml_um=None,
           keep_levels=True,
           export_csv=True):
    """Build the annotation and open the GUI.  See run_lhy_roi_gui.py."""
    if annotation_path is None:
        annotation_path = os.path.join(hist_dir, '%s_lhy_rois.json' % bird)
    if isinstance(ac_section, (tuple, list)):
        ac_section = lrt.section_key(*ac_section)
    series = OrderedDict(section_thickness_um=section_thickness_um,
                         section_interval=section_interval, ap_sign=ap_sign,
                         ac_section=ac_section, ac_ap_um=ac_ap_um,
                         implant_side=implant_side, inplane_scale=inplane_scale)
    ann = build_annotation(bird, hist_dir, annotation_path, file_pattern,
                           channel, series)
    if ann['series']['ac_section'] and ann['series']['ac_section'] not in ann['sections']:
        print('  WARNING: ac_section %s is not one of the sections'
              % ann['series']['ac_section'])

    images = ImageCache(hist_dir, downsample=display_downsample,
                        cache_dir=cache_dir)

    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = LHyROIGUI(ann, annotation_path, images,
                    pixel_size_override=pixel_size_override,
                    default_ellipse_axes_um=default_ellipse_axes_um,
                    dv_mode=dv_mode, ref_ml_um=ref_ml_um,
                    keep_levels=keep_levels, export_csv=export_csv,
                    shank_labels=shank_labels)
    gui.resize(1550, 980)
    gui.show()
    gui.raise_()
    gui.activateWindow()
    return (getattr(app, 'exec', None) or app.exec_)()
