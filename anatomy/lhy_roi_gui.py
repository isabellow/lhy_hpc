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
    * whether the entry is a section at all ('d': detritus that got imaged by
      mistake leaves the AP count without counting as a lost section)
    * which entries are pieces of one section that broke up during mounting
      ('g': the pieces share one step of the AP series, and each piece is
      annotated and measured entirely on its own)

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
EV_RESIZE = _event_type('Resize')

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
w &nbsp; DM/DL boundary at the surface (hippocampal width -> AP)<br>
Del &nbsp; selected ellipse, else the scar under the cursor<br>
[ / ] &nbsp; move section earlier / later on its slide<br>
x &nbsp; flipped &nbsp;&nbsp; a &nbsp; AC reference = this section<br>
d &nbsp; not a section (detritus): out of the AP count, not a lost section<br>
g &nbsp; another piece of the SAME section as the entry before it<br>
&nbsp;&nbsp;&nbsp;(one AP step for the group; annotate each piece separately)<br>
shift+G &nbsp; split it off again<br>
c &nbsp; annotate this section on the slide overview instead (for the part<br>
&nbsp;&nbsp;&nbsp;the high-res scan missed); c again returns to the scan<br>
v &nbsp; copy midline/surface/ellipses from previous section<br>
h &nbsp; hide overlays<br>
+ / - &nbsp; zoom in / out at the cursor &nbsp;&nbsp; z or f &nbsp; whole section<br>
i &nbsp; slide inset on / off; click it to place this section by hand<br>
&nbsp;&nbsp;&nbsp;orange <b>?</b> = tissue on the slide with no image, counted as a gap<br>
o &nbsp; locate this slide again (drops hand placements)<br>
shift+O &nbsp; the same, and re-order the slide from the overview<br>
Ctrl+S save &nbsp;&nbsp; q &nbsp; quit"""


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
        band = arr[r:min(r + step, h), :w].astype(np.float32)
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


DEFAULT_SLIDE_PATTERN = (r'Slide\d+-(?P<slide>\d+)_Channel(?P<channel>.+?)'
                         r'_Seq(?P<seq>\d+)\.nd2$')


def read_nd2_header(path):
    """Pixel size, shape and stage position of an nd2, without reading pixels."""
    if not path.lower().endswith('.nd2'):
        return dict(px=None, shape=None, stage=None)
    import nd2
    px = shape = stage = None
    with nd2.ND2File(path) as f:
        try:
            vs = f.voxel_size()
            if vs.x > 0 and vs.y > 0:
                px = (float(vs.x), float(vs.y))
        except Exception:
            pass
        sizes = f.sizes
        if 'Y' in sizes and 'X' in sizes:
            shape = (int(sizes['Y']), int(sizes['X']))
        try:
            p = f.frame_metadata(0).channels[0].position.stagePositionUm
            stage = (float(p.x), float(p.y))
        except Exception:
            pass
    return dict(px=px, shape=shape, stage=stage)


def find_tissue_blobs(img, min_frac=0.2):
    """
    The tissue pieces on a slide overview: centroids (n, 2) as (x, y) and
    bounding boxes (n, 4) as (x0, y0, x1, y1), in pixels of `img`.  Empty if
    scipy is missing or nothing is found.
    """
    try:
        from scipy import ndimage
    except ImportError:
        warnings.warn('scipy not installed -- cannot find the sections on the '
                      'slide overview automatically')
        return np.zeros((0, 2)), np.zeros((0, 4))
    a = np.asarray(img, float)
    nz = a[a > 0]
    if nz.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 4))
    mid, high = np.percentile(nz, [50, 99.5])
    mask = ndimage.binary_fill_holes(
        ndimage.binary_closing(a > mid + 0.25 * (high - mid), np.ones((5, 5))))
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0, 4))
    sizes = np.asarray(ndimage.sum(mask, lab, range(1, n + 1)))
    keep = [i + 1 for i, sz in enumerate(sizes) if sz > min_frac * sizes.max()]
    cents = ndimage.center_of_mass(mask, lab, keep)
    boxes = ndimage.find_objects(lab)
    bb = [[boxes[k - 1][1].start, boxes[k - 1][0].start,
           boxes[k - 1][1].stop, boxes[k - 1][0].stop] for k in keep]
    return (np.array([[c[1], c[0]] for c in cents], float), np.array(bb, float))


def _assign(cost):
    """Hungarian assignment if scipy is around, greedy otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        rows, cols = [], []
        cost = cost.copy()
        for _ in range(min(cost.shape)):
            r, c = np.unravel_index(np.argmin(cost), cost.shape)
            rows.append(r)
            cols.append(c)
            cost[r, :] = np.inf
            cost[:, c] = np.inf
        return np.array(rows), np.array(cols)


def locate_sections_on_slide(slide, sections, blobs_px, px_um, shape,
                             axis_signs=None):
    """
    Work out where each section image sits on its slide overview, from the
    stage positions in the nd2 metadata.

    The stage axes may run either way relative to the image, so all four sign
    combinations are tried and scored against the tissue blobs found on the
    overview; the best is kept if it lines up well enough.

    sections  : {key: dict(stage=(x_um, y_um), shape=(h, w), px=(x, y))} for the
                sections on this slide, plus the slide's own 'stage' under the
                key 'slide'
    blobs_px  : (n, 2) tissue centroids in slide-overview pixels (full res)
    px_um     : um per pixel of the overview
    shape     : (h, w) of the overview, full resolution

    Returns (positions {key: (x_px, y_px)}, signs, residual_um or None).
    Empty if it could not be worked out.
    """
    slide_stage = sections.get('slide', {}).get('stage')
    keys = [k for k in sections if k != 'slide' and sections[k].get('stage')]
    if slide_stage is None or not keys or len(blobs_px) == 0:
        return {}, axis_signs, None

    center = np.array([shape[1] / 2.0, shape[0] / 2.0])
    delta = np.array([np.asarray(sections[k]['stage'], float) -
                      np.asarray(slide_stage, float) for k in keys])
    combos = [axis_signs] if axis_signs else [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # how far apart the pieces are, for the tolerance
    if len(blobs_px) > 1:
        dd = np.sqrt(((blobs_px[:, None, :] - blobs_px[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        spacing = np.median(dd.min(1))
    else:
        spacing = max(shape) / 2.0

    best = None
    for signs in combos:
        pred = center + delta * np.asarray(signs, float) / px_um
        cost = np.sqrt(((pred[:, None, :] - blobs_px[None, :, :]) ** 2).sum(-1))
        rows, cols = _assign(cost)
        resid = float(np.median(cost[rows, cols]))
        if best is None or resid < best[0]:
            best = (resid, signs, pred)
    resid, signs, pred = best
    if resid > 0.35 * spacing:
        warnings.warn('slide %s: the stage positions do not line up with the '
                      'sections on the overview (median offset %.0f um) -- click '
                      'in the inset to place sections by hand, or set '
                      'slide_axis_signs' % (slide, resid * px_um))
        return {}, None, resid * px_um
    return ({k: tuple(p) for k, p in zip(keys, pred)}, signs, resid * px_um)


def slide_order_from_positions(positions, footprint_px=None):
    """
    Cutting order of the sections on one slide from where they sit on it:
    columns left to right, and within a column top to bottom -- i.e.

        1 | 3 | 5
        2 | 4 | 6

    positions     : {key: (x, y)} in overview pixels
    footprint_px  : width of one section on the overview, used to decide where
                    one column ends and the next begins; falls back to half the
                    typical spacing between sections.

    Returns {key: slide_order}, 1-based.
    """
    keys = list(positions)
    if not keys:
        return {}
    xy = np.array([positions[k] for k in keys], float)
    if footprint_px:
        gap = 0.5 * footprint_px
    elif len(keys) > 1:
        dd = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        gap = 0.5 * np.median(dd.min(1))
    else:
        gap = np.inf

    by_x = np.argsort(xy[:, 0])
    col = np.zeros(len(keys), int)
    for i in range(1, len(by_x)):
        step = xy[by_x[i], 0] - xy[by_x[i - 1], 0]
        col[by_x[i]] = col[by_x[i - 1]] + (1 if step > gap else 0)
    rank = sorted(range(len(keys)), key=lambda i: (col[i], xy[i, 1]))
    return {keys[i]: r + 1 for r, i in enumerate(rank)}


def crop_section_key(slide, slot):
    return 'slide%02d_slot%03d' % (int(slide), int(slot))


def split_shared_images(secs, slide, blobs, boxes, px_slide, slides, verbose=True):
    """
    Some high-resolution images cover two sections at once (sections mounted
    close together or touching).  Such an image is roughly twice the usual size
    and is the nearest image to more than one piece of tissue on the overview.

    Each piece becomes a section of its own, cropped out of the shared image,
    so the two are shown separately and in order and each takes its own slot in
    the AP series; the shared entry steps aside.  A normal image is left alone
    even though its frame is wider than the section it holds, because each of
    the neighbouring pieces is nearer to its own image.

    The crop comes from the stage coordinates: a displacement on the overview
    is the same displacement on the image, scaled by the ratio of pixel sizes,
    whichever way the stage axes run.
    """
    keys = [k for k, s in secs.items()
            if int(s['slide']) == slide and s.get('source', 'image') == 'image'
            and not s.get('superseded') and not s.get('discarded')
            and not s.get('crop_px') and s.get('slide_xy_px')]
    if len(keys) == 0 or len(blobs) == 0:
        return []
    centers = np.array([secs[k]['slide_xy_px'] for k in keys], float)
    owner = np.argmin(np.sqrt(((blobs[:, None, :] - centers[None, :, :]) ** 2).sum(-1)), axis=1)

    made = []
    for ki, key in enumerate(keys):
        sec = secs[key]
        h = slides.header(sec['file'])
        if not (h['shape'] and h['px']):
            continue
        center = centers[ki]
        px_img = float(np.mean(h['px']))
        half = np.array([h['shape'][1], h['shape'][0]], float) * px_img / px_slide / 2.0
        mine = [i for i in np.nonzero(owner == ki)[0]
                if np.all(np.abs(blobs[i] - center) < 0.95 * half)]
        if len(mine) < 2:
            continue

        img_center = np.array([h['shape'][1], h['shape'][0]], float) / 2.0
        for j, i in enumerate(mine):
            box = boxes[i]
            pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
            corners = np.array([[box[0] - pad, box[1] - pad],
                                [box[2] + pad, box[3] + pad]], float)
            crop = img_center + (corners - center) * px_slide / px_img
            crop[:, 0] = np.clip(crop[:, 0], 0, h['shape'][1])
            crop[:, 1] = np.clip(crop[:, 1], 0, h['shape'][0])
            part_key = '%s_p%d' % (key, j + 1)
            if part_key not in secs:
                secs[part_key] = lrt.empty_section(
                    dict(file=sec['file'], slide=slide, region=sec['region'],
                         source='image', crop_px=crop.ravel().tolist()), j + 1)
                secs[part_key]['pixel_size_um'] = list(h['px'])
                secs[part_key]['image_shape'] = list(h['shape'])
            secs[part_key]['part'] = '%d/%d' % (j + 1, len(mine))
            secs[part_key]['crop_px'] = crop.ravel().tolist()
            secs[part_key]['slide_xy_px'] = [float(blobs[i][0]), float(blobs[i][1])]
            secs[part_key]['slide_xy_source'] = 'auto'
            made.append(part_key)
        sec['superseded'] = True
        if verbose:
            print('  %s covers %d sections (%.1f x %.1f mm frame) -- split into %s'
                  % (sec['file'], len(mine), 2 * half[0] * px_slide / 1000,
                     2 * half[1] * px_slide / 1000,
                     ', '.join('%s_p%d' % (key, j + 1) for j in range(len(mine)))))
    return made


def apply_slide_layout(ann, slides, auto_order=True, only_slide=None,
                       force=False, verbose=True, annotate_unimaged=True):
    """
    Put every section on its slide overview (from the nd2 stage coordinates)
    and, with auto_order, take the cutting order from those positions.

    Every piece of tissue on the overview is a slot, ordered down each column
    and then left to right.  Sections with a high-resolution image take their
    slide_order from their slot.  A slot with no image of its own becomes,
    with annotate_unimaged, a section of its own that is annotated on the
    overview crop (coarse -- ~8 um/px -- but better than nothing, especially
    where the scar is); otherwise it is only counted into the gap_before of
    the next imaged section.  Either way section_index steps once per cut
    section, so AP stays right.  Trailing empty slots carry over to the next
    slide.

    A slot whose entry is marked discarded ('d' in the GUI: detritus, not a
    section) is skipped entirely -- it is neither a step of the series nor a
    lost section, so the sections around it end up adjacent.  Keep
    annotate_unimaged on if the slide has junk on it that was never imaged:
    that is what gives you an entry to mark as discarded.

    Positions are stored in the annotation, so this is only slow the first
    time.  Sections placed by hand keep their position, a slide with any
    hand-set order keeps its order, and a hand-set gap_before is never
    overwritten -- unless force=True (shift+O in the GUI), which redoes
    everything for that slide.
    """
    if slides is None:
        return
    secs = ann['sections']
    wanted = sorted({int(s['slide']) for s in secs.values()}
                    if only_slide is None else {int(only_slide)})
    carry = 0                      # empty slots left over from the last slide
    for slide in wanted:
        real = [k for k, s in secs.items()
                if int(s['slide']) == slide and s.get('source', 'image') == 'image'
                and not s.get('superseded')]
        crops = [k for k, s in secs.items()
                 if int(s['slide']) == slide and s.get('source') == 'slide_crop'
                 and not s.get('superseded') and not s.get('companion_of')]
        if not real and not crops:
            continue
        missing = [k for k in real
                   if secs[k].get('slide_xy_px') is None or
                   (force and secs[k].get('slide_xy_source') != 'manual')]
        if missing and slides.has(slide):
            if verbose:
                print('  slide %d: locating %d section(s) on the overview...'
                      % (slide, len(missing)))
            if force:
                slides.forget(slide)
            pos = slides.positions(slide, secs)
            for k in missing:
                if k in pos:
                    secs[k]['slide_xy_px'] = [float(pos[k][0]), float(pos[k][1])]
                    secs[k]['slide_xy_source'] = 'auto'

        info = slides.info(slide)
        if info is not None and len(info['blobs']):
            if split_shared_images(secs, slide, info['blobs'], info['boxes'],
                                   info['px_um'], slides, verbose=verbose):
                real = [k for k, s in secs.items()
                        if int(s['slide']) == slide
                        and s.get('source', 'image') == 'image'
                        and not s.get('superseded')]
        placed = {k: np.asarray(secs[k]['slide_xy_px'], float) for k in real
                  if secs[k].get('slide_xy_px')}
        if not auto_order or info is None or len(placed) < len(real):
            if auto_order and verbose and len(placed) < len(real):
                print('  slide %d: only %d of %d sections could be placed -- '
                      'order and gaps left as they are' % (slide, len(placed), len(real)))
            carry = 0
            continue
        by_hand = [k for k in real + crops if secs[k].get('order_source') == 'manual']
        if by_hand and not force:
            if verbose:
                print('  slide %d: order set by hand, left alone' % slide)
            carry = 0
            continue

        footprint = None
        if real:
            h = slides.header(secs[real[0]]['file'])
            if h['shape'] and h['px']:
                footprint = h['shape'][1] * np.mean(h['px']) / info['px_um']

        # every piece of tissue on the slide is a slot, in cutting order
        blobs, boxes = info['blobs'], info['boxes']
        if len(blobs) < len(placed):
            blobs = np.array(list(placed.values()))
            boxes = None
        slots = slide_order_from_positions(
            {i: tuple(b) for i, b in enumerate(blobs)}, footprint)

        # match each imaged section to its slot, one to one
        slot_key = {}
        if placed:
            cost = np.sqrt(((np.array(list(placed.values()))[:, None, :] -
                             blobs[None, :, :]) ** 2).sum(-1))
            rows, cols = _assign(cost)
            tol = 0.5 * (footprint or np.inf)
            bad = 0
            for r, c in zip(rows, cols):
                if cost[r, c] > tol:
                    bad += 1
                else:
                    slot_key[slots[c]] = list(placed)[r]
            if bad:
                if verbose:
                    print('  slide %d: %d section(s) did not land on a piece of '
                          'tissue -- order and gaps left as they are' % (slide, bad))
                carry = 0
                continue

        # slots with no high-resolution image: annotate them on the overview
        empty = [c for c in slots if slots[c] not in slot_key]
        if annotate_unimaged and boxes is not None and slides.has(slide):
            px = slides.info(slide)['px_um']
            for c in empty:
                slot = slots[c]
                key = crop_section_key(slide, slot)
                box = boxes[c]
                pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
                crop = [float(box[0] - pad), float(box[1] - pad),
                        float(box[2] + pad), float(box[3] + pad)]
                if key not in secs:
                    secs[key] = lrt.empty_section(
                        dict(file=info['file'], slide=slide, region=1000 + slot,
                             source='slide_crop', crop_px=crop), slot)
                    secs[key]['pixel_size_um'] = [px, px]
                    secs[key]['image_shape'] = [int(info['shape'][0]),
                                                int(info['shape'][1])]
                    if verbose:
                        print('  slide %d: slot %d has no high-resolution image -- '
                              'added it as a crop of the overview (%.2f um/px)'
                              % (slide, slot, px))
                secs[key]['crop_px'] = crop
                secs[key]['slide_xy_px'] = [float(blobs[c][0]), float(blobs[c][1])]
                secs[key]['slide_xy_source'] = 'auto'
                slot_key[slot] = key

        # a crop that has since been imaged properly steps aside
        for key in crops:
            if key not in slot_key.values():
                if not secs[key].get('superseded'):
                    secs[key]['superseded'] = True
                    if verbose:
                        print('  %s: a high-resolution image now covers this slot -- '
                              'the crop is set aside%s' % (key, ' (it has annotations: '
                              'redraw them on the new image)'
                              if secs[key].get('ellipses') or secs[key].get('scars')
                              else ''))

        # slots holding something that is not a section are skipped over: they
        # are not steps of the series and not lost sections either
        dead = {slot for slot, k in slot_key.items() if secs[k].get('discarded')}

        def live_between(a, b):
            return sum(1 for t in range(a + 1, b) if t not in dead)

        order = sorted(slot_key)
        moved = gapped = 0
        prev = 0
        first_live = True
        for i, slot in enumerate(order):
            k = slot_key[slot]
            secs[k]['superseded'] = False
            if secs[k]['slide_order'] != i + 1:
                moved += 1
            secs[k]['slide_order'] = i + 1
            secs[k]['order_source'] = 'slide'
            if slot in dead:                      # not a section: no gap, no step
                secs[k]['gap_before'] = 0
                secs[k]['gap_source'] = 'slide'
                continue
            gap = live_between(prev, slot) + (carry if first_live else 0)
            prev = slot
            first_live = False
            if secs[k].get('gap_source') == 'manual' and not force:
                continue
            if secs[k]['gap_before'] != gap:
                gapped += 1
            secs[k]['gap_before'] = int(gap)
            secs[k]['gap_source'] = 'slide'
        trailing = live_between(prev, len(slots) + 1)
        n_empty = sum(1 for c in empty if slots[c] not in slot_key) + trailing
        if verbose and (moved or n_empty):
            msg = '  slide %d: order taken from the overview' % slide
            if moved:
                msg += ' (%d section(s) moved)' % moved
            if n_empty:
                msg += '; %d piece(s) of tissue with no image counted as ' \
                       'unimaged sections' % n_empty
            print(msg)
        carry = trailing if only_slide is None else 0
    lrt.assign_section_indices(ann)


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


def discover_slides(hist_dir, slide_pattern=DEFAULT_SLIDE_PATTERN, channel=None):
    """{slide number: overview file name} for the whole-slide scans."""
    import re
    rx = re.compile(slide_pattern)
    out = {}
    for name in sorted(os.listdir(hist_dir)):
        m = rx.search(name)
        if m is None:
            continue
        g = m.groupdict()
        if channel is not None and \
                lrt._norm_channel(g.get('channel', '')) != lrt._norm_channel(channel):
            continue
        out[int(g['slide'])] = name
    return out


class SlideOverview(object):
    """
    Whole-slide overview images, and where each section image sits on them.

    Only the overview pixels are read eagerly; section files are opened for
    their headers alone (~0.1 s each), never for pixels.
    """

    def __init__(self, hist_dir, slide_pattern=DEFAULT_SLIDE_PATTERN, channel=None,
                 downsample=4, cache_dir=None, axis_signs=None):
        self.hist_dir = hist_dir
        self.images = ImageCache(hist_dir, downsample=downsample,
                                 cache_dir=cache_dir, max_items=2)
        self.full_images = ImageCache(hist_dir, downsample=1,
                                      cache_dir=cache_dir, max_items=2)
        self.axis_signs = tuple(axis_signs) if axis_signs else None
        try:
            self.files = discover_slides(hist_dir, slide_pattern, channel)
        except OSError as err:
            warnings.warn('could not list %s: %s' % (hist_dir, err))
            self.files = {}
        self._info = {}
        self._pos = {}
        self._headers = {}

    def has(self, slide):
        return int(slide) in self.files

    def header(self, fname):
        if fname not in self._headers:
            try:
                self._headers[fname] = read_nd2_header(
                    os.path.join(self.hist_dir, fname))
            except Exception as err:
                warnings.warn('could not read the header of %s: %s' % (fname, err))
                self._headers[fname] = dict(px=None, shape=None, stage=None)
        return self._headers[fname]

    def info(self, slide):
        """dict(img, shape, px, blobs) for one slide overview, or None."""
        slide = int(slide)
        if slide in self._info:
            return self._info[slide]
        if not self.has(slide):
            self._info[slide] = None
            return None
        fname = self.files[slide]
        try:
            img, shape, px = self.images.get(fname)
        except Exception as err:
            warnings.warn('could not load slide overview %s: %s' % (fname, err))
            self._info[slide] = None
            return None
        px_um = float(np.mean(px)) if px else 1.0
        scale = shape[1] / float(img.shape[1])              # display -> full res
        blobs, boxes = find_tissue_blobs(img)
        self._info[slide] = dict(file=fname, img=img, shape=shape, px_um=px_um,
                                 blobs=blobs * scale, boxes=boxes * scale,
                                 levels=display_levels(img))
        return self._info[slide]

    def positions(self, slide, sections):
        """
        {section key: (x, y)} in full-resolution overview pixels, worked out
        from the stage positions.  `sections` maps key -> section dict.
        """
        slide = int(slide)
        if slide in self._pos:
            return self._pos[slide][0]
        info = self.info(slide)
        if info is None:
            self._pos[slide] = ({}, None, None)
            return {}
        entries = {'slide': dict(stage=self.header(info['file'])['stage'])}
        for key, sec in sections.items():
            if int(sec['slide']) != slide:
                continue
            h = self.header(sec['file'])
            if h['stage']:
                entries[key] = h
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pos, signs, resid = locate_sections_on_slide(
                slide, entries, info['blobs'], info['px_um'], info['shape'],
                axis_signs=self.axis_signs)
        for c in caught:
            print('  ' + str(c.message))
        if signs and self.axis_signs is None:
            self.axis_signs = tuple(signs)                  # reuse on other slides
        self._pos[slide] = (pos, signs, resid)
        return pos

    def full_image(self, slide):
        """The overview at full resolution, for annotating a crop of it."""
        info = self.info(slide)
        if info is None:
            return None
        return self.full_images.get(info['file'])

    def forget(self, slide):
        self._pos.pop(int(slide), None)


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
                 shank_labels=('A',), slides=None, inset_width_px=380,
                 auto_slide_order=True, annotate_unimaged=True):
        super(LHyROIGUI, self).__init__()
        self.ann = ann
        self.ann_path = ann_path
        self.images = images
        self.pixel_size_override = pixel_size_override
        self.default_axes_um = default_ellipse_axes_um
        self.dv_mode = dv_mode
        self.ref_ml_um = ref_ml_um
        self.export_csv = export_csv
        self.slides = slides
        self.auto_slide_order = auto_slide_order
        self.annotate_unimaged = annotate_unimaged
        self.inset_width_px = int(inset_width_px)
        self.shank_labels = list(shank_labels)
        for s in ann['sections'].values():           # labels already in the file
            for scar in s.get('scars', []):
                if scar.get('shank') not in self.shank_labels:
                    self.shank_labels.append(scar.get('shank'))

        lrt.assign_section_indices(ann)
        self.keys = lrt.display_keys(ann)     # discarded entries stay listed
        self._pieces = lrt.piece_labels(ann)
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
        self.dmdl_items = []

        self._build_ui(keep_levels)
        QtWidgets.QApplication.instance().installEventFilter(self)
        first = next((i for i, k in enumerate(self.keys)
                      if self.sec(k)['lhy'] is None
                      and not self.sec(k).get('discarded')), 0)
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
        key = self.keys[self.cur] if self.keys else None
        lrt.assign_section_indices(self.ann)
        self.keys = lrt.display_keys(self.ann)
        if key in self.keys:
            self.cur = self.keys.index(key)
        else:                                # split up or set aside meanwhile
            parts = [k for k in self.keys if k.startswith('%s_p' % key)]
            self.cur = self.keys.index(parts[0]) if parts else min(self.cur, len(self.keys) - 1)
            self._loaded_key = self.keys[self.cur]
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
        self.discard_label = pg.TextItem('NOT A SECTION', color=(255, 100, 100),
                                         anchor=(0, -1.3))
        self.discard_label.setFont(QtGui.QFont('Arial', 20, QtGui.QFont.Bold
                                               if hasattr(QtGui.QFont, 'Bold')
                                               else QtGui.QFont.Weight.Bold))
        self.discard_label.setZValue(40)
        self.discard_label.setVisible(False)
        self.vb.addItem(self.discard_label)
        # slide overview inset, a viewbox sitting on top of the image in
        # scene (= widget pixel) coordinates
        self.inset_vb = pg.ViewBox(enableMenu=False, border=pg.mkPen((150, 150, 150)))
        self.inset_vb.setMouseEnabled(False, False)
        self.inset_vb.setAspectLocked(True)
        self.inset_vb.invertY(True)
        self.inset_vb.setZValue(100)
        self.inset_img = pg.ImageItem(axisOrder='row-major')
        self.inset_vb.addItem(self.inset_img)
        self.inset_dots = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(90, 200, 255),
                                             size=7)
        self.inset_here = pg.ScatterPlotItem(pen=pg.mkPen('w', width=2), brush=None,
                                             size=22, symbol='o')
        self.inset_gaps = pg.ScatterPlotItem(pen=pg.mkPen((255, 170, 60), width=2),
                                             brush=None, size=13, symbol='o')
        self.inset_dead = pg.ScatterPlotItem(pen=None,
                                             brush=pg.mkBrush(150, 90, 90), size=7)
        self.inset_vb.addItem(self.inset_gaps)
        self.inset_vb.addItem(self.inset_dead)
        self.inset_vb.addItem(self.inset_dots)
        self.inset_vb.addItem(self.inset_here)
        self.inset_texts = []
        self.view.scene().addItem(self.inset_vb)
        self.view.installEventFilter(self)
        self.inset_visible = self.slides is not None

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
            '<b>sections</b>  (index[piece]  slide #order  region  flags)'))
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
        gap_label = QtWidgets.QLabel('lost before')
        gap_label.setToolTip('cut sections between the previous section in the '
                             'series and this one that have no image; filled in '
                             'from the slide overview, editable by hand')
        order_row.addWidget(gap_label)
        order_row.addWidget(self.gap_spin)
        order_row.addWidget(self.flip_box)
        form.addRow(order_row)

        piece_row = QtWidgets.QHBoxLayout()
        self.discard_box = QtWidgets.QCheckBox('not a section (d)')
        self.discard_box.setFocusPolicy(_qt('NoFocus'))
        self.discard_box.setToolTip('detritus imaged by mistake: drops out of the '
                                    'AP count without counting as a lost section')
        self.discard_box.clicked.connect(lambda checked: self.set_discarded(checked))
        self.group_label = QtWidgets.QLabel('whole')
        group_btn = QtWidgets.QPushButton('+ piece (g)')
        group_btn.setFocusPolicy(_qt('NoFocus'))
        group_btn.setToolTip('this entry is another piece of the SAME section as '
                             'the previous entry: the pieces share one AP step')
        group_btn.clicked.connect(self.group_with_previous)
        ungroup_btn = QtWidgets.QPushButton('split (G)')
        ungroup_btn.setFocusPolicy(_qt('NoFocus'))
        ungroup_btn.clicked.connect(self.ungroup)
        piece_row.addWidget(self.discard_box)
        piece_row.addWidget(self.group_label)
        piece_row.addWidget(group_btn)
        piece_row.addWidget(ungroup_btn)
        form.addRow('section', piece_row)

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

    def reset_view(self):
        """Back to the whole section (the crop, for an overview section)."""
        rect = getattr(self, 'home_rect', None)
        if rect is None:
            self.vb.autoRange(padding=0.02)
        else:
            self.vb.setRange(rect, padding=0.02)

    def zoom(self, factor):
        """Zoom the section view about the cursor (or the view centre)."""
        center = None
        if self.mouse_xy is not None:
            center = pg.Point(*self.mouse_xy)
        self.vb.scaleBy(s=(factor, factor), center=center)

    # -------------------------------------------------------- slide inset --
    def _layout_inset(self):
        info = self.slides.info(self.sec()['slide']) if self.slides else None
        if not self.inset_visible or info is None:
            self.inset_vb.setVisible(False)
            return
        w = min(self.inset_width_px, 0.4 * max(self.view.width(), 1))
        h = w * info['shape'][0] / float(info['shape'][1])
        self.inset_vb.setVisible(True)
        self.inset_vb.setGeometry(QtCore.QRectF(8, 8, w, h))
        self.inset_vb.setRange(QtCore.QRectF(0, 0, info['shape'][1], info['shape'][0]),
                               padding=0.02)

    def _update_inset(self):
        for t in self.inset_texts:
            self.inset_vb.removeItem(t)
        self.inset_texts = []
        if self.slides is None or not self.inset_visible:
            self.inset_vb.setVisible(False)
            return
        s = self.sec()
        info = self.slides.info(s['slide'])
        if info is None:
            self.inset_vb.setVisible(False)
            return
        self.inset_img.setImage(info['img'], autoLevels=False, levels=info['levels'])
        self.inset_img.setRect(QtCore.QRectF(0, 0, info['shape'][1], info['shape'][0]))

        auto = self.slides.positions(s['slide'], self.ann['sections'])
        spots, dead, here = [], [], None
        for key in self.keys:
            sec = self.sec(key)
            if int(sec['slide']) != int(s['slide']):
                continue
            xy = sec.get('slide_xy_px') or auto.get(key)
            if xy is None:
                continue
            junk = bool(sec.get('discarded'))
            (dead if junk else spots).append(dict(pos=xy))
            if junk:
                text, color = 'x', (200, 120, 120)
            else:
                text = self._index_text(key).strip()
                color = ((255, 255, 255) if key == self.keys[self.cur]
                         else (150, 210, 255))
            label = pg.TextItem(text, color=color, anchor=(0.5, -0.4))
            label.setPos(*xy)
            self.inset_vb.addItem(label)
            self.inset_texts.append(label)
            if key == self.keys[self.cur]:
                here = xy
        self.inset_dots.setData(spots)
        self.inset_dead.setData(dead)
        self.inset_here.setData([dict(pos=here)] if here else [])

        # tissue on the slide with no high-resolution image of its own
        gaps = []
        if len(info['blobs']) and (spots or dead):
            xy = np.array([sp['pos'] for sp in spots + dead], float)
            tol = (0.5 * float(np.median(info['boxes'][:, 2] - info['boxes'][:, 0]))
                   if len(info['boxes']) else np.inf)
            for b in info['blobs']:
                if np.sqrt(((xy - b) ** 2).sum(1)).min() > tol:
                    gaps.append(dict(pos=tuple(b)))
                    label = pg.TextItem('?', color=(255, 170, 60), anchor=(0.5, -0.4))
                    label.setPos(*b)
                    self.inset_vb.addItem(label)
                    self.inset_texts.append(label)
        self.inset_gaps.setData(gaps)
        self._layout_inset()

    def _inset_click(self, scene_pos):
        """Place the current section on the slide by hand."""
        p = self.inset_vb.mapSceneToView(scene_pos)
        self.sec()['slide_xy_px'] = [float(p.x()), float(p.y())]
        self.sec()['slide_xy_source'] = 'manual'
        self._mark_dirty()
        self._update_inset()
        self.flash('placed %s on slide %d by hand'
                   % (self.keys[self.cur], self.sec()['slide']))

    def relocate_on_slide(self, reorder=False):
        """
        Work this slide out again from the stage coordinates: drop hand-placed
        positions, and with reorder also drop hand-set orders and take the
        cutting order from the overview again.
        """
        if self.slides is None:
            return
        slide = self.sec()['slide']
        for k, sec in self.ann['sections'].items():
            if int(sec['slide']) != int(slide):
                continue
            sec['slide_xy_px'] = None
            sec['slide_xy_source'] = None
            if reorder:
                sec['order_source'] = 'default'
                sec['gap_source'] = 'default'
        apply_slide_layout(self.ann, self.slides, auto_order=self.auto_slide_order,
                           only_slide=slide, force=True,
                           annotate_unimaged=self.annotate_unimaged)
        self._mark_dirty()
        self._reorder()
        self._load_items()
        self._update_inset()
        self.flash('slide %d located from the stage coordinates%s'
                   % (slide, ' and reordered' if reorder else ''))

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
        crop = s.get('crop_px')
        try:
            if s.get('source') == 'slide_crop':
                img, full_shape, px = self.slides.full_image(s['slide'])
            else:
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
                                               and self.img.image is not None
                                               and crop is None) else None
            if levels is None:
                scale = img.shape[1] / float(full_shape[1])
                levels = display_levels(
                    img[int(crop[1] * scale):int(crop[3] * scale),
                        int(crop[0] * scale):int(crop[2] * scale)]
                    if crop is not None else img)
            self.img.setImage(img, autoLevels=False, levels=levels)
            self.img.setRect(QtCore.QRectF(0, 0, full_shape[1], full_shape[0]))
            self.home_rect = (QtCore.QRectF(crop[0], crop[1], crop[2] - crop[0],
                                            crop[3] - crop[1]) if crop is not None
                              else QtCore.QRectF(0, 0, full_shape[1], full_shape[0]))
            self.reset_view()
        except Exception as err:
            self.img.clear()
            self.flash('could not load %s: %s' % (s['file'], err), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._load_items()
        self._update_inset()
        self._refresh_panel()
        self._refresh_list()
        self.view.setFocus()

    def next_unreviewed(self):
        n = len(self.keys)
        for step in range(1, n + 1):
            j = (self.cur + step) % n
            s = self.sec(self.keys[j])
            if s['lhy'] is None and not s.get('discarded'):
                return self.goto(j)
        self.flash('every section has been reviewed')

    # ----------------------------------------------------- overlay items --
    def _all_items(self):
        items = ([self.midline_roi, self.surface_roi] + self.midline_labels
                 + self.ellipse_rois + self.dmdl_items)
        for roi in self.scar_rois:
            items += [roi, roi._text]
        return [it for it in items if it is not None]

    def _clear_items(self):
        for item in self._all_items():
            self.vb.removeItem(item)
        self.midline_roi = self.surface_roi = self.selected = None
        self.midline_labels, self.ellipse_rois, self.scar_rois = [], [], []
        self.dmdl_items = []

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
        for xy in s.get('dmdl_px', []):
            self._make_dmdl(xy)
        self._set_overlays_visible(self.overlays_visible)
        self._geom = None

    def _pull_items(self):
        """Copy the current geometry of every overlay into the section dict."""
        if self._loaded_key not in self.ann['sections']:
            return
        s = self.sec(self._loaded_key)
        s['midline_px'] = self._midline_points()
        s['surface_px'] = self._poly_points(self.surface_roi)
        s['ellipses'] = [self._ellipse_dict(r) for r in self.ellipse_rois]
        s['scars'] = [dict(shank=r._shank, points_px=self._poly_points(r))
                      for r in self.scar_rois]
        s['dmdl_px'] = [[float(t.pos().x()), float(t.pos().y())]
                        for t in self.dmdl_items]

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

    # DM/DL boundary at the surface (hippocampal width)
    def _make_dmdl(self, xy):
        t = pg.TargetItem(pos=list(xy), size=13, symbol='x',
                          pen=pg.mkPen((120, 255, 255), width=2), label='DM/DL',
                          labelOpts=dict(color=(120, 255, 255)))
        t.setZValue(23)
        t.sigPositionChangeFinished.connect(self._changed)
        self.vb.addItem(t)
        self.dmdl_items.append(t)
        return t

    def add_dmdl(self):
        self._make_dmdl(self._cursor_or_center())
        self._changed()
        self.flash('put it where the DM/DL boundary meets the dorsal surface; '
                   'its distance from the midline is the hippocampal width')

    def _hp_readout(self):
        geom = self._geometry()
        if not geom or not self.dmdl_items:
            return None
        ml, _ = geom.px_to_brain(np.array([[t.pos().x(), t.pos().y()]
                                           for t in self.dmdl_items], float))
        width = float(np.mean(np.abs(ml)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return width, float(lrt.hp_width_to_ap(width)), float(lrt.hp_ap_sd(width))

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
        if self.dmdl_items and self.mouse_xy is not None:
            (x0, x1), _ = self.vb.viewRange()
            d = [np.hypot(t.pos().x() - self.mouse_xy[0], t.pos().y() - self.mouse_xy[1])
                 for t in self.dmdl_items]
            j = int(np.argmin(d))
            if d[j] < 0.02 * abs(x1 - x0):
                self.vb.removeItem(self.dmdl_items.pop(j))
                self._changed()
                return
        self.flash('nothing to delete (click an ellipse, or hover a scar / DM-DL mark)')

    def copy_previous(self):
        for j in range(self.cur - 1, -1, -1):
            src = self.sec(self.keys[j])
            if src.get('discarded'):
                continue
            if src.get('midline_px') or src.get('ellipses'):
                break
        else:
            self.flash('no earlier section with annotations', error=True)
            return
        same_piece = (self.sec().get('piece_group') is not None and
                      self.sec().get('piece_group') == src.get('piece_group'))
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
        if same_piece:
            self.flash('copied %s from %s, ANOTHER PIECE OF THIS SAME SECTION -- '
                       'the pieces shifted apart when they were mounted, so this is '
                       'only a starting point: redraw it on this piece'
                       % (', '.join(copied) or 'nothing', self.keys[j]), error=True)
        else:
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

    # ------------------------------------------- not a section / broken up --
    def set_discarded(self, discarded=None):
        """
        Mark this entry as not being a section at all -- detritus, a bubble or
        a smear that was picked up as a region and imaged.  It stays in the
        list so it can be brought back, but it drops out of the AP series
        without being counted as a lost section.
        """
        s = self.sec()
        s['discarded'] = (not s.get('discarded')) if discarded is None \
            else bool(discarded)
        if s['discarded'] and s.get('piece_group'):
            s['piece_group'] = None          # junk cannot be a piece of a section
        if s['discarded'] and self.ann['series'].get('ac_section') == self.keys[self.cur]:
            self._set_series('ac_section', None)
            self._refresh_ac_label()
            self.flash('this was the AC reference section -- set it again with a',
                       error=True)
        self._mark_dirty(implant=True)
        self._reorder()
        self._load_items()
        self._update_inset()
        if s['discarded']:
            self.flash('%s is not a section: dropped from the AP count '
                       '(d puts it back)' % self.keys[self.cur])
        else:
            self.flash('%s counts as a section again' % self.keys[self.cur])

    def group_with_previous(self):
        """
        Join this entry to the previous one as another piece of the same
        broken section.  The whole group takes one step of the AP series;
        every piece keeps its own midline, surface, ellipses and scars.
        """
        s = self.sec()
        if s.get('discarded'):
            self.flash('this entry is marked as not a section -- press d first',
                       error=True)
            return
        prev = None
        for j in range(self.cur - 1, -1, -1):
            if not self.sec(self.keys[j]).get('discarded'):
                prev = self.keys[j]
                break
        if prev is None:
            self.flash('nothing before this to group it with', error=True)
            return
        gid = self.sec(prev).get('piece_group') or lrt.new_group_id(self.ann, prev)
        self.sec(prev)['piece_group'] = gid
        s['piece_group'] = gid
        self._mark_dirty()
        self._reorder()
        n = len(lrt.group_members(self.ann, self.keys[self.cur]))
        self.flash('%d pieces of one section, AP index %s -- mark the midline and '
                   'surface separately on each piece'
                   % (n, self.sec()['section_index']))

    def ungroup(self):
        """Take this entry out of its piece group; it is its own section again."""
        s = self.sec()
        if not s.get('piece_group'):
            self.flash('this entry is not part of a broken section')
            return
        gid = s['piece_group']
        s['piece_group'] = None
        left = [k for k, o in self.ann['sections'].items()
                if o.get('piece_group') == gid]
        if len(left) == 1:                   # a group of one is just a section
            self.ann['sections'][left[0]]['piece_group'] = None
        self._mark_dirty()
        self._reorder()
        self.flash('%s is its own section again -- it now takes its own step of '
                   'the AP series' % self.keys[self.cur])

    def add_lowres_companion(self):
        """
        Open a companion view of this section on the whole-slide overview, so
        the part of the section that the high-resolution scan missed can still
        be annotated.  It is a separate entry with its own pixels, midline and
        surface, grouped with the high-res one so the two share a single step
        of the AP series.
        """
        host = self.keys[self.cur]
        s = self.sec()
        if s.get('companion_of'):
            return self.goto(self.keys.index(s['companion_of']))
        key = '%s_lowres' % host
        if key in self.ann['sections']:
            return self.goto(self.keys.index(key))
        if s.get('source') == 'slide_crop':
            self.flash('this entry is already an overview crop', error=True)
            return
        if self.slides is None:
            self.flash('no slide overview images -- nothing to fall back to',
                       error=True)
            return
        info = self.slides.info(s['slide'])
        xy = s.get('slide_xy_px')
        if info is None or not len(info['boxes']) or xy is None:
            self.flash('this section has not been placed on its slide overview '
                       '-- click it in the inset first', error=True)
            return
        b = info['boxes'][int(np.argmin(np.sqrt(
            ((info['blobs'] - np.asarray(xy, float)) ** 2).sum(1))))]
        pad = 0.08 * max(b[2] - b[0], b[3] - b[1])
        crop = [float(b[0] - pad), float(b[1] - pad),
                float(b[2] + pad), float(b[3] + pad)]
        sec = lrt.empty_section(
            dict(file=info['file'], slide=s['slide'], region=5000 + s['region'],
                 source='slide_crop', crop_px=crop), s['slide_order'])
        sec['pixel_size_um'] = [info['px_um'], info['px_um']]
        sec['image_shape'] = [int(info['shape'][0]), int(info['shape'][1])]
        sec['slide_xy_px'] = [float(xy[0]), float(xy[1])]
        sec['slide_xy_source'] = s.get('slide_xy_source')
        sec['companion_of'] = host
        sec['lhy'] = s.get('lhy')
        self.ann['sections'][key] = sec
        self._mark_dirty()
        self._reorder()
        self.goto(self.keys.index(key))
        self.flash('overview companion of %s (%.1f um/px): draw its OWN midline '
                   'and surface, then annotate only what the high-res scan '
                   'missed -- c goes back' % (host, info['px_um']))

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
        a['order_source'] = b['order_source'] = 'manual'
        self._reorder()

    def set_shank(self, i):
        if i < self.shank_combo.count():
            self.shank_combo.setCurrentIndex(i)
            self.flash('current shank %s' % self.shank_combo.currentText())

    def set_ac_here(self):
        if self.sec().get('discarded'):
            self.flash('this entry is marked as not a section -- it cannot be the '
                       'AC reference', error=True)
            return
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
        s['order_source'] = 'manual'
        self._reorder()

    def _on_gap(self, value):
        if self.sec()['gap_before'] != value:
            self.sec()['gap_before'] = int(value)
            self.sec()['gap_source'] = 'manual'
            self._reorder()

    # -------------------------------------------------------------- events --
    def eventFilter(self, obj, ev):
        if obj is self.view and ev.type() == EV_RESIZE:
            self._layout_inset()
            return False
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
        elif txt == 'w':
            self.add_dmdl()
        elif txt == 'v':
            self.copy_previous()
        elif txt == 'x':
            self.set_flipped()
        elif txt == 'd':
            self.set_discarded()
        elif txt == 'c':
            self.add_lowres_companion()
        elif txt == 'g':
            self.group_with_previous()
        elif txt == 'G':
            self.ungroup()
        elif txt == '[':
            self.move_on_slide(-1)
        elif txt == ']':
            self.move_on_slide(1)
        elif txt == 'a':
            self.set_ac_here()
        elif txt == 'i':
            self.inset_visible = not self.inset_visible
            self._update_inset()
        elif txt == 'o':
            self.relocate_on_slide(reorder=False)
        elif txt == 'O':
            self.relocate_on_slide(reorder=True)
        elif txt == 'h':
            self.overlays_visible = not self.overlays_visible
            self._set_overlays_visible(self.overlays_visible)
        elif txt in ('f', 'z'):
            self.reset_view()
        elif txt in ('+', '='):
            self.zoom(1 / 1.25)
        elif txt in ('-', '_'):
            self.zoom(1.25)
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
        if ev.button() != LEFT_BUTTON:
            return
        if self.inset_vb.isVisible() and \
                self.inset_vb.sceneBoundingRect().contains(ev.scenePos()):
            self._inset_click(ev.scenePos())
            ev.accept()
            return
        if not self.drawing:
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

    def _piece_suffix(self, k):
        """'' for an intact section, 'a' / 'b' / ... for a piece of a broken one."""
        i, n = self._pieces.get(k, (1, 1))
        return '' if n < 2 else 'abcdefghijklmnopqrstuvwxyz'[(i - 1) % 26]

    def _index_text(self, k):
        s = self.sec(k)
        if s.get('discarded'):
            return '   -'
        if s.get('section_index') is None:
            return '   ?'
        return '%3d%s' % (s['section_index'], self._piece_suffix(k) or ' ')

    def _list_text(self, k):
        s = self.sec(k)
        mark = {None: '  .', 'yes': 'LHy', 'no': ' no', 'unsure': '  ?'}[s['lhy']]
        flags = ('F' if s.get('flipped') else ' ')
        flags += ('C' if s.get('companion_of')
                  else 'L' if s.get('source') == 'slide_crop'
                  else 'P' if s.get('crop_px') else ' ')
        if s.get('discarded'):
            return '%s  s%02d #%-2d r%03d %s  not a section' % (
                self._index_text(k), s['slide'], s['slide_order'], s['region'], flags)
        extra = ''
        if self._piece_suffix(k):
            extra += ' %dpc' % self._pieces[k][1]
        if s['ellipses']:
            extra += ' %de' % len(s['ellipses'])
        if s.get('scars'):
            extra += ' scar:' + ''.join(sorted({sc['shank'] for sc in s['scars']}))
        if s.get('dmdl_px'):
            extra += ' Hp'
        ac = '  [AC]' if self.ann['series'].get('ac_section') == k else ''
        gap = '  (+%d lost)' % s['gap_before'] if s.get('gap_before') else ''
        return '%s  s%02d #%-2d r%03d %s %s%s%s%s' % (
            self._index_text(k), s['slide'], s['slide_order'], s['region'],
            flags, mark, extra, ac, gap)

    def _refresh_list_item(self, i):
        item = self.section_list.item(i)
        if item is None:
            return
        k = self.keys[i]
        s = self.sec(k)
        item.setText(self._list_text(k))
        if s.get('discarded'):
            item.setBackground(QtGui.QBrush(QtGui.QColor(35, 35, 35)))
            item.setForeground(QtGui.QBrush(QtGui.QColor(130, 130, 130)))
        else:
            item.setBackground(QtGui.QBrush(QtGui.QColor(*STATUS_COLORS[s['lhy']])))
            item.setForeground(QtGui.QBrush(QtGui.QColor(240, 240, 240)))

    def _refresh_list(self):
        self._pieces = lrt.piece_labels(self.ann)
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
        split = lrt.noncontiguous_groups(self.ann)
        if split:
            self.flash('a whole section sits between the pieces of one broken '
                       'section (%s) -- check the grouping or the slide order'
                       % ', '.join(sorted({k for v in split.values() for k in v})),
                       error=True)

    def _refresh_ac_label(self):
        ac = self.ann['series'].get('ac_section')
        self.ac_label.setText(ac or 'unset')
        if self.keys:
            self._refresh_list()

    def _refresh_panel(self):
        s = self.sec()
        if s.get('companion_of'):
            low = ('&nbsp; <span style="color:#ffaa3c">[overview companion of %s, '
                   '%.1f um/px]</span>' % (s['companion_of'],
                                           np.mean(s['pixel_size_um'] or [np.nan])))
        elif s.get('source') == 'slide_crop':
            low = ('&nbsp; <span style="color:#ffaa3c">[overview crop, %.1f um/px]</span>'
                   % np.mean(s['pixel_size_um'] or [np.nan]))
        elif s.get('crop_px'):
            low = ('&nbsp; <span style="color:#7fd3ff">[part %s of a shared image]</span>'
                   % (s.get('part') or '?'))
        else:
            low = ''
        if s.get('discarded'):
            low += ('&nbsp; <span style="color:#ff6464">[not a section -- out of '
                    'the AP count]</span>')
        piece, n_pieces = self._pieces.get(self.keys[self.cur], (1, 1))
        if n_pieces > 1:
            low += ('&nbsp; <span style="color:#7fd3ff">[piece %d of %d of one '
                    'broken section]</span>' % (piece, n_pieces))
        idx = ('--' if s.get('section_index') is None
               else '%d%s' % (s['section_index'], self._piece_suffix(self.keys[self.cur])))
        self.file_label.setText('<b>%s</b> &nbsp; index %s &nbsp; (%d / %d)%s<br>%s'
                                % (self.keys[self.cur], idx,
                                   self.cur + 1, len(self.keys), low, s['file']))
        self.discard_box.setChecked(bool(s.get('discarded')))
        self.group_label.setText('whole' if n_pieces < 2
                                 else 'piece %d/%d' % (piece, n_pieces))
        self.discard_label.setVisible(bool(s.get('discarded')))
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
        try:
            self._update_status_inner()
        except Exception as err:            # a readout is never worth a crash
            self.status.setText('status unavailable: %s' % err)

    def _update_status_inner(self):
        if not self.keys or self.cur >= len(self.keys):
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
        hp = self._hp_readout()
        if hp is not None:
            parts.append('Hp %5.0f -> AP %s' % (
                hp[0], 'n/a' if not np.isfinite(hp[1]) else '%+.0f+-%.0f' % (hp[1], hp[2])))
        if s.get('discarded'):
            parts.append('NOT A SECTION')
        else:
            ap_rel, ap_abs = lrt.section_ap(self.ann)[self.keys[self.cur]]
            parts.append('AP-AC %s'
                         % ('%+.0f' % ap_rel if np.isfinite(ap_rel) else 'n/a'))
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
        on_disk = {i['file'] for i in found.values()}
        gone = sorted({secs[k]['file'] for k in secs
                       if secs[k].get('source', 'image') == 'image'
                       and secs[k]['file'] not in on_disk})
        if gone:
            print('  %d annotated image file(s) are not on disk: %s'
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
           ac_section_other=None,
           ac_ap_um=None,
           ac_dv_um=None,
           ap_anchor=None,
           implant_side=None,
           inplane_scale=None,
           shank_labels=('A',),
           pixel_size_override=None,
           show_slide_inset=True,
           slide_pattern=DEFAULT_SLIDE_PATTERN,
           slide_downsample=4,
           slide_axis_signs=None,
           auto_slide_order=True,
           annotate_unimaged=True,
           inset_width_px=380,
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
    if isinstance(ac_section_other, (tuple, list)):
        ac_section_other = lrt.section_key(*ac_section_other)
    series = OrderedDict(section_thickness_um=section_thickness_um,
                         section_interval=section_interval, ap_sign=ap_sign,
                         ac_section=ac_section, ac_ap_um=ac_ap_um,
                         ac_section_other=ac_section_other,
                         ac_dv_um=ac_dv_um, ap_anchor=ap_anchor,
                         implant_side=implant_side, inplane_scale=inplane_scale)
    ann = build_annotation(bird, hist_dir, annotation_path, file_pattern,
                           channel, series)
    if ann['series']['ac_section'] and ann['series']['ac_section'] not in ann['sections']:
        print('  WARNING: ac_section %s is not one of the sections'
              % ann['series']['ac_section'])

    images = ImageCache(hist_dir, downsample=display_downsample,
                        cache_dir=cache_dir)
    slides = None
    if show_slide_inset:
        slides = SlideOverview(hist_dir, slide_pattern=slide_pattern,
                               channel=channel, downsample=slide_downsample,
                               cache_dir=cache_dir, axis_signs=slide_axis_signs)
        print('  %d slide overview image(s) for the inset' % len(slides.files))
        if not slides.files:
            print('    (none matched %s -- the inset stays off)' % slide_pattern)
            slides = None
        else:
            apply_slide_layout(ann, slides, auto_order=auto_slide_order,
                               annotate_unimaged=annotate_unimaged)

    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = LHyROIGUI(ann, annotation_path, images,
                    pixel_size_override=pixel_size_override,
                    default_ellipse_axes_um=default_ellipse_axes_um,
                    dv_mode=dv_mode, ref_ml_um=ref_ml_um,
                    keep_levels=keep_levels, export_csv=export_csv,
                    shank_labels=shank_labels, slides=slides,
                    inset_width_px=inset_width_px,
                    auto_slide_order=auto_slide_order,
                    annotate_unimaged=annotate_unimaged)
    gui.resize(1550, 980)
    gui.show()
    gui.raise_()
    gui.activateWindow()
    return (getattr(app, 'exec', None) or app.exec_)()
