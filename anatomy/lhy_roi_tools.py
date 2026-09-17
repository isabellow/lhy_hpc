"""
lhy_roi_tools.py
================

Geometry, storage and analysis helpers for the lateral hypothalamus (LHy)
histology annotations made with lhy_roi_gui.py.  No Qt dependency, so this is
the module to import from analysis code:

    from lhy_roi_tools import load_lhy_rois
    rois = load_lhy_rois('.../LMN86_lhy_rois.json')

    # probe track from the traced scars, in get_probe_coords_lhy's format
    insert_coords, tip_coords = rois.probe_inputs()     # (n_shanks, 2), (n_shanks, 3)
    rois.scar_tracks()                                   # slopes, angles, fit quality

    # distance of channels / cells to the annotated LHy
    d = rois.distance(cell_pos)                          # (n, 3) [ML, AP, DV] um
    in_lhy = rois.label(cell_pos, tol_um=150)

WHAT IS STORED
--------------
The JSON stores raw image geometry only, in full-resolution pixels:

    midline_px  : [[x, y] dorsal point, [x, y] ventral point]
    surface_px  : [[x, y], ...] polyline along the dorsal brain surface
    ellipses    : [{center_px, axes_px, angle_deg}, ...]
    scars       : [{shank, points_px: [[x, y], ...]}, ...]  probe-scar traces
    slide_order : position of the section on its slide, in cutting order
    gap_before  : number of sections lost / not imaged just before this one
    flipped     : section was mounted mirror-imaged

Brain coordinates are derived at load time, so changing the thickness, the AC
reference, the DV convention or a flip flag never means redrawing anything.

COORDINATE CONVENTIONS  (match get_probe_coords_lhy.py: [ML, AP, DV] in um)
---------------------------------------------------------------------------
ML : signed distance from the section's midline, + = implanted hemisphere,
     i.e. the hemisphere the probe ENTERED (a track angled across the midline
     ends at negative ML).
     In image terms, +ML on an unflipped, dorsal-up section is image-right
     when implant_side = 'right'.  `flipped` sections have their ML negated.
     With implant_side = None the side is taken from the traced scars.
AP : sections are ordered by (slide, slide_order); section_index is that rank
     plus the cumulative gap_before.  Then
         ap_from_ac_um = ap_sign * (section_index - ac_index)
                         * section_thickness_um * section_interval * ap_scale
         ap_um         = ac_ap_um + ap_from_ac_um
     ap_sign = -1 when slides run anterior -> posterior.
DV : measured along the midline direction, + ventral, zero set by `dv_mode`:
        'local'   depth below the dorsal-surface polyline at the same ML [default]
        'ref_ml'  depth below the surface at a fixed ML (ref_ml_um), same side
        'midline' depth below the dorsal midline handle
"""

import os
import re
import json
import glob
import shutil
import warnings
from collections import OrderedDict

import numpy as np

try:
    import pandas as pd
except ImportError:          # the GUI does not need pandas; the loader does
    pd = None


FORMAT_VERSION = 2

# Slide1-N_Region000M_Channel395 nm_Seq0008.nd2
DEFAULT_FILE_PATTERN = (r'Slide\d+-(?P<slide>\d+)_Region(?P<region>\d+)'
                        r'_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.nd2$')

LHY_STATES = ('yes', 'no', 'unsure')

DEFAULT_SERIES_PARAMS = OrderedDict([
    ('section_thickness_um', None),  # cut thickness
    ('section_interval', 1),         # cut sections per step of section_index
    ('ap_sign', -1),                 # -1: higher section_index is more POSTERIOR
    ('ac_section', None),            # section key of the anterior commissure ref
    ('ac_ap_um', None),              # AP of that reference, relative to lambda
    ('implant_side', None),          # 'right' / 'left' image side, None = from scars
    ('inplane_scale', 1.0),          # multiply ML/DV by this (tissue shrinkage)
    ('ap_scale', 1.0),               # multiply AP spacing by this
])


# --------------------------------------------------------------------------- #
#  FILES                                                                       #
# --------------------------------------------------------------------------- #

def section_key(slide, region):
    return 'slide%02d_region%03d' % (int(slide), int(region))


def _norm_channel(name):
    """'395 nm', '395_nm' and '395nm ' all compare equal."""
    return re.sub(r'[\s_]+', '', str(name)).lower()


def discover_sections(hist_dir, file_pattern=DEFAULT_FILE_PATTERN,
                      channel=None, extensions=('.nd2',)):
    """
    Find the section images in hist_dir.

    Returns an OrderedDict  key -> dict(file, slide, region, channel, seq),
    sorted by (slide, region).  If one slide/region has several files
    (e.g. re-imaged), the highest Seq wins and a warning is printed.
    """
    rx = re.compile(file_pattern)
    found = {}
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(hist_dir, '*' + ext))
    for path in sorted(files):
        name = os.path.basename(path)
        m = rx.search(name)
        if m is None:
            continue
        g = m.groupdict()
        if channel is not None and _norm_channel(g.get('channel', '')) != _norm_channel(channel):
            continue
        slide, region = int(g['slide']), int(g['region'])
        seq = int(g.get('seq') or 0)
        key = section_key(slide, region)
        if key in found:
            warnings.warn('%s: several files for slide %d region %d -- keeping '
                          'the highest Seq' % (hist_dir, slide, region))
            if seq < found[key]['seq']:
                continue
        found[key] = dict(file=name, slide=slide, region=region,
                          channel=g.get('channel'), seq=seq)
    order = sorted(found, key=lambda k: (found[k]['slide'], found[k]['region']))
    return OrderedDict((k, found[k]) for k in order)


def empty_section(info, slide_order):
    return OrderedDict([
        ('file', info['file']),
        ('slide', info['slide']),
        ('region', info['region']),
        ('slide_order', int(slide_order)),
        ('gap_before', 0),
        ('flipped', False),
        ('section_index', None),       # derived; rewritten by assign_section_indices
        ('pixel_size_um', None),
        ('image_shape', None),
        ('lhy', None),
        ('midline_px', None),
        ('surface_px', None),
        ('ellipses', []),
        ('scars', []),
        ('notes', ''),
    ])


def next_slide_order(sections, slide):
    orders = [s['slide_order'] for s in sections.values() if s['slide'] == slide]
    return max(orders + [0]) + 1


def assign_section_indices(ann):
    """
    Order sections by (slide, slide_order, region) and write section_index =
    rank + cumulative gap_before.  Returns the ordered list of keys.
    """
    secs = ann['sections']
    keys = sorted(secs, key=lambda k: (secs[k]['slide'], secs[k]['slide_order'],
                                       secs[k]['region']))
    idx = 0
    for i, k in enumerate(keys):
        idx += int(secs[k].get('gap_before') or 0) + (1 if i else 0)
        secs[k]['section_index'] = idx
    return keys


def duplicate_slide_orders(ann):
    seen, dup = {}, []
    for k, s in ann['sections'].items():
        tag = (s['slide'], s['slide_order'])
        if tag in seen:
            dup.append((seen[tag], k))
        seen[tag] = k
    return dup


def new_annotation(bird, hist_dir, sections, series_params=None):
    """Fresh annotation dict for the sections returned by discover_sections."""
    params = OrderedDict(DEFAULT_SERIES_PARAMS)
    params.update(series_params or {})
    secs = OrderedDict()
    for k, info in sections.items():
        secs[k] = empty_section(info, next_slide_order(secs, info['slide']))
    ann = OrderedDict([
        ('format_version', FORMAT_VERSION),
        ('bird', bird),
        ('hist_dir', hist_dir),
        ('series', params),
        ('sections', secs),
    ])
    assign_section_indices(ann)
    return ann


def migrate_annotation(ann):
    """Bring an older annotation dict up to FORMAT_VERSION in place."""
    if ann.get('format_version', 1) < 2:
        secs = ann['sections']
        by_slide = {}
        for k in sorted(secs, key=lambda k: (secs[k]['slide'], secs[k]['region'])):
            s = secs[k]
            by_slide[s['slide']] = by_slide.get(s['slide'], 0) + 1
            s.setdefault('slide_order', by_slide[s['slide']])
            s.setdefault('gap_before', 0)
            s.setdefault('flipped', False)
            s.setdefault('scars', [])
            if s.pop('points', None):
                warnings.warn('%s: v1 tip points dropped -- trace the scar '
                              'instead' % k)
        ann['format_version'] = 2
    for name, default in DEFAULT_SERIES_PARAMS.items():
        ann['series'].setdefault(name, default)
    assign_section_indices(ann)
    return ann


def load_annotation(path):
    with open(path, 'r') as f:
        ann = json.load(f, object_pairs_hook=OrderedDict)
    if ann.get('format_version', 0) > FORMAT_VERSION:
        warnings.warn('%s was written by a newer version of lhy_roi_tools' % path)
    return migrate_annotation(ann)


def save_annotation(ann, path, backup=True):
    """Atomic write (tmp file + rename), keeping the previous file as .bak."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ann, f, indent=1, default=_json_default)
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + '.bak')
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


# --------------------------------------------------------------------------- #
#  PER-SECTION GEOMETRY                                                        #
# --------------------------------------------------------------------------- #

def qt_rotation(angle_deg):
    """Rotation matrix matching QGraphicsItem.setRotation (x right, y down)."""
    t = np.deg2rad(angle_deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t), np.cos(t)]])


def ellipse_boundary_px(ellipse, n=360):
    c = np.asarray(ellipse['center_px'], float)
    a, b = np.asarray(ellipse['axes_px'], float)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    return c + local @ qt_rotation(ellipse['angle_deg']).T


def inside_ellipse_px(xy_px, ellipse):
    xy = np.atleast_2d(np.asarray(xy_px, float))
    c = np.asarray(ellipse['center_px'], float)
    a, b = np.asarray(ellipse['axes_px'], float)
    local = (xy - c) @ qt_rotation(ellipse['angle_deg'])     # R^T (p - c)
    return (local[:, 0] / a) ** 2 + (local[:, 1] / b) ** 2 <= 1.0


def implant_side_sign(side):
    if side in (None, ''):
        return None
    side = str(side).lower()
    if side in ('right', 'image right', '+', '1'):
        return 1
    if side in ('left', 'image left', '-', '-1'):
        return -1
    raise ValueError("implant_side must be 'right', 'left' or None")


class SectionGeometry(object):
    """
    Maps between full-resolution image pixels and in-section brain coordinates
    (ML, DV) for one section.  Needs the midline and a pixel size.

    implant_sign : +1 if the implanted hemisphere is image-right on an
                   unflipped, dorsal-up section, -1 if image-left.  The
                   section's own `flipped` flag is applied on top.
    """

    def __init__(self, section, inplane_scale=1.0, dv_mode='local',
                 ref_ml_um=None, implant_sign=1):
        if section.get('midline_px') is None:
            raise ValueError('section has no midline')
        if section.get('pixel_size_um') is None:
            raise ValueError('section has no pixel size (open it in the GUI '
                             'once, or set pixel_size_override)')
        if dv_mode not in ('local', 'ref_ml', 'midline'):
            raise ValueError("dv_mode must be 'local', 'ref_ml' or 'midline'")
        if dv_mode == 'ref_ml' and ref_ml_um is None:
            raise ValueError("dv_mode='ref_ml' needs ref_ml_um")

        px = np.asarray(section['pixel_size_um'], float).ravel()
        self.px = (px if px.size == 2 else np.repeat(px[0], 2)) * inplane_scale
        self.dv_mode = dv_mode
        self.ref_ml_um = ref_ml_um

        dorsal, ventral = (np.asarray(p, float) for p in section['midline_px'])
        self.origin = dorsal * self.px
        axis = ventral * self.px - self.origin
        if np.linalg.norm(axis) == 0:
            raise ValueError('midline has zero length')
        self.dv_hat = axis / np.linalg.norm(axis)
        # image-right when dorsal is up, then flip / implant side
        sign = implant_sign * (-1 if section.get('flipped') else 1)
        self.ml_hat = sign * np.array([self.dv_hat[1], -self.dv_hat[0]])

        self.surface_ml = self.surface_u = None
        surf = section.get('surface_px')
        if surf is not None and len(surf) >= 2:
            ml, u = self._px_to_axis(np.asarray(surf, float))
            order = np.argsort(ml)
            self.surface_ml, self.surface_u = ml[order], u[order]

    def _px_to_axis(self, xy_px):
        q = np.atleast_2d(xy_px) * self.px - self.origin
        return q @ self.ml_hat, q @ self.dv_hat

    def px_to_um(self, xy_px):
        return np.atleast_2d(np.asarray(xy_px, float)) * self.px

    def surface_u_at(self, ml):
        ml = np.atleast_1d(np.asarray(ml, float))
        if self.surface_ml is None:
            return np.full(ml.shape, np.nan)
        return np.interp(ml, self.surface_ml, self.surface_u,
                         left=np.nan, right=np.nan)

    def dv_zero(self, ml):
        ml = np.atleast_1d(np.asarray(ml, float))
        if self.dv_mode == 'midline':
            return np.zeros(ml.shape)
        if self.dv_mode == 'local':
            return self.surface_u_at(ml)
        side = np.where(ml < 0, -1.0, 1.0)
        return self.surface_u_at(side * self.ref_ml_um)

    def px_to_brain(self, xy_px):
        """(n, 2) pixels -> (ml, dv) um."""
        ml, u = self._px_to_axis(xy_px)
        return ml, u - self.dv_zero(ml)

    def brain_to_px(self, ml, dv):
        ml = np.atleast_1d(np.asarray(ml, float))
        u = np.atleast_1d(np.asarray(dv, float)) + self.dv_zero(ml)
        q = ml[:, None] * self.ml_hat + u[:, None] * self.dv_hat + self.origin
        return q / self.px


def signed_distance_to_ellipse_um(xy_px, ellipse, geom, n=1440):
    """
    Distance (um, in the section plane) from each point to the ellipse
    boundary; negative inside.  The boundary is sampled densely, which is
    accurate to well under a micron at LHy scale.
    """
    xy_px = np.atleast_2d(np.asarray(xy_px, float))
    out = np.full(len(xy_px), np.nan)
    ok = np.all(np.isfinite(xy_px), axis=1)
    if not np.any(ok):
        return out
    bnd = geom.px_to_um(ellipse_boundary_px(ellipse, n=n))
    pts = geom.px_to_um(xy_px[ok])
    d = np.empty(len(pts))
    for i in range(0, len(pts), 512):
        diff = pts[i:i + 512, None, :] - bnd[None, :, :]
        d[i:i + 512] = np.sqrt((diff ** 2).sum(-1)).min(1)
    inside = inside_ellipse_px(xy_px[ok], ellipse)
    out[ok] = np.where(inside, -d, d)
    return out


def resample_polyline_px(points_px, geom, step_um=20.0):
    """Evenly spaced points (px) along a polyline, step_um apart in tissue."""
    p = np.asarray(points_px, float)
    if len(p) < 2:
        return p
    seg = np.sqrt((np.diff(geom.px_to_um(p), axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return p[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step_um)) + 1))
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


# --------------------------------------------------------------------------- #
#  SERIES (AP, HEMISPHERE)                                                     #
# --------------------------------------------------------------------------- #

def section_ap(ann, series=None):
    """
    {key: (ap_from_ac_um, ap_um)} for every section.  NaN where the AC
    reference or the thickness is not set yet.
    """
    p = OrderedDict(DEFAULT_SERIES_PARAMS)
    p.update(ann.get('series', {}))
    p.update(series or {})
    secs = ann['sections']
    out = OrderedDict()
    thick = p['section_thickness_um']
    ac = p['ac_section']
    if thick is None or ac is None or ac not in secs:
        for k in secs:
            out[k] = (np.nan, np.nan)
        return out
    step = float(thick) * float(p['section_interval']) * float(p['ap_scale'])
    ac_idx = secs[ac]['section_index']
    for k, s in secs.items():
        rel = float(p['ap_sign']) * (s['section_index'] - ac_idx) * step
        absolute = rel + float(p['ac_ap_um']) if p['ac_ap_um'] is not None else np.nan
        out[k] = (rel, absolute)
    return out


def resolve_implant_sign(ann, series=None, pixel_size_override=None,
                         entry_frac=0.25):
    """
    (+1 / -1 / None, source).  From series['implant_side'] if set, otherwise
    from the side of the midline where the scars ENTER the brain: the
    shallowest `entry_frac` of the traced depth range (flip-corrected).

    The entry side, not the side where most of the scar lies, defines the
    implanted hemisphere -- as in get_probe_coords_lhy, where a probe angled
    across the midline has + insertion ML and - tip ML.
    """
    p = OrderedDict(DEFAULT_SERIES_PARAMS)
    p.update(ann.get('series', {}))
    p.update(series or {})
    sign = implant_side_sign(p['implant_side'])
    if sign is not None:
        return sign, 'implant_side'
    ml_all, depth_all = [], []
    for s in ann['sections'].values():
        if not s.get('scars'):
            continue
        s = dict(s)
        if pixel_size_override is not None:
            s['pixel_size_um'] = pixel_size_override
        try:
            g = SectionGeometry(s, p['inplane_scale'], dv_mode='midline',
                                implant_sign=1)
        except ValueError:
            continue
        for scar in s['scars']:
            ml, dv = g.px_to_brain(resample_polyline_px(scar['points_px'], g))
            ml_all.append(ml)
            depth_all.append(dv)
    if not ml_all:
        return None, 'unknown'
    ml_all = np.concatenate(ml_all)
    depth_all = np.concatenate(depth_all)
    shallow = depth_all <= np.min(depth_all) + entry_frac * np.ptp(depth_all)
    ml_entry = ml_all[shallow]
    frac = np.mean(ml_entry > 0)
    if 0.2 < frac < 0.8:
        warnings.warn('the dorsal ends of the scar traces sit on both sides of the '
                      'midline (%.0f%% image-right after flip correction) -- check '
                      'flip flags or set implant_side' % (100 * frac))
    return (1 if np.median(ml_entry) > 0 else -1), 'scars'


# --------------------------------------------------------------------------- #
#  LOADING FOR ANALYSIS                                                        #
# --------------------------------------------------------------------------- #

class LHyROIs(object):
    """
    Loaded annotations for one bird.

    sections : DataFrame, one row per section
    rois     : DataFrame, one row per ellipse; 'boundary' holds an (n, 3)
               [ML, AP, DV] array for plotting
    scars    : DataFrame, one row per resampled scar-trace point
               (shank, key, section_index, ml_um, ap_um, ap_from_ac_um, dv_um)
    """

    def __init__(self, ann, dv_mode='local', ref_ml_um=None, n_boundary=180,
                 series=None, pixel_size_override=None, scar_step_um=20.0):
        if pd is None:
            raise ImportError('LHyROIs needs pandas')
        migrate_annotation(ann)
        self.ann = ann
        self.bird = ann.get('bird')
        self.dv_mode = dv_mode
        self.ref_ml_um = ref_ml_um
        self.series = OrderedDict(DEFAULT_SERIES_PARAMS)
        self.series.update(ann.get('series', {}))
        self.series.update(series or {})

        sign, source = resolve_implant_sign(ann, self.series, pixel_size_override)
        self.implant_sign, self.implant_source = sign, source
        if sign is None:
            warnings.warn('implanted hemisphere unknown (no scars traced and '
                          'implant_side unset) -- ML is image-right-positive and '
                          'distance() will mirror both hemispheres')

        aps = section_ap(ann, self.series)
        self.geoms = {}
        sec_rows, roi_rows, scar_rows = [], [], []
        for key, s in ann['sections'].items():
            s = dict(s)
            if pixel_size_override is not None:
                s['pixel_size_um'] = pixel_size_override
            ap_rel, ap_abs = aps[key]
            geom, why = None, ''
            try:
                geom = SectionGeometry(s, self.series['inplane_scale'],
                                       dv_mode=dv_mode, ref_ml_um=ref_ml_um,
                                       implant_sign=sign or 1)
            except ValueError as err:
                why = str(err)
            self.geoms[key] = geom
            sec_rows.append(dict(
                key=key, slide=s['slide'], region=s['region'],
                slide_order=s['slide_order'], gap_before=s['gap_before'],
                flipped=bool(s['flipped']), section_index=s['section_index'],
                lhy=s['lhy'], ap_from_ac_um=ap_rel, ap_um=ap_abs,
                has_midline=s.get('midline_px') is not None,
                has_surface=bool(s.get('surface_px')),
                n_ellipses=len(s.get('ellipses', [])),
                n_scars=len(s.get('scars', [])),
                geometry_error=why))

            if geom is None:
                if s.get('ellipses') or s.get('scars'):
                    warnings.warn('%s: annotations skipped -- %s' % (key, why))
                continue

            for j, e in enumerate(s.get('ellipses', [])):
                c_ml, c_dv = geom.px_to_brain(np.asarray(e['center_px'], float))
                b_ml, b_dv = geom.px_to_brain(ellipse_boundary_px(e, n_boundary))
                axes_um = np.asarray(e['axes_px'], float) * geom.px.mean()
                if np.any(np.isnan(b_dv)):
                    warnings.warn('%s ellipse %d: DV undefined for part of the '
                                  'boundary (surface polyline does not span its '
                                  'ML range?)' % (key, j))
                roi_rows.append(dict(
                    key=key, ellipse=j, section_index=s['section_index'],
                    lhy=s['lhy'], ap_from_ac_um=ap_rel, ap_um=ap_abs,
                    center_ml_um=c_ml[0], center_dv_um=c_dv[0],
                    semi_major_um=axes_um.max(), semi_minor_um=axes_um.min(),
                    boundary=np.column_stack([b_ml, np.full(b_ml.shape, ap_abs),
                                              b_dv])))

            for j, scar in enumerate(s.get('scars', [])):
                pts = resample_polyline_px(scar['points_px'], geom, scar_step_um)
                ml, dv = geom.px_to_brain(pts)
                for m, d in zip(ml, dv):
                    scar_rows.append(dict(shank=scar.get('shank', 'A'), key=key,
                                          trace=j, section_index=s['section_index'],
                                          ml_um=m, ap_um=ap_abs,
                                          ap_from_ac_um=ap_rel, dv_um=d))

        self.sections = pd.DataFrame(sec_rows)
        self.rois = pd.DataFrame(roi_rows)
        self.scars = pd.DataFrame(scar_rows, columns=[
            'shank', 'key', 'trace', 'section_index', 'ml_um', 'ap_um',
            'ap_from_ac_um', 'dv_um'])

        if len(self.rois) and self.rois['ap_um'].isna().all():
            warnings.warn('AP is NaN everywhere: set section_thickness_um, the '
                          'AC section and ac_ap_um (or use ap_ref="ac")')

    # ----------------------------------------------------------- scars ---- #
    def scar_tracks(self, ap_ref='lambda', shared_direction=False):
        """
        Fit a straight line to each shank's scar traces (total least squares).

        shared_direction : fit one common direction for all shanks (shanks are
            parallel), each shank keeping its own position.  More robust when
            some shanks are traced on only a few sections.

        Returns a DataFrame, one row per shank (sorted by label):
            tip_ml/ap/dv_um   deepest traced point projected onto the fitted
                              line (the tip estimate)
            deepest_*_um      the deepest traced point itself
            insert_ml/ap_um   fitted line extrapolated to DV = 0
            track_length_um   insertion -> tip, compare with 'final depth'
            dap_ddv, dml_ddv  slopes; ap_angle_deg / ml_angle_deg from vertical
                              (+ap = tip anterior of entry, +ml = tip lateral)
            n_sections, dv_span_um, rms_um
            worst_section     section whose trace fits worst -- an unticked
                              'flipped' box or a mis-ordered section shows up here
        AP slopes need the scar traced on >= 2 sections (NaN otherwise).
        """
        ap_col = {'lambda': 'ap_um', 'ac': 'ap_from_ac_um'}[ap_ref]
        sc = self.scars.dropna(subset=['ml_um', 'dv_um', ap_col])
        if len(sc) == 0:
            return pd.DataFrame()
        groups = OrderedDict((sh, g) for sh, g in sorted(sc.groupby('shank'),
                                                          key=lambda x: x[0]))
        pts = {sh: g[['ml_um', ap_col, 'dv_um']].to_numpy(float)
               for sh, g in groups.items()}
        cents = {sh: p.mean(0) for sh, p in pts.items()}

        def fit_dir(centered):
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            d = vt[0]
            return d if d[2] >= 0 else -d

        if shared_direction:
            common = fit_dir(np.concatenate([p - cents[sh] for sh, p in pts.items()]))

        rows = []
        for sh, p in pts.items():
            g = groups[sh]
            c = cents[sh]
            n_sec = g['key'].nunique()
            d = common if shared_direction else fit_dir(p - c)
            if n_sec < 2 and not shared_direction:
                warnings.warn('shank %s: scar traced on one section only -- no AP '
                              'slope; the track is assumed to stay in that section'
                              % sh)
                d = d.copy()
                d[1] = 0.0
                d /= np.linalg.norm(d)
            resid = (p - c) - np.outer((p - c) @ d, d)
            sec_rms = (pd.Series((resid ** 2).sum(1), index=g.index)
                       .groupby(g['key']).mean() ** 0.5)
            deepest = p[np.argmax(p[:, 2])]
            tip = c + ((deepest - c) @ d) * d
            insert = c + (-c[2] / d[2]) * d if abs(d[2]) > 1e-9 else np.full(3, np.nan)
            ap_ok = n_sec >= 2 or shared_direction
            rows.append(OrderedDict([
                ('shank', sh),
                ('tip_ml_um', tip[0]), ('tip_ap_um', tip[1]), ('tip_dv_um', tip[2]),
                ('deepest_ml_um', deepest[0]), ('deepest_ap_um', deepest[1]),
                ('deepest_dv_um', deepest[2]),
                ('insert_ml_um', insert[0]), ('insert_ap_um', insert[1]),
                ('track_length_um', np.linalg.norm(tip - insert)),
                ('dap_ddv', d[1] / d[2] if ap_ok else np.nan),
                ('dml_ddv', d[0] / d[2]),
                ('ap_angle_deg', np.rad2deg(np.arctan2(-d[1], d[2])) if ap_ok else np.nan),
                ('ml_angle_deg', np.rad2deg(np.arctan2(d[0], d[2]))),
                ('n_sections', n_sec),
                ('dv_span_um', np.ptp(p[:, 2])),
                ('rms_um', np.sqrt(np.mean((resid ** 2).sum(1)))),
                ('worst_section', sec_rms.idxmax()),
                ('worst_section_rms_um', sec_rms.max()),
                ('ap_ref', ap_ref),
            ]))
        out = pd.DataFrame(rows)
        # AP angle sign: + means the track runs posterior with depth
        out.attrs['note'] = ('ap_angle_deg > 0: tip posterior of entry; '
                             'ml_angle_deg > 0: tip lateral of entry')
        return out

    def probe_inputs(self, shanks=None, ap_ref='lambda', shared_direction=False):
        """
        insert_coords (n_shanks, 2) [ML, AP] and tip_coords (n_shanks, 3)
        [ML, AP, DV], um, in get_probe_coords_lhy's format, from the scar fits.
        `shanks` gives the label order (default sorted labels); shanks without
        a trace get NaN rows.
        """
        tr = self.scar_tracks(ap_ref=ap_ref, shared_direction=shared_direction)
        if shanks is None:
            shanks = list(tr['shank']) if len(tr) else []
        insert = np.full((len(shanks), 2), np.nan)
        tip = np.full((len(shanks), 3), np.nan)
        for i, sh in enumerate(shanks):
            row = tr[tr['shank'] == sh] if len(tr) else tr
            if len(row) == 0:
                continue
            r = row.iloc[0]
            insert[i] = [r['insert_ml_um'], r['insert_ap_um']]
            tip[i] = [r['tip_ml_um'], r['tip_ap_um'], r['tip_dv_um']]
        return insert, tip

    # ------------------------------------------------------------ LHy ----- #
    def _slabs(self, ap_col):
        """AP half-extent of each reviewed section: half the gap to each neighbour."""
        rev = self.sections[self.sections['lhy'].notna()].sort_values(ap_col)
        ap = rev[ap_col].to_numpy(float)
        step = abs(float(self.series['section_thickness_um'] or 0)
                   * float(self.series['section_interval'])
                   * float(self.series['ap_scale']))
        lo = np.empty(len(ap))
        hi = np.empty(len(ap))
        if len(ap):
            gaps = np.diff(ap)
            lo[0], hi[-1] = step / 2, step / 2
            lo[1:], hi[:-1] = gaps / 2, gaps / 2
        return dict(zip(rev['key'], zip(ap, lo, hi)))

    def distance(self, positions, ap_ref='lambda', mirror=None):
        """
        Approximate 3-D distance (um) from each position to the annotated LHy.

        positions : (n, 3) [ML, AP, DV] um, as get_probe_coords_lhy returns
                    (signed ML, + = implanted hemisphere).
        mirror    : treat both hemispheres as one (compare |ML| against every
                    ellipse reflected onto the implanted side).  Default: only
                    if the implanted hemisphere is unknown.

        Each section's ellipses are a slab spanning half the distance to the
        reviewed neighbours.  Per ellipse, the position is placed in that
        section's image (so its own surface sets DV), the in-plane signed
        distance d is measured, and
            dist = d                      if AP falls within the slab
                 = sqrt(max(d,0)^2 + g^2) otherwise (g = AP gap to the slab)
        The minimum over ellipses is returned; negative = inside.
        """
        if mirror is None:
            mirror = self.implant_sign is None
        ap_col = {'lambda': 'ap_um', 'ac': 'ap_from_ac_um'}[ap_ref]
        pos = np.atleast_2d(np.asarray(positions, float))
        n = len(pos)
        best = np.full(n, np.inf)
        best_in = np.full(n, np.nan)
        best_gap = np.full(n, np.nan)
        best_key = np.full(n, None, dtype=object)
        best_e = np.full(n, -1)

        rois = self.rois[self.rois['lhy'] == 'yes'] if len(self.rois) else self.rois
        if len(rois) == 0:
            warnings.warn('no ellipses on sections marked lhy=yes')
        slabs = self._slabs(ap_col)

        for _, r in rois.iterrows():
            geom = self.geoms[r['key']]
            e = self.ann['sections'][r['key']]['ellipses'][r['ellipse']]
            ap_c, lo, hi = slabs[r['key']]
            gap = np.maximum(0.0, np.maximum((ap_c - lo) - pos[:, 1],
                                             pos[:, 1] - (ap_c + hi)))
            ml = (np.sign(r['center_ml_um']) or 1) * np.abs(pos[:, 0]) if mirror else pos[:, 0]
            d = signed_distance_to_ellipse_um(geom.brain_to_px(ml, pos[:, 2]), e, geom)
            total = np.where(gap > 0, np.sqrt(np.maximum(d, 0) ** 2 + gap ** 2), d)
            better = np.isfinite(total) & (total < best)
            best[better] = total[better]
            best_in[better] = d[better]
            best_gap[better] = gap[better]
            best_key[better] = r['key']
            best_e[better] = r['ellipse']

        best[~np.isfinite(best)] = np.nan
        return pd.DataFrame(dict(dist_um=best, inplane_um=best_in,
                                 ap_gap_um=best_gap, nearest_key=best_key,
                                 nearest_ellipse=best_e))

    def label(self, positions, tol_um=150.0, ap_ref='lambda', mirror=None):
        """True where the position is inside LHy or within tol_um of it."""
        return (self.distance(positions, ap_ref=ap_ref, mirror=mirror)['dist_um']
                .le(tol_um).to_numpy())

    def boundary_points(self):
        """All ellipse boundaries stacked, (N, 3) [ML, AP, DV], for plotting."""
        if len(self.rois) == 0:
            return np.zeros((0, 3))
        return np.concatenate(self.rois['boundary'].to_list())


def load_lhy_rois(path, dv_mode='local', ref_ml_um=None, n_boundary=180,
                  pixel_size_override=None, scar_step_um=20.0, **series_overrides):
    """
    Load an annotation JSON for analysis.  Any series parameter
    (section_thickness_um, ac_ap_um, ap_sign, implant_side, inplane_scale, ...)
    can be overridden by keyword without touching the file.
    """
    return LHyROIs(load_annotation(path), dv_mode=dv_mode, ref_ml_um=ref_ml_um,
                   n_boundary=n_boundary, series=series_overrides,
                   pixel_size_override=pixel_size_override,
                   scar_step_um=scar_step_um)
