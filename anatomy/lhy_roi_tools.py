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
    ac_px       : [[x, y], ...] the anterior commissure on the AC reference
                  section, one point per hemisphere.  Its depth below the
                  dorsal surface is the lever arm ac_dv_um that turns the
                  AC-vs-hippocampus offset into a section-plane angle, so it
                  is measured the same way as every other DV in this file
    dmdl_px     : [[x, y], ...] the DM/DL boundary where it meets the dorsal
                  surface, one point per hemisphere; its distance from the
                  midline is the hippocampal width, which gives an AP estimate
                  from a landmark at the surface rather than from the AC
    source      : 'image' for a high-resolution scan, or 'slide_crop' for a
                  section annotated on the whole-slide overview.  crop_px is
                  the part of that image this section occupies -- set both for
                  overview crops and for one section of an image that covers
                  two.  Pixels, and so all annotations, are always in the
                  coordinates of the image named by `file`.
    slide_xy_px : position of this section on its slide overview image, from the
                  nd2 stage coordinates or set by hand in the GUI; with
                  slide_order it also fixes the cutting order (display only)
    slide_order : position of the section on its slide, in cutting order
    gap_before  : sections cut between this one and the previous one that are
                  not in the series -- lost, or on the slide but never imaged
                  at high resolution (counted from the slide overview)
    flipped     : section was mounted mirror-imaged
    discarded   : this entry is not a section at all (detritus, a bubble, a
                  smear that got picked up as a region and imaged).  It keeps
                  its place in the GUI list so it can be un-discarded, but it
                  takes no section_index, contributes nothing to the AP count
                  and is NOT counted as a lost section either -- the slot it
                  occupies on the slide is simply skipped
    piece_group : a section that broke into several pieces during mounting has
                  one entry per piece, all sharing this group id.  The whole
                  group takes ONE section_index (one step of the AP series)
                  and each piece keeps its own midline / surface / ellipses /
                  scars, so every annotation is compared only against the
                  landmarks on its own piece.  None for an intact section

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
     plus the cumulative gap_before.  Discarded entries are left out entirely,
     and all the pieces of one broken section share a single rank.  Then
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


FORMAT_VERSION = 4

# Slide1-N_Region000M_Channel395 nm_Seq0008.nd2
DEFAULT_FILE_PATTERN = (r'Slide\d+-(?P<slide>\d+)_Region(?P<region>\d+)'
                        r'_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.nd2$')

LHY_STATES = ('yes', 'no', 'unsure')

DEFAULT_SERIES_PARAMS = OrderedDict([
    ('section_thickness_um', None),  # cut thickness
    ('section_interval', 1),         # cut sections per step of section_index
    ('ap_sign', -1),                 # -1: higher section_index is more POSTERIOR
    ('ac_section', None),            # section key of the anterior commissure ref
    ('ac_section_other', None),      # the AC in the OTHER hemisphere, if it sits
                                     # on a different section (slicing yaw)
    ('ac_ap_um', None),              # AP of that reference, relative to lambda
    ('implant_side', None),          # 'right' / 'left' image side, None = from scars
    ('hp_L_um', 'auto'),             # hippocampus plateau width: 'auto' = fit this
                                     # bird, None = atlas, or a number in um
    ('ap_anchor', 'ac'),             # 'ac', 'hp', or 'shear' (both, by depth)
    ('ac_dv_um', None),              # depth of the AC below the surface, for the angle estimate
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
        ('source', info.get('source', 'image')),   # 'image' or 'slide_crop'
        ('crop_px', info.get('crop_px')),          # [x0, y0, x1, y1]: part of the image to show
        ('part', None),                            # 'j/n' when an image covers several sections
        ('superseded', False),                     # replaced by a better image, or split up
        ('discarded', False),                      # not a section: detritus, imaged by mistake
        ('piece_group', None),                     # pieces of one broken section share this id
        ('companion_of', None),                    # overview crop used to annotate the part of
                                                   # this section the high-res scan missed
        ('slide', info['slide']),
        ('region', info['region']),
        ('slide_order', int(slide_order)),
        ('gap_before', 0),
        ('flipped', False),
        ('slide_xy_px', None),         # where this section sits on the slide overview
        ('slide_xy_source', None),      # 'auto' (stage metadata) or 'manual'
        ('order_source', 'default'),    # 'default', 'slide' (from the overview) or 'manual'
        ('gap_source', 'default'),      # what set gap_before: 'default', 'slide' or 'manual'
        ('section_index', None),       # derived; rewritten by assign_section_indices
        ('pixel_size_um', None),
        ('image_shape', None),
        ('lhy', None),
        ('midline_px', None),
        ('surface_px', None),
        ('ellipses', []),
        ('scars', []),
        ('dmdl_px', []),               # DM/DL boundary at the surface, one point per side
        ('ac_px', []),                 # anterior commissure, one point per hemisphere:
                                       # its depth below the surface is ac_dv_um
        ('notes', ''),
    ])


def next_slide_order(sections, slide):
    orders = [s['slide_order'] for s in sections.values() if s['slide'] == slide]
    return max(orders + [0]) + 1


def group_id(ann, key):
    """
    Which section this entry belongs to: its piece_group if it is one piece of
    a broken section, the host's group if it is an overview companion, and
    otherwise the key itself.  Everything in one group counts as a single
    section in the AP series.
    """
    secs = ann['sections']
    host = secs[key].get('companion_of')
    if host in secs and host != key:
        return group_id(ann, host)
    return secs[key].get('piece_group') or key


def _own(ann, key):
    """Sort position of one entry; a companion sits right after its host."""
    secs = ann['sections']
    s = secs[key]
    host = s.get('companion_of')
    if host in secs and host != key:
        h = secs[host]
        return (h['slide'], h['slide_order'], h['region'], 1)
    return (s['slide'], s['slide_order'], s['region'], 0)


def group_members(ann, key):
    """Every entry in the same piece group as `key` (itself included), in order."""
    gid = group_id(ann, key)
    return [k for k in display_keys(ann) if group_id(ann, k) == gid]


def new_group_id(ann, key):
    """An unused piece-group id, based on the key it is seeded from."""
    base = 'grp_%s' % key
    used = {s.get('piece_group') for s in ann['sections'].values()}
    gid, n = base, 1
    while gid in used:
        n += 1
        gid = '%s_%d' % (base, n)
    return gid


def _sort_keys(ann, keys):
    """
    Sort entries by (slide, slide_order, region), keeping the pieces of one
    broken section together: a group sits where its earliest piece sits.
    """
    def own(k):
        return _own(ann, k)

    first = {}
    for k in keys:
        gid = group_id(ann, k)
        if gid not in first or own(k) < first[gid]:
            first[gid] = own(k)
    return sorted(keys, key=lambda k: (first[group_id(ann, k)], own(k)))


def display_keys(ann):
    """
    Every entry the GUI should list, in order -- including discarded ones, so
    they can be looked at and un-discarded.  Superseded entries are left out.
    """
    secs = ann['sections']
    return _sort_keys(ann, [k for k in secs if not secs[k].get('superseded')])


def series_keys(ann):
    """The entries that make up the AP series: not superseded, not discarded."""
    secs = ann['sections']
    return _sort_keys(ann, [k for k in secs if not secs[k].get('superseded')
                            and not secs[k].get('discarded')])


def piece_labels(ann):
    """
    {key: (i, n)} -- this entry is piece i of n of its section.  (1, 1) for an
    intact section.
    """
    out = OrderedDict()
    seen = OrderedDict()
    for k in display_keys(ann):
        gid = group_id(ann, k)
        seen.setdefault(gid, []).append(k)
    for gid, members in seen.items():
        for i, k in enumerate(members):
            out[k] = (i + 1, len(members))
    return out


def assign_section_indices(ann):
    """
    Order sections by (slide, slide_order, region) and write section_index =
    rank + cumulative gap_before.

    Superseded entries and discarded ones (detritus that was imaged by
    mistake) are left out and get section_index None; a discard is NOT counted
    as a lost section, it simply does not exist.  All the pieces of a broken
    section share one section_index, so a break costs no AP steps; the group
    takes the gap_before of its first piece.

    Returns the ordered list of keys in the series.
    """
    secs = ann['sections']
    keys = series_keys(ann)
    for k in secs:
        if secs[k].get('superseded') or secs[k].get('discarded'):
            secs[k]['section_index'] = None
    idx = 0
    prev_gid = None
    for k in keys:
        gid = group_id(ann, k)
        if gid != prev_gid:
            idx += int(secs[k].get('gap_before') or 0) + (1 if prev_gid is not None else 0)
            prev_gid = gid
        secs[k]['section_index'] = idx
    return keys


def duplicate_slide_orders(ann):
    seen, dup = {}, []
    for k, s in ann['sections'].items():
        if s.get('superseded') or s.get('discarded') or s.get('companion_of'):
            continue
        tag = (s['slide'], s['slide_order'])
        if tag in seen:
            dup.append((seen[tag], k))
        seen[tag] = k
    return dup


def noncontiguous_groups(ann):
    """
    Piece groups with another section sitting between their pieces, in slide
    order.  A section cannot have been cut between two pieces of itself, so
    this means either the grouping or the slide order is wrong.  Returns
    {group id: [keys in the way]}.
    """
    secs = ann['sections']
    live = [k for k in secs if not secs[k].get('superseded')
            and not secs[k].get('discarded') and not secs[k].get('companion_of')]
    live.sort(key=lambda k: _own(ann, k))
    out = OrderedDict()
    for gid in {secs[k].get('piece_group') for k in live} - {None}:
        at = [i for i, k in enumerate(live) if secs[k].get('piece_group') == gid]
        if len(at) < 2:
            continue
        between = [live[i] for i in range(at[0], at[-1] + 1)
                   if secs[live[i]].get('piece_group') != gid]
        if between:
            out[gid] = between
    return out


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
    if ann.get('format_version', 2) < 3:
        ann['format_version'] = 3
    if ann.get('format_version', 3) < 4:
        ann['format_version'] = 4
    for s in ann['sections'].values():          # fields added along the way
        s.setdefault('source', 'image')
        s.setdefault('crop_px', None)
        s.setdefault('part', None)
        s.setdefault('superseded', False)
        s.setdefault('discarded', False)
        s.setdefault('piece_group', None)
        s.setdefault('companion_of', None)
        s.setdefault('dmdl_px', [])
        s.setdefault('ac_px', [])
        for field in ('dmdl_px', 'ac_px', 'ellipses', 'scars'):
            if s.get(field) is None:       # written as null by an earlier version
                s[field] = []
        s.setdefault('slide_xy_px', None)
        s.setdefault('slide_xy_source', None)
        s.setdefault('order_source', 'default')
        s.setdefault('gap_source', 'manual' if s.get('gap_before') else 'default')
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
#  HIPPOCAMPAL WIDTH AS AN AP LANDMARK                                         #
# --------------------------------------------------------------------------- #
#
# Width of the hippocampus (the DM/DL boundary's distance from the midline at
# the brain surface) against AP, fitted to 848 measurements from 11 atlas
# brains (hippocampusWidths.fig).  A Gompertz curve,
#
#     width(ap) = L * exp(-exp((ap - ap0) / k))         ap in um from lambda
#
# fits them with an RMS of 156 um, with no bias anywhere along the range.  It
# has the same three parameters as a plain logistic but is asymmetric: it
# approaches its anterior plateau slowly and falls away gently posteriorly,
# which is what the measurements do (a logistic gives RMS 163 and runs ~90 um
# low posterior of 4.5 mm), and its upper asymptote sits above all but 2% of
# the measured widths, so few sections fall outside the invertible range.
#
# Most of that spread is between birds, not within one: fitting each bird
# separately leaves an RMS of only 72 um, while the birds' curves differ from
# each other by ~150 um.  So resid_sd is the right uncertainty for a single
# section from a new bird, and DM/DL marks on several sections of one bird
# share most of their error rather than averaging it away.
#
# Because the landmark sits AT THE SURFACE, an AP from it is much less
# sensitive to the section angle than one measured from a deep landmark such
# as the anterior commissure -- which is what makes comparing the two useful.
HP_WIDTH_FIT = dict(L=2285.8, ap0=3682.6, k=1001.8, resid_sd=156.4,
                    within_bird_sd=72.0, n_brains=11, n_points=848)

# Left/right asymmetry, from the 403 sections in hippocampusWidths.fig that
# carry a measurement for BOTH hemispheres of the same brain.
#
# The two hemispheres of one section differ in width by 66 um (median), but
# most of that is measurement scatter: |difference| is roughly flat at ~85 um
# across the whole width range, which is the signature of independent error on
# each boundary mark rather than of geometry (a slicing yaw would peak where
# the width-AP curve is steepest and collapse at small widths, which it does
# not) or of a proportional size difference (which would grow with width).
#
# Underneath that scatter there IS a consistent per-brain offset.  Taking the
# sd-weighted mean over each brain's own sections -- essential, because the
# unweighted mean is dominated by the flat ends of the curve where width
# carries almost no AP information -- the hemispheres of one brain differ in
# apparent AP by 73 um (SD across brains), at most 172 um, against a
# per-brain standard error of ~65 um.
#
# So a per-hemisphere AP anchor is worth having, but it is a small correction
# and its cause is ambiguous: a 172 um offset is only ~1.6 deg of slicing yaw
# at a 1500 um half-width, and equally consistent with genuine asymmetry.
# Anchoring each hemisphere on its own marks is agnostic about which it is;
# reading a yaw angle off it would not be.
HP_HEMISPHERE_FIT = dict(offset_sd=73.0, offset_mean_abs=58.0, offset_max=172.0,
                         paired_sections=403, n_brains=11, se_per_brain=65.0)

# a per-bird hippocampus size is only used when the marks pin it down this
# well, and only if the answer is anatomically plausible
HP_SIZE_FIT_MAX_SE = 200.0        # um
HP_SIZE_RANGE = (1200.0, 3000.0)  # um
HP_SIZE_FIT_MAX_MIN_WIDTH = 0.62  # the narrowest mark must be under this x L


def fit_hp_size(section_index, width_um, step_um=100.0, fit=None):
    """
    Fit this bird's OWN hippocampal width curve to its DM/DL marks.

    Hippocampus size varies a lot between birds -- enough that inverting a
    bird's widths through the atlas plateau L can bias its AP badly, and in
    an AP-dependent way.  Here the atlas curve SHAPE is kept (k fixed) but the
    plateau width L is fitted to this bird, along with an AP scale factor that
    doubles as a check on the section thickness.

    Only the shape of width against section index is used -- the AP origin is
    free -- so the fitted L does not depend on the anterior commissure
    reference, and the AC-vs-hippocampus offset stays a meaningful diagnostic.

    This only works when the marks reach the steep part of the curve.  Marks
    confined to the plateau cannot separate "small hippocampus" from "sections
    closer together in AP", and the fit says so through its standard errors.

    Returns dict(ok, L_um, L_se_um, scale, scale_se, step_um, rms_um, n, why).
    """
    f = dict(fit or HP_WIDTH_FIT)
    i = np.asarray(section_index, float)
    w = np.asarray(width_um, float)
    good = np.isfinite(i) & np.isfinite(w) & (w > 0)
    i, w = i[good], w[good]
    out = dict(ok=False, L_um=f['L'], L_se_um=np.nan, scale=np.nan,
               scale_se=np.nan, step_um=np.nan, rms_um=np.nan, n=int(len(w)),
               why='')
    if len(w) < 6:
        out['why'] = 'only %d marks: need at least 6' % len(w)
        return out
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        out['why'] = 'scipy not installed'
        return out

    k = f['k']

    def model(idx, L, scale, c):
        return L * np.exp(-np.exp((-idx * scale * step_um + c) / k))

    try:
        p, cov = curve_fit(model, i, w, p0=[f['L'], 1.0, 0.0], maxfev=40000)
        se = np.sqrt(np.diag(cov))
    except Exception as err:                       # pragma: no cover
        out['why'] = 'fit failed (%s)' % err
        return out
    L, scale, c = p
    out.update(L_um=float(L), L_se_um=float(se[0]), scale=float(scale),
               scale_se=float(se[1]), step_um=float(scale * step_um),
               rms_um=float(np.sqrt(np.mean((w - model(i, *p)) ** 2))))
    if not np.isfinite(se[0]) or se[0] > HP_SIZE_FIT_MAX_SE:
        out['why'] = ('the marks do not pin the hippocampus size down '
                      '(L = %.0f +- %.0f um): they need to reach the steep part '
                      'of the curve, below about %.0f um wide'
                      % (L, se[0], 0.55 * f['L']))
        out['L_um'] = f['L']
        return out
    if not (HP_SIZE_RANGE[0] <= L <= HP_SIZE_RANGE[1]):
        out['why'] = 'fitted hippocampus size %.0f um is not plausible' % L
        out['L_um'] = f['L']
        return out
    # The standard error alone is not enough: marks confined to the plateau
    # can still return a confident-looking L that is biased low, and using it
    # makes AP worse than the atlas value.  Demand that the marks actually
    # reach the steep part of the curve.
    if w.min() > HP_SIZE_FIT_MAX_MIN_WIDTH * L:
        out['why'] = ('the narrowest mark (%.0f um) is still on the plateau: to '
                      'fit this bird\'s hippocampus size, mark sections where it '
                      'is under about %.0f um wide'
                      % (w.min(), HP_SIZE_FIT_MAX_MIN_WIDTH * L))
        out['L_um'] = f['L']
        return out
    out['ok'] = True
    out['why'] = 'fitted from %d marks (widths %.0f - %.0f um)' % (
        len(w), w.min(), w.max())
    return out


def bird_hp_fit(L_um, fit=None):
    """The atlas width fit with this bird's own plateau width substituted."""
    f = dict(fit or HP_WIDTH_FIT)
    f['L'] = float(L_um)
    # fitting L removes the between-bird size spread from the residual, so
    # what is left is the within-bird scatter
    f['resid_sd'] = float(f.get('within_bird_sd', f['resid_sd']))
    return f


def ap_to_hp_width(ap_um, fit=None):
    """Expected hippocampal width (um) at an AP (um from lambda, + anterior)."""
    f = fit or HP_WIDTH_FIT
    return f['L'] * np.exp(-np.exp((np.asarray(ap_um, float) - f['ap0']) / f['k']))


def hp_width_to_ap(width_um, fit=None):
    """
    AP (um from lambda) for a hippocampal width (um) -- the inverse of the
    Gompertz above.  NaN where the width is outside the fitted range.
    """
    f = fit or HP_WIDTH_FIT
    w = np.asarray(width_um, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ap = f['ap0'] + f['k'] * np.log(np.log(f['L'] / w))
    bad = ~np.isfinite(ap) | (w <= 0) | (w >= f['L'])
    if np.any(bad):
        warnings.warn('hippocampal width outside the fitted range (0 - %.0f um): '
                      'AP undefined there' % f['L'])
    return np.where(bad, np.nan, ap)


def hp_ap_sd(width_um, fit=None):
    """
    Rough SD (um) of an AP estimated from a hippocampal width, from the spread
    of the atlas measurements propagated through the fit.  About 230 um near a
    width of 1000 um, where the curve is steepest, and several hundred um
    either side of that -- the anterior plateau carries little AP information.
    """
    f = fit or HP_WIDTH_FIT
    w = np.asarray(width_um, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dap_dw = f['k'] / (w * np.log(f['L'] / w))
    return np.where((w > 0) & (w < f['L']), np.abs(dap_dw) * f['resid_sd'], np.nan)


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
    if ac_idx is None:                       # the AC section itself was set aside
        for k in secs:
            out[k] = (np.nan, np.nan)
        return out
    for k, s in secs.items():
        if s.get('section_index') is None:   # superseded: not part of the series
            out[k] = (np.nan, np.nan)
            continue
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
        if not s.get('scars') or s.get('superseded') or s.get('discarded'):
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

    sections : DataFrame, one row per section, including hp_width_um / ap_hp_um
               (AP from the hippocampal-width landmark) and ap_ac_um (AP from
               the anterior commissure)
    rois     : DataFrame, one row per ellipse; 'boundary' holds an (n, 3)
               [ML, AP, DV] array for plotting
    scars    : DataFrame, one row per resampled scar-trace point
               (shank, key, section_index, ml_um, ap_um, ap_from_ac_um, dv_um)
    discarded: keys of entries marked as not being sections at all (detritus
               that was imaged by mistake); they are left out of everything
               above and take no place in the AP series
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
        self.discarded = []
        pieces = piece_labels(ann)
        sec_rows, roi_rows, scar_rows, hp_rows, ac_rows = [], [], [], [], []
        for key, s in ann['sections'].items():
            if s.get('superseded'):
                if s.get('ellipses') or s.get('scars') or s.get('dmdl_px'):
                    warnings.warn('%s: this entry was superseded (a better image, or '
                                  'an image split into its separate sections), so its '
                                  'annotations are ignored -- redraw them' % key)
                continue
            if s.get('discarded'):
                self.discarded.append(key)
                if s.get('ellipses') or s.get('scars') or s.get('dmdl_px'):
                    warnings.warn('%s: marked as not a section (discarded), so its '
                                  'annotations are ignored -- un-discard it in the '
                                  'GUI if that is wrong' % key)
                continue
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
            width = np.nan
            if geom is not None and s.get('dmdl_px'):
                ml, _ = geom.px_to_brain(np.asarray(s['dmdl_px'], float))
                width = float(np.mean(np.abs(ml)))
                for m in ml:                # one constraint per hemisphere
                    hp_rows.append(dict(
                        key=key, section_index=s['section_index'],
                        side=1 if m >= 0 else -1, ml_um=float(m),
                        width_um=abs(float(m)), ap_hp_um=np.nan,
                        ap_hp_sd_um=np.nan, ap_series_um=ap_abs))
            if geom is not None and s.get('ac_px'):
                aml, adv = geom.px_to_brain(np.asarray(s['ac_px'], float))
                for m, d in zip(aml, adv):
                    ac_rows.append(dict(key=key, section_index=s['section_index'],
                                        side=1 if m >= 0 else -1,
                                        ml_um=float(m), dv_um=float(d)))
            piece, n_pieces = pieces.get(key, (1, 1))
            sec_rows.append(dict(
                key=key, slide=s['slide'], region=s['region'],
                hp_width_um=width, ap_hp_um=np.nan, ap_hp_sd_um=np.nan,
                piece_group=s.get('piece_group'), piece=piece, n_pieces=n_pieces,
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
                    extra = ('  Every piece of a broken section needs its own '
                             'midline and surface: pixel distances do not carry '
                             'across pieces.' if n_pieces > 1 else '')
                    warnings.warn('%s: annotations skipped -- %s%s' % (key, why, extra))
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
        self.hp_marks = pd.DataFrame(hp_rows, columns=[
            'key', 'section_index', 'side', 'ml_um', 'width_um', 'ap_hp_um',
            'ap_hp_sd_um', 'ap_series_um'])
        self.ac_marks = pd.DataFrame(ac_rows, columns=[
            'key', 'section_index', 'side', 'ml_um', 'dv_um'])
        self._calibrate_hp()

        self.sections['ap_ac_um'] = self.sections['ap_um']
        self.ap_shift_um = 0.0
        self.shear = None
        anchor = str(self.series.get('ap_anchor', 'ac')).lower()
        if anchor == 'hp':
            self._anchor_on_hp()
        elif anchor == 'shear':
            self._anchor_shear()
        elif anchor not in ('ac', 'none'):
            warnings.warn("ap_anchor %r not understood -- using 'ac'" % anchor)

        if len(self.rois) and self.rois['ap_um'].isna().all():
            warnings.warn('AP is NaN everywhere: set section_thickness_um, the '
                          'AC section and ac_ap_um (or use ap_ref="ac")')

    # -------------------------------------------------------- AP anchor ---- #
    def _anchor_on_hp(self):
        """Shift the whole AP series so it matches the Hp-width landmark."""
        marked = self.sections.dropna(subset=['ap_hp_um', 'ap_um'])
        if len(marked) == 0:
            warnings.warn("ap_anchor='hp' but no section has a DM/DL mark -- "
                          'AP left on the anterior commissure')
            return
        weights = 1.0 / np.maximum(marked['ap_hp_sd_um'].to_numpy(float), 50.0) ** 2
        delta = marked['ap_hp_um'].to_numpy(float) - marked['ap_um'].to_numpy(float)
        shift = float(np.sum(weights * delta) / np.sum(weights))
        self.ap_shift_um = shift
        self.sections['ap_um'] += shift
        for tbl in (self.rois, self.scars):
            if len(tbl):
                tbl['ap_um'] += shift
        for b in self.rois['boundary'] if len(self.rois) else []:
            b[:, 1] += shift
        warnings.warn('AP shifted by %+.0f um to match the Hp-width landmark on '
                      '%d section(s)' % (shift, len(marked)))

    # ------------------------------------------------------- shear anchor -- #
    # ------------------------------------------------- hippocampus size -- #
    def _calibrate_hp(self):
        """
        Choose the hippocampal width curve for this bird, then turn every
        DM/DL mark into an AP estimate with it.

        series['hp_L_um'] is 'auto' to fit the bird's own plateau width from
        its own marks, None to use the atlas value, or a number to force one.
        The fit needs marks reaching the steep part of the curve; when they do
        not, it falls back to the atlas and says so.
        """
        want = self.series.get('hp_L_um', 'auto')
        step = abs(float(self.series.get('section_thickness_um') or 100.0)
                   * float(self.series.get('section_interval') or 1))
        self.hp_fit_info = dict(ok=False, why='atlas value used', L_um=HP_WIDTH_FIT['L'])
        if isinstance(want, str) and want.lower() == 'auto':
            w = self.sections.dropna(subset=['hp_width_um', 'section_index'])
            self.hp_fit_info = fit_hp_size(w['section_index'], w['hp_width_um'], step)
            if self.hp_fit_info['ok']:
                warnings.warn(
                    "hippocampus size fitted to this bird: L = %.0f +- %.0f um "
                    "(atlas %.0f); the section step it implies is %.0f um, set as "
                    "%.0f -- %s"
                    % (self.hp_fit_info['L_um'], self.hp_fit_info['L_se_um'],
                       HP_WIDTH_FIT['L'], self.hp_fit_info['step_um'], step,
                       self.hp_fit_info['why']))
            elif self.hp_fit_info['why']:
                warnings.warn('using the atlas hippocampus size (L = %.0f um): %s'
                              % (HP_WIDTH_FIT['L'], self.hp_fit_info['why']))
        elif want is not None:
            self.hp_fit_info = dict(ok=True, L_um=float(want), L_se_um=np.nan,
                                    scale=np.nan, scale_se=np.nan, step_um=np.nan,
                                    rms_um=np.nan, n=0, why='set by hand')
        self.hp_L_um = float(self.hp_fit_info['L_um'])
        self.hp_fit = (bird_hp_fit(self.hp_L_um)
                       if self.hp_fit_info['ok'] else dict(HP_WIDTH_FIT))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for df, col in ((self.sections, 'hp_width_um'),
                            (self.hp_marks, 'width_um')):
                if not len(df):
                    continue
                w = df[col].to_numpy(float)
                df['ap_hp_um'] = hp_width_to_ap(w, self.hp_fit)
                df['ap_hp_sd_um'] = hp_ap_sd(w, self.hp_fit)
        if len(self.sections):
            bad = (self.sections['hp_width_um'].notna()
                   & self.sections['ap_hp_um'].isna())
            if bad.any():
                warnings.warn('%d section(s) have a hippocampal width at or above '
                              'the fitted plateau (%.0f um), so no AP can be read '
                              'from them: %s' % (int(bad.sum()), self.hp_L_um,
                                                 list(self.sections.loc[bad, 'key'])))

    def ac_depth_um(self, side=None):
        """
        Depth of the anterior commissure below the brain surface (um), from
        the AC marks -- the lever arm ac_dv_um.  Falls back to the series
        value, then to NaN.  `side` is +1 (implanted) / -1 / None (both).
        """
        m = self.ac_marks
        if len(m):
            if side is not None and (m['side'] == side).any():
                m = m[m['side'] == side]
            d = m['dv_um'].to_numpy(float)
            d = d[np.isfinite(d)]
            if len(d):
                return float(np.mean(d))
        v = self.series.get('ac_dv_um')
        return float(v) if v else np.nan

    def _hp_shift(self, side=None):
        """
        Weighted mean of (AP from the hippocampal width) - (AP from the AC
        series), over the DM/DL marks of one hemisphere (side = +1 implanted,
        -1 other, None = both).  Returns (shift_um, se_um, n_marks).
        """
        m = self.hp_marks.dropna(subset=['ap_hp_um', 'ap_series_um'])
        if side is not None:
            m = m[m['side'] == side]
        if not len(m):
            return np.nan, np.nan, 0
        w = 1.0 / np.maximum(m['ap_hp_sd_um'].to_numpy(float), 50.0) ** 2
        delta = m['ap_hp_um'].to_numpy(float) - m['ap_series_um'].to_numpy(float)
        return (float(np.sum(w * delta) / np.sum(w)),
                float(np.sqrt(1.0 / np.sum(w))), int(len(m)))

    def _ac_offset(self, side):
        """
        AP (in the series) of the section on which the AC appears in this
        hemisphere, minus the AC's own AP.  Zero unless a separate AC section
        was marked for the other hemisphere.
        """
        key = self.series.get('ac_section') if side == 1 else \
            (self.series.get('ac_section_other') or self.series.get('ac_section'))
        row = self.sections[self.sections['key'] == key]
        ap0 = self.series.get('ac_ap_um')
        if not len(row) or ap0 is None or not np.isfinite(row['ap_ac_um'].iloc[0]):
            return 0.0
        return float(row['ap_ac_um'].iloc[0]) - float(ap0)

    def _anchor_shear(self):
        """
        Anchor AP on BOTH landmarks at once, per hemisphere.

        The AC sits deep and already defines the series; the DM/DL marks sit
        at the brain surface.  If the section plane is tilted away from the
        atlas plane the two disagree by an offset that grows with depth, so
        instead of choosing one, interpolate linearly between them:

            ap = ap_series + shift * (1 - dv / ac_dv) - ac_off * (dv / ac_dv)

        At the surface (dv = 0) this lands on the hippocampal-width landmark;
        at the depth of the AC it lands on the AC.  shift and ac_off are taken
        per hemisphere, so each side is anchored on its own marks.

        tan(pitch) = (shift + ac_off) / ac_dv expresses the AC-vs-hippocampus
        offset as an angle, on the same sign convention as ap_calibration
        (positive = the surface landmark sits anterior of where the AC series
        puts the section).

        Read it as a section-plane tilt only if you believe the offset is
        geometric.  A misidentified AC section, or a wrong ac_ap_um, produces
        exactly the same constant offset, and the two landmarks alone cannot
        tell them apart -- but they call for opposite treatments: a shift of
        the whole series (ap_anchor='hp') rather than a shear.  A useful check
        is whether shearing makes the probe angles of several birds agree
        better or worse; if worse, the offset is probably not a tilt.

        The difference between the
        the hemispheres' shifts is an apparent yaw about the DV axis -- but
        see HP_HEMISPHERE_FIT: at the size it usually takes, that difference
        is as easily genuine asymmetry as geometry, which is why it is applied
        as a per-hemisphere anchor and only REPORTED as an angle.
        """
        ac_dv = self.ac_depth_um()
        if not np.isfinite(ac_dv) or ac_dv <= 0:
            warnings.warn("ap_anchor='shear' needs the depth of the AC below the "
                          'brain surface: mark the commissure with A on the AC '
                          'reference section, or set ac_dv_um -- falling back to '
                          'the surface landmark alone')
            return self._anchor_on_hp()

        rows = []
        self._shear = {}
        both = self._hp_shift(None)
        for side in (1, -1):
            shift, se, n = self._hp_shift(side)
            fallback = not n
            if fallback:                  # nothing marked on this side
                shift, se = both[0], both[1]
            if not np.isfinite(shift):
                shift, se = 0.0, np.nan
            ac_off = self._ac_offset(side)
            dv_side = self.ac_depth_um(side)
            if not np.isfinite(dv_side) or dv_side <= 0:
                dv_side = ac_dv
            self._shear[side] = (shift, ac_off, dv_side)
            rows.append(OrderedDict([
                ('side', 'implanted' if side == 1 else 'other'),
                ('n_marks', n), ('from_other_side', fallback),
                ('hp_shift_um', shift), ('hp_shift_se_um', se),
                ('ac_offset_um', ac_off),
                ('ac_dv_um', dv_side),
                ('pitch_deg', float(np.rad2deg(np.arctan2(shift + ac_off, dv_side)))),
            ]))
        if not np.isfinite(both[0]):
            warnings.warn("ap_anchor='shear' but no section has a DM/DL mark -- "
                          'AP left on the anterior commissure')
            self._shear = {}
            return

        self.shear = pd.DataFrame(rows)
        d_shift = self._shear[1][0] - self._shear[-1][0]
        w_ref = float(self.hp_marks['width_um'].median()) if len(self.hp_marks) else np.nan
        self.shear.attrs.update(
            hemisphere_offset_um=d_shift,
            implied_yaw_deg=float(np.rad2deg(np.arctan2(d_shift, 2 * w_ref)))
            if np.isfinite(w_ref) and w_ref > 0 else np.nan,
            note='pitch_deg: section plane vs the atlas plane, about the ML axis. '
                 'hemisphere_offset_um / implied_yaw_deg are reported, not assumed '
                 'geometric -- see HP_HEMISPHERE_FIT.')
        self.ap_shift_um = 0.5 * (self._shear[1][0] + self._shear[-1][0])

        # apply, per point, to everything that carries an ML and a DV
        if len(self.scars):
            self.scars['ap_um'] += self.ap_correction(
                self.scars['ml_um'].to_numpy(float),
                self.scars['dv_um'].to_numpy(float))
        for i, r in self.rois.iterrows() if len(self.rois) else []:
            b = r['boundary']
            b[:, 1] += self.ap_correction(b[:, 0], b[:, 2])
            self.rois.at[i, 'ap_um'] = r['ap_um'] + float(self.ap_correction(
                r['center_ml_um'], r['center_dv_um'])[0])
        # sections['ap_um'] is left in the section frame on purpose: under
        # shear a section no longer HAS one AP, it spans a range with depth.
        # The corrected values are the per-point ones in rois and scars.
        warnings.warn('AP anchored on both landmarks: hp shift %+.0f / %+.0f um '
                      '(implanted / other), pitch %+.1f / %+.1f deg'
                      % (self._shear[1][0], self._shear[-1][0],
                         self.shear['pitch_deg'].iloc[0],
                         self.shear['pitch_deg'].iloc[1]))

    def ap_correction(self, ml_um, dv_um):
        """
        AP (um) to ADD to a point's section AP to put it in the atlas frame.
        Zero unless ap_anchor='shear'.  ml_um only picks the hemisphere.
        """
        ml = np.atleast_1d(np.asarray(ml_um, float))
        dv = np.atleast_1d(np.asarray(dv_um, float))
        out = np.zeros(np.broadcast(ml, dv).shape)
        for side, (shift, ac_off, ac_dv) in getattr(self, '_shear', {}).items():
            m = (ml >= 0) if side == 1 else (ml < 0)
            frac = np.where(np.isfinite(dv), dv / ac_dv, 0.0)
            out = np.where(m, shift * (1 - frac) - ac_off * frac, out)
        return out

    def ap_calibration(self, ac_dv_um=None):
        """
        Compare the AP of each DM/DL-marked section with the AP the anterior
        commissure gives it.

        The Hp landmark sits at the brain surface and the AC lies deep, so a
        section plane tilted away from the atlas plane shows up as a constant
        offset between the two: offset ~ tan(tilt) * (depth of the AC).  Pass
        ac_dv_um (or set the series parameter) to turn the offset into an angle.
        A trend in the offset along the series instead means the AP step per
        section is off -- the effective thickness, not the angle.

        Returns a DataFrame of the marked sections; the summary is in .attrs
        (offset_um, offset_sd_um, slope_um_per_section, tilt_deg).
        """
        if pd is None:
            raise ImportError('ap_calibration needs pandas')
        cols = ['key', 'section_index', 'hp_width_um', 'ap_hp_um', 'ap_hp_sd_um',
                'ap_ac_um']
        out = self.sections.dropna(subset=['ap_hp_um', 'ap_ac_um'])[cols].copy()
        if len(out) == 0:
            warnings.warn('no section has both a DM/DL mark and an AC-based AP')
            return out
        out['offset_um'] = out['ap_hp_um'] - out['ap_ac_um']
        w = 1.0 / np.maximum(out['ap_hp_sd_um'].to_numpy(float), 50.0) ** 2
        off = out['offset_um'].to_numpy(float)
        mean_off = float(np.sum(w * off) / np.sum(w))
        slope = np.nan
        if len(out) > 1 and out['section_index'].nunique() > 1:
            slope = float(np.polyfit(out['section_index'].to_numpy(float), off, 1,
                                     w=np.sqrt(w))[0])
        ac_dv = ac_dv_um if ac_dv_um is not None else self.series.get('ac_dv_um')
        tilt = (float(np.rad2deg(np.arctan2(mean_off, float(ac_dv))))
                if ac_dv else np.nan)
        out.attrs.update(offset_um=mean_off,
                         offset_sd_um=float(np.std(off, ddof=1)) if len(out) > 1 else np.nan,
                         slope_um_per_section=slope, tilt_deg=tilt,
                         note='offset = AP from the Hp width - AP from the AC; '
                              'positive means the surface landmark sits anterior '
                              'of where the AC series puts the section')
        return out

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
            n_sections        distinct sections traced (pieces of one broken
                              section count once -- they share an AP)
            n_entries         distinct entries traced, pieces counted separately
            dv_span_um, rms_um
            worst_section     entry whose trace fits worst -- an unticked
                              'flipped' box, a mis-ordered section or a piece
                              with a badly placed midline shows up here
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
            # distinct SECTIONS, not distinct entries: two pieces of one broken
            # section sit at the same AP and carry no AP information between them
            n_sec = g['section_index'].nunique()
            n_traces = g['key'].nunique()
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
                ('n_entries', n_traces),
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
        """
        AP half-extent of each reviewed section: half the gap to each
        neighbour.  Keyed by entry key, but worked out per SECTION -- the
        pieces of a broken section sit at the same AP and share one slab,
        rather than looking like neighbours zero microns apart.
        """
        rev = self.sections[self.sections['lhy'].notna()]
        if not len(rev):
            return {}
        per = rev.groupby('section_index')[ap_col].first().sort_values()
        ap = per.to_numpy(float)
        step = abs(float(self.series['section_thickness_um'] or 0)
                   * float(self.series['section_interval'])
                   * float(self.series['ap_scale']))
        lo = np.empty(len(ap))
        hi = np.empty(len(ap))
        if len(ap):
            gaps = np.diff(ap)
            lo[0], hi[-1] = step / 2, step / 2
            lo[1:], hi[:-1] = gaps / 2, gaps / 2
        by_index = dict(zip(per.index, zip(ap, lo, hi)))
        return {k: by_index[i] for k, i in zip(rev['key'], rev['section_index'])
                if i in by_index}

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
        shear = bool(getattr(self, '_shear', None))
        if shear and ap_col == 'ap_um':
            ap_col = 'ap_ac_um'       # slabs live in the uncorrected section frame
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

        # the slabs are section APs; with ap_anchor='shear' the caller's AP is
        # in the atlas frame, so take the correction back off the query points
        ap_q = pos[:, 1].copy()
        if shear:
            ap_q = ap_q - self.ap_correction(pos[:, 0], pos[:, 2])

        for _, r in rois.iterrows():
            geom = self.geoms[r['key']]
            e = self.ann['sections'][r['key']]['ellipses'][r['ellipse']]
            ap_c, lo, hi = slabs[r['key']]
            gap = np.maximum(0.0, np.maximum((ap_c - lo) - ap_q,
                                             ap_q - (ap_c + hi)))
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
