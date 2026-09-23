'''
3D summary plots across birds from lhy_roi_gui annotations.

    from lhy_roi_tools import load_lhy_rois
    import lhy_3d_plots

    rois_by_bird = {bird: load_lhy_rois(f'{root_dir}{bird}/histology/{bird}_lhy_rois.json')
                    for bird in ['LMN86', 'LMN91', 'LMN93']}
    fig, ax, avg = lhy_3d_plots.plot_tracks_and_lhy(rois_by_bird)

Conventions follow get_probe_coords_lhy: [ML, AP, DV] in um, ML signed with
+ = implanted hemisphere (the side the probe entered, so a track angled across
the midline ends at negative ML), AP relative to lambda (+ anterior), DV + ventral.

HEMISPHERES (`hemisphere` in plot_tracks_and_lhy)
-------------------------------------------------
    'signed'   both LHy sides averaged separately and drawn at their signed ML;
               tracks drawn as they are, so midline-crossing tracks can be seen
               reaching the contralateral LHy
    'mirror'   all LHy ellipses folded onto +ML into one average, and every
               track whose tip is at negative ML reflected so its tip is on the
               + side too -- for comparing targeting across birds directly
    'implant'  only LHy ellipses on the implanted side; tracks as they are

HOW THE LHY ELLIPSOID IS BUILT
------------------------------
Per bird, the ellipses on sections marked LHy are filled with a grid of points
(mapped into brain coordinates with that section's own midline and surface),
and each section is given the AP thickness of its slab (half the distance to
its reviewed neighbours on each side).  That volume has a centroid and a
covariance; the ellipsoid with the same centroid and covariance is a solid
ellipsoid with semi-axes sqrt(5 * eigenvalues).  So a bird whose annotations
really are an ellipsoid gets that ellipsoid back exactly, and anything else
gets its best second-moment match.

Across birds (`combine`):
    'mean_shape'  average the per-bird centroids and covariances -- the typical
                  LHy, not inflated by between-bird differences      [default]
    'pooled'      treat all birds' LHy volumes as one cloud (each bird weighted
                  equally) -- also includes between-bird scatter in position
'''
import warnings

import numpy as np
import matplotlib.pyplot as plt

import lhy_roi_tools as lrt

# --------------------------------------------------------------------------- #
#  FIGURE PARAMS                                                                 #
# --------------------------------------------------------------------------- #
LHY_IMPLANT_COLOR = 'xkcd:dark gray'
LHY_CONTRA_COLOR = 'xkcd:dusty rose'
LHY_MIRROR_COLOR = 'xkcd:scarlet'

# --------------------------------------------------------------------------- #
#  ELLIPSOIDS                                                                 #
# --------------------------------------------------------------------------- #

def lhy_moments(rois, side='implant', grid_um=25.0, ap_ref='lambda'):
    '''
    Volume centroid and covariance of one bird's LHy annotations on one side.

    rois : LHyROIs (from load_lhy_rois)
    side : 'implant'  ellipses on the implanted side (ML > 0)
           'contra'   ellipses on the contralateral side (ML < 0)
           'mirror'   all ellipses, contralateral ones reflected to ML > 0
    grid_um : spacing of the fill grid within each section

    Returns dict(center (3,), cov (3, 3), volume_um3, n_sections) or None if
    the bird has no LHy ellipses on that side.
    '''
    if side not in ('implant', 'contra', 'mirror'):
        raise ValueError("side must be 'implant', 'contra' or 'mirror'")
    ap_col = {'lambda': 'ap_um', 'ac': 'ap_from_ac_um'}[ap_ref]
    if len(rois.rois) == 0:
        return None
    ell_rows = rois.rois[rois.rois['lhy'] == 'yes']
    if side == 'implant':
        ell_rows = ell_rows[ell_rows['center_ml_um'] > 0]
    elif side == 'contra':
        ell_rows = ell_rows[ell_rows['center_ml_um'] < 0]
    if len(ell_rows) == 0:
        return None
    if ell_rows[ap_col].isna().any():
        raise ValueError(f'{rois.bird}: AP is NaN -- set the thickness, AC section and '
                         'ac_ap_um (or use ap_ref="ac" for every bird)')
    if rois.implant_sign is None and side != 'mirror':
        warnings.warn(f'{rois.bird}: implanted hemisphere unknown, ML sign is image-based')

    slabs = rois._slabs(ap_col)
    w_sum = 0.0
    m1 = np.zeros(3)
    m2 = np.zeros((3, 3))
    for _, r in ell_rows.iterrows():
        geom = rois.geoms[r['key']]
        e = rois.ann['sections'][r['key']]['ellipses'][r['ellipse']]

        # fill the ellipse with a grid in pixel space, keep points inside
        bnd = lrt.ellipse_boundary_px(e, 90)
        step = grid_um / geom.px
        xs = np.arange(bnd[:, 0].min(), bnd[:, 0].max() + step[0], step[0])
        ys = np.arange(bnd[:, 1].min(), bnd[:, 1].max() + step[1], step[1])
        gx, gy = np.meshgrid(xs, ys)
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        xy = xy[lrt.inside_ellipse_px(xy, e)]
        if len(xy) == 0:
            xy = np.asarray(e['center_px'], float)[None]
        ml, dv = geom.px_to_brain(xy)
        ok = np.isfinite(ml) & np.isfinite(dv)
        if not np.all(ok):
            warnings.warn(f"{rois.bird} {r['key']}: DV undefined for part of the "
                          'ellipse (surface too short?) -- those points dropped')
        ml, dv = ml[ok], dv[ok]
        if side == 'mirror':
            ml = np.abs(ml)

        # this section's slab along AP: offset centre and uniform spread
        ap_c, lo, hi = slabs[r['key']]
        thickness = lo + hi
        ap_mid = ap_c + (hi - lo) / 2.0
        area = len(ml) * grid_um ** 2
        w = area * thickness                       # volume of this slab

        pts = np.column_stack([ml, np.full(ml.shape, ap_mid), dv])
        mean = pts.mean(0)
        second = (pts.T @ pts) / len(pts)
        second[1, 1] += thickness ** 2 / 12.0      # uniform spread within the slab
        w_sum += w
        m1 += w * mean
        m2 += w * second

    if w_sum == 0:
        return None
    center = m1 / w_sum
    cov = m2 / w_sum - np.outer(center, center)
    return dict(center=center, cov=cov, volume_um3=w_sum,
                n_sections=ell_rows['key'].nunique())


def ellipsoid_from_cov(center, cov):
    '''Solid ellipsoid with this centroid/covariance: semi-axes and rotation.'''
    evals, evecs = np.linalg.eigh(cov)
    axes = np.sqrt(5.0 * np.clip(evals, 0, None))
    return dict(center=np.asarray(center, float), cov=cov, axes=axes, rotation=evecs)


def average_ellipsoid(moments, combine='mean_shape'):
    '''Combine per-bird lhy_moments into one ellipsoid.'''
    moments = [m for m in moments if m is not None]
    if not moments:
        raise ValueError('no LHy annotations to average')
    centers = np.array([m['center'] for m in moments])
    covs = np.array([m['cov'] for m in moments])
    center = centers.mean(0)
    cov = covs.mean(0)
    if combine == 'pooled':
        # mixture of equally weighted birds: within + between covariance
        dev = centers - center
        cov = cov + (dev.T @ dev) / len(moments)
    elif combine != 'mean_shape':
        raise ValueError("combine must be 'mean_shape' or 'pooled'")
    out = ellipsoid_from_cov(center, cov)
    out['n_birds'] = len(moments)
    out['center_sd'] = centers.std(0, ddof=1) if len(moments) > 1 else np.full(3, np.nan)
    return out


def ellipsoid_surface(ell, n=40):
    '''X, Y, Z grids (ML, AP, DV) for ax.plot_surface.'''
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    sphere = np.stack([np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v))], axis=-1)
    pts = (sphere * ell['axes']) @ ell['rotation'].T + ell['center']
    return pts[..., 0], pts[..., 1], pts[..., 2]


# --------------------------------------------------------------------------- #
#  TRACKS                                                                      #
# --------------------------------------------------------------------------- #

def collect_tracks(rois_by_bird, ap_ref='lambda', shared_direction=False,
                   mirror=False):
    '''
    One row per bird/shank with insertion and tip, from each bird's scar fits.

    mirror : reflect tracks whose tip is at negative ML (crossed the midline)
             so their tips land on the + side, matching a mirrored LHy.

    Returns (labels, birds, shank_locations, mirrored) where shank_locations
    is a list of (2, 3) [ML, AP, DV] arrays (entry, tip) -- the format
    plot_probe_by_pos in waveform_plots.py takes -- and mirrored flags the
    tracks that were reflected.
    '''
    labels, birds, locs, mirrored = [], [], [], []
    for bird, rois in rois_by_bird.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tracks = rois.scar_tracks(ap_ref=ap_ref, shared_direction=shared_direction)
        if len(tracks) == 0:
            print(f'  {bird}: no scar traces')
            continue
        for _, t in tracks.iterrows():
            loc = np.array([[t['insert_ml_um'], t['insert_ap_um'], 0.0],
                            [t['tip_ml_um'], t['tip_ap_um'], t['tip_dv_um']]])
            flip = bool(mirror and loc[1, 0] < 0)
            if flip:
                loc[:, 0] *= -1
            labels.append(f"{bird}_{t['shank']}")
            birds.append(bird)
            locs.append(loc)
            mirrored.append(flip)
    return labels, birds, locs, mirrored


# --------------------------------------------------------------------------- #
#  PLOT                                                                        #
# --------------------------------------------------------------------------- #

def _clip_track(loc, dv_range):
    """Cut an (entry, tip) segment to dv_range = (dv_min, dv_max)."""
    if dv_range is None:
        return loc
    a, b = loc
    if b[2] == a[2]:
        return loc
    t = np.clip((np.asarray(dv_range, float) - a[2]) / (b[2] - a[2]), 0, 1)
    return np.array([a + t[0] * (b - a), a + t[1] * (b - a)])


def _style_3d_axes(ax, points, pad_um=200.0, dv_range=None, true_aspect=True):
    """
    Same look as waveform_plots, but with tight limits around `points` and
    true proportions (1 um is the same length on every axis) via the box
    aspect, instead of ax.axis('equal') padding ML/AP out to the DV range.
    """
    ax.set_xlabel('ML (um)')
    ax.set_ylabel('AP (um)')
    ax.set_zlabel('DV (um)')
    pts = np.asarray(points, float)
    lo = np.nanmin(pts, 0) - pad_um
    hi = np.nanmax(pts, 0) + pad_um
    if dv_range is not None:
        lo[2], hi[2] = dv_range
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(hi[2], lo[2])                      # DV increases downward
    ax.set_box_aspect(hi - lo if true_aspect else (1, 1, 1.3), zoom=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('xkcd:grey')
        axis.pane.set_color('xkcd:light grey')


def plot_tracks_and_lhy(rois_by_bird, colors=None, combine='mean_shape',
                        hemisphere='signed', show_bird_lhy=False,
                        show_bird_centers=True, label_tracks=True,
                        lhy_colors=None, lhy_alpha=0.25,
                        ap_ref='lambda', shared_direction=False, grid_um=25.0,
                        dv_range=None, true_aspect=True, azim=-50, elev=20,
                        show=True):
    '''
    Probe tracks (entry -> tip) for every bird/shank, with the average LHy as a
    semi-transparent ellipsoid (one per hemisphere with hemisphere='signed').

    rois_by_bird : dict bird -> LHyROIs
    colors : dict bird -> color (default: tab10 by bird)
    hemisphere : 'signed', 'mirror' or 'implant' -- see the module docstring
    lhy_colors : dict side -> color, sides 'implant' / 'contra' / 'mirror'
    show_bird_lhy : also draw each bird's own ellipsoid(s) as a faint wireframe
    show_bird_centers : mark each bird's LHy centroid(s)
    dv_range : (dv_min, dv_max) um to show, e.g. (3000, 7000) to crop the
        dorsal part of the tracks; None = everything
    true_aspect : same um-per-inch on every axis (a full-depth view is tall and
        narrow); False stretches ML/AP to a roughly cubic box

    Returns fig, ax, avg where avg maps side -> averaged ellipsoid (center,
    axes, rotation, cov, n_birds, center_sd): {'implant', 'contra'} for
    'signed', {'mirror'} for 'mirror', {'implant'} for 'implant'.
    '''
    sides = {'signed': ['implant', 'contra'], 'mirror': ['mirror'],
             'implant': ['implant']}.get(hemisphere)
    if sides is None:
        raise ValueError("hemisphere must be 'signed', 'mirror' or 'implant'")
    birds = list(rois_by_bird.keys())
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = {b: cmap(i % 10) for i, b in enumerate(birds)}
    side_colors = {'implant': LHY_IMPLANT_COLOR, 
                    'contra': LHY_CONTRA_COLOR,
                    'mirror': LHY_MIRROR_COLOR}
    side_colors.update(lhy_colors or {})
    side_names = {'implant': 'implanted side', 'contra': 'contralateral',
                  'mirror': 'mirrored'}

    # per-bird moments and averages, per side
    moments = {side: {} for side in sides}
    avg = {}
    for side in sides:
        for bird, rois in rois_by_bird.items():
            moments[side][bird] = lhy_moments(rois, side=side, grid_um=grid_um,
                                              ap_ref=ap_ref)
        have = [b for b, m in moments[side].items() if m is not None]
        missing = [b for b in birds if b not in have]
        if missing:
            print(f'  no {side_names[side]} LHy ellipses for: {", ".join(missing)}')
        if have:
            avg[side] = average_ellipsoid(list(moments[side].values()), combine=combine)
    if not avg:
        raise ValueError('no LHy annotations to average')

    labels, track_birds, locs, mirrored = collect_tracks(
        rois_by_bird, ap_ref=ap_ref, shared_direction=shared_direction,
        mirror=(hemisphere == 'mirror'))

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')
    extent = [np.zeros(0)] * 3

    # average LHy, one ellipsoid per side
    for side, ell in avg.items():
        X, Y, Z = ellipsoid_surface(ell)
        ax.plot_wireframe(X, Y, Z, color=side_colors[side], lw=0.3, alpha=lhy_alpha)
        # ax.plot_surface(X, Y, Z, color=side_colors[side], alpha=lhy_alpha, lw=0,
        #                 shade=True, zorder=0)
        ax.plot([], [], color=side_colors[side], lw=0.3, alpha=0.5,
                label=f'LHy {side_names[side]} ({combine}, n={ell["n_birds"]})')
        extent = [np.concatenate([e, g.ravel()]) for e, g in zip(extent, (X, Y, Z))]

    # per-bird LHy
    for side in sides:
        for bird, m in moments[side].items():
            if m is None:
                continue
            if show_bird_lhy:
                Xb, Yb, Zb = ellipsoid_surface(ellipsoid_from_cov(m['center'], m['cov']), n=24)
                ax.plot_wireframe(Xb, Yb, Zb, color=colors[bird], lw=0.3, alpha=0.3)
            if show_bird_centers:
                ax.scatter(*m['center'], color=colors[bird], marker='o', s=25,
                           edgecolor='k', lw=0.5, depthshade=False)

    # probe tracks (dashed if reflected)
    labelled = set()
    for label, bird, loc, flip in zip(labels, track_birds, locs, mirrored):
        seg = _clip_track(loc, dv_range)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=colors[bird], lw=2,
                ls='--' if flip else '-',
                label=None if bird in labelled else bird)
        # ax.scatter(*loc[1], color=colors[bird], marker='v', s=20, depthshade=False)
        labelled.add(bird)
        if label_tracks:
            ax.text(*seg[0], label + (' (mirrored)' if flip else ''),
                    size='x-small', weight='semibold', ha='left', va='bottom')
        extent = [np.concatenate([e, seg[:, i]]) for i, e in enumerate(extent)]

    if hemisphere == 'signed' and len(extent[0]):
        ax.plot([0, 0], [np.min(extent[1]), np.max(extent[1])],
                [np.max(extent[2])] * 2, c='xkcd:grey', lw=1, ls=':')   # midline

    _style_3d_axes(ax, np.column_stack(extent), dv_range=dv_range,
                   true_aspect=true_aspect)
    ax.set_xlabel('|ML| (um)' if hemisphere == 'mirror' else 'ML (um, + = implanted)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), markerscale=2)
    ax.view_init(azim=azim, elev=elev)

    for side, ell in avg.items():
        c, a = ell['center'], ell['axes']
        print(f'LHy {side_names[side]}, {combine} over {ell["n_birds"]} bird(s): '
              f'center ML {c[0]:.0f}, AP {c[1]:.0f}, DV {c[2]:.0f} um; '
              f'semi-axes {np.round(np.sort(a)[::-1]).astype(int)} um')
        if ell['n_birds'] > 1:
            sd = ell['center_sd']
            print(f'  between-bird SD of the center: ML {sd[0]:.0f}, AP {sd[1]:.0f}, '
                  f'DV {sd[2]:.0f} um')
    if any(mirrored):
        print(f'  mirrored tracks: {", ".join(l for l, f in zip(labels, mirrored) if f)}')

    if show:
        plt.show()
    return fig, ax, avg
