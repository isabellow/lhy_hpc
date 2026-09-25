'''
Flag which channels and cells sat in or near the lateral hypothalamus, using
the LHy boundaries and probe scars annotated in the GUI (lhy_roi_gui.py).

Channel positions are rebuilt from the ANNOTATED scar track rather than from
the insertion notes, so they and the LHy boundaries come from the same
histology and can be compared directly. Everything else reuses the existing
anatomy code: get_probe_coords_lhy.get_channel_cell_pos does the probe
geometry, exactly as step 3 does.

Step 8 of build_data_dict:

    data_dict[bird]['lhy_dvs']      (n_shanks, 2) [shallowest, deepest] DV of
                                    the LHy along each shank, same units as
                                    cell_pos[:, -1] -- use like 'nucleus_dvs'

    data_dict[bird][session]['lhy_ch_pos']  (n_channels, 3) from the scar track
    data_dict[bird][session]['lhy_cell_pos'](n_cells, 3)
    data_dict[bird][session]['lhy_dist_ch'] (n_channels,) um, negative inside
    data_dict[bird][session]['in_lhy_ch']   (n_channels,) bool
    data_dict[bird][session]['lhy_dist']    (n_cells,) um
    data_dict[bird][session]['in_lhy']      (n_cells,) bool

    Currently not adding to dict, but could if useful:
    data_dict[bird]['lhy_travel']   (n_shanks, 2) the same two points as
                                    distance along the track from the surface
    data_dict[bird]['lhy_segments'] per shank, (n_segments, 4):
                                    [dv_in, dv_out, travel_in, travel_out]
'''
import os
import numpy as np
import get_probe_coords_lhy
from estimate_target_coords import hist_rad_from_pitch
from lhy_roi_tools import load_lhy_rois

# set path to annotation data
ANNOTATION_FMT = '{root}{bird}/histology_annotations/{bird}_lhy_rois.json'

# how far outside the annotated LHy a channel can sit and still count: a probe
# picks up spikes from cells some distance away, so this is deliberately loose
TOL_UM = 50.0


def load_probe_track(bird, data_root, shank_ids, exclude_shank=None,
                     annotation_fmt=ANNOTATION_FMT, **roi_overrides):
    '''
    Load a bird's annotations and pull out the scar track.

    Returns (rois, insert_coords, tip_coords) in get_probe_coords_lhy's format,
    or (None, None, None) if the bird has no annotation file. Excluded shanks,
    and shanks with no scar traced, come back as NaN.
    '''
    path = annotation_fmt.format(root=data_root, bird=bird)
    if not os.path.isfile(path):
        return None, None, None
    rois = load_lhy_rois(path, **roi_overrides)
    insert, tip = rois.probe_inputs(shanks=[str(s) for s in np.ravel(shank_ids)])
    if exclude_shank is not None:
        drop = np.ravel(np.asarray(exclude_shank, bool))
        insert[drop[:len(insert)]] = np.nan
        tip[drop[:len(tip)]] = np.nan
    return rois, insert, tip


def lhy_track_bounds(rois, insert_coords, tip_coords, tol_um=TOL_UM,
                     step_um=10.0, past_tip_um=1500.0):
    '''
    Where each shank's track crosses into and out of the LHy.

    Walks the straight line from the brain surface down along each shank --
    past the traced tip, so sessions recorded deeper are covered -- and finds
    the stretches lying inside the LHy or within tol_um of it.

    Returns
    -------
    dvs      : (n_shanks, 2) [shallowest, deepest] DV of the LHy stretch, in
               the same brain DV units as channel_pos / cell_pos
    travel   : (n_shanks, 2) the same two points as distance along the shank
    segments : list, one (n_segments, 4) array per shank:
               [dv_in, dv_out, travel_in, travel_out]. More than one row means
               the track leaves the LHy and re-enters
    '''
    insert_coords = np.atleast_2d(np.asarray(insert_coords, float))
    tip_coords = np.atleast_2d(np.asarray(tip_coords, float))
    n_shanks = insert_coords.shape[0]
    dvs = np.full((n_shanks, 2), np.nan)
    travel = np.full((n_shanks, 2), np.nan)
    segments = [np.zeros((0, 4)) for _ in range(n_shanks)]

    for s in range(n_shanks):
        if np.isnan(insert_coords[s]).any() or np.isnan(tip_coords[s]).any():
            continue
        insert_ml, insert_ap = insert_coords[s]
        tip_ml, tip_ap, tip_dv = tip_coords[s]
        traj = np.array([insert_ml - tip_ml, insert_ap - tip_ap, tip_dv])
        traj_len = float(np.sqrt((traj ** 2).sum()))
        if traj_len <= 0:
            continue
        u = traj / traj_len

        d = np.arange(0.0, traj_len + past_tip_um + step_um, step_um)
        pts = np.column_stack([insert_ml - d * u[0], insert_ap - d * u[1],
                               d * u[2]])
        dist = rois.distance(pts)['dist_um'].to_numpy()
        inside = np.isfinite(dist) & (dist <= tol_um)
        if not inside.any():
            continue

        edges = np.diff(np.concatenate(([0], inside.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1) - 1
        seg = np.column_stack([pts[starts, 2], pts[stops, 2], d[starts], d[stops]])
        segments[s] = seg
        dvs[s] = [seg[:, 0].min(), seg[:, 1].max()]
        travel[s] = [seg[:, 2].min(), seg[:, 3].max()]
    return dvs, travel, segments


def collect_lhy_positions(data_dict, bird_ids, root_dir, data_root=None,
                          tol_um=TOL_UM, overwrite=False, **roi_overrides):
    '''
    Step 8: add the LHy boundaries and the in-LHy flags.

    -> needs: 'all_sessions', 'preprocessed_data' (step 1), 'keep_cells' (1b),
              'ephys_id', 'ks_folder', 'depth' (step 3)
    -> cell-level outputs are masked by 'keep_cells', like cell_pos
    -> needs: {bird}_lhy_rois.json from the annotation GUI

    Birds with no annotation file are skipped.
    '''
    data_root = root_dir if data_root is None else data_root

    for bird in bird_ids:
        bird_data = data_dict[bird]
        rois, insert, tip = load_probe_track(
            bird, data_root, bird_data['shank_ids'],
            exclude_shank=bird_data.get('exclude_shank'), **roi_overrides)
        if rois is None:
            print(f'\n{bird}: no LHy annotations on file - skipped')
            continue
        print(f'\nlocating the LHy for {bird}')

        # where the track crosses the LHy: a property of the bird, not the
        # session, so it can be drawn onto a depth axis like nucleus_dvs
        dvs, travel, segments = lhy_track_bounds(rois, insert, tip, tol_um=tol_um)
        bird_data['lhy_dvs'] = dvs
        # bird_data['lhy_travel'] = travel
        # bird_data['lhy_segments'] = segments
        bird_data['lhy_tol_um'] = float(tol_um)
        for s, shank in enumerate(np.ravel(bird_data['shank_ids'])):
            if np.isnan(dvs[s]).all():
                print(f'  shank {shank}: the track never enters the LHy')
            else:
                print(f'  shank {shank}: LHy from DV {dvs[s, 0]:.0f} to '
                      f'{dvs[s, 1]:.0f} um')

        final_depth = bird_data['final_depth']
        pitch_deg = np.ravel(bird_data.get('head_angle', [np.nan]))[-1]
        hist_rad = 0.0 if not np.isfinite(pitch_deg) \
            else hist_rad_from_pitch(pitch_deg)

        for session_id in bird_data['all_sessions']:
            session_data = data_dict[bird][session_id]
            if (not overwrite) and ('in_lhy_ch' in session_data):
                continue
            if 'ephys' not in session_data['preprocessed_data']:
                continue

            # same paths as step 3
            session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
            ephys_id = session_data['ephys_id']
            ks_dir = f"{bird}_{ephys_id}/{session_data['ks_folder']}/"
            ephys_dir = f'{session_dir}{bird}_{ephys_id}/'

            # channel and cell positions, but anchored to the annotated scar
            ch_pos, _, cell_pos, _ = get_probe_coords_lhy.get_channel_cell_pos(
                session_dir, ks_dir, ephys_dir, insert, tip,
                session_data['depth'], hist_rad=hist_rad,
                final_depth=final_depth)

            # cell-level arrays are masked by keep_cells,
            keep = get_probe_coords_lhy.get_keep_mask(data_dict, bird, session_id,
                                                      cell_pos.shape[0])
            if keep is None:
                print(f'  skipping {session_id}: keep_cells is stale '
                      f'(re-run flag_excluded_cells with overwrite=True)')
                continue
            cell_pos = cell_pos[keep]

            # in LHy?
            for pos, dist_key, flag_key in ((ch_pos, 'lhy_dist_ch', 'in_lhy_ch'),
                                            (cell_pos, 'lhy_dist', 'in_lhy')):
                dist = np.full(len(pos), np.nan)
                good = np.isfinite(pos).all(1)
                if good.any():
                    dist[good] = rois.distance(pos[good])['dist_um'].to_numpy()
                session_data[dist_key] = dist
                session_data[flag_key] = np.isfinite(dist) & (dist <= tol_um)

            print(f"  {session_id}: {session_data['in_lhy_ch'].sum()} channels "
                  f"and {session_data['in_lhy'].sum()} cells in or within "
                  f'{tol_um:.0f} um of the LHy')
    return data_dict
