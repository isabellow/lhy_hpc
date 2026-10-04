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
    data_dict[bird]['lhy_travel']   (n_shanks, 2) the same two points as
                                    distance along the track from the surface
    data_dict[bird]['lhy_segments'] per shank, (n_segments, 4):
                                    [dv_in, dv_out, travel_in, travel_out]

    data_dict[bird][session]['lhy_ch_pos']  (n_channels, 3) from the scar track
    data_dict[bird][session]['lhy_cell_pos'](n_cells, 3)
    data_dict[bird][session]['lhy_dist_ch'] (n_channels,) um, negative inside
    data_dict[bird][session]['in_lhy_ch']   (n_channels,) bool
    data_dict[bird][session]['lhy_dist']    (n_cells,) um
    data_dict[bird][session]['in_lhy']      (n_cells,) bool
'''
import os

import numpy as np

import get_probe_coords_lhy
from estimate_target_coords import hist_rad_from_pitch
from lhy_roi_tools import load_lhy_rois

ANNOTATION_FMT = '{root}{bird}/histology_annotations/{bird}_lhy_rois.json'

# how far outside the annotated LHy a channel can sit and still count: a probe
# picks up spikes from cells some distance away, so this is deliberately loose
TOL_UM = 150.0


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


def fill_anatomy_from_annotations(data_dict, bird_ids, root_dir, data_root=None,
                                  tol_frac=0.1):
    '''
    Per shank, where the good-sessions spreadsheet has no histology, take the
    insertion point and tip from the GUI annotations instead. Shanks that DO 
    have spreadsheet histology are left as they were--the annotation is only 
    reported next to them, as a cross-check.

    Call between get_probe_coords_lhy.convert_anatomy_info() and
    save_cell_positions().

    Writes data_dict[bird]['anatomy_source'], one label per shank
    ('spreadsheet' / 'annotation' / 'intended' / 'excluded'), so later steps
    can tell which frame a bird's positions are in.
    '''
    data_root = root_dir if data_root is None else data_root
    for bird in bird_ids:
        bd = data_dict[bird]
        shank_ids = np.ravel(bd['shank_ids'])
        exclude = np.ravel(np.asarray(bd['exclude_shank'], bool))
        raw_insert = np.atleast_2d(np.asarray(bd['raw_insert_coords'], float))
        raw_tip = np.atleast_2d(np.asarray(bd['raw_tip_coords'], float))
        insert = np.atleast_2d(np.asarray(bd['insert_coords'], float)).copy()
        tip = np.atleast_2d(np.asarray(bd['tip_coords'], float)).copy()

        # what the spreadsheet actually measured (same test convert_anatomy_info
        # uses: tip DV is optional)
        sheet_insert = np.isfinite(raw_insert[:, :2]).all(axis=1)
        sheet_tip = np.isfinite(raw_tip[:, :2]).all(axis=1)
        source = np.where(sheet_insert | sheet_tip, 'spreadsheet', 'intended')
        source[exclude] = 'excluded'

        need = ~(sheet_insert & sheet_tip) & ~exclude
        if not need.any() and not (sheet_tip & ~exclude).any():
            bd['anatomy_source'] = source
            continue

        _, gui_insert, gui_tip = load_probe_track(
            bird, data_root, shank_ids, exclude_shank=exclude)
        if gui_insert is None:
            if need.any():
                print(f'  {bird}: no histology annotations on file -- shanks '
                      f'{[str(x) for x in shank_ids[need]]} keep the '
                      f'intended-coords estimate')
            bd['anatomy_source'] = source
            continue

        final_depth = get_probe_coords_lhy.parse_per_shank(
            bd['final_depth'], len(shank_ids), name=f'{bird} final_depth')
        for s, sh in enumerate(shank_ids):
            if exclude[s]:
                continue
            gi, gt = gui_insert[s], gui_tip[s]
            if not (np.isfinite(gi).all() and np.isfinite(gt).all()):
                if need[s]:
                    print(f'  {bird} shank {sh}: no scar traced in the '
                          f'annotations -- keeping the intended-coords estimate')
                continue

            if need[s]:
                insert[s], tip[s] = gi, gt
                source[s] = 'annotation'
                print(f'  {bird} shank {sh}: no histology in the spreadsheet -- '
                      f'using the GUI annotation: insert ML/AP = '
                      f'{gi[0]:.0f}/{gi[1]:.0f} um, tip ML/AP/DV = '
                      f'{gt[0]:.0f}/{gt[1]:.0f}/{gt[2]:.0f} um')
                # the same depth check convert_anatomy_info does for the sheet:
                # drive depth and annotation are still independent measurements
                hist_depth = get_probe_coords_lhy.estimate_depth_hist(
                    insert[[s]], tip[[s]])
                if np.isfinite(hist_depth) and np.isfinite(final_depth[s]) \
                        and final_depth[s] != 0:
                    pct = abs(hist_depth - final_depth[s]) / abs(final_depth[s])
                    if pct > tol_frac:
                        print(f'      noted final depth = {final_depth[s]:.0f} um '
                              f'vs annotation depth = {hist_depth:.0f} um '
                              f'-- check the scar trace and the insertion notes')
            else:
                # spreadsheet wins; report the disagreement as a cross-check
                d_ins = np.linalg.norm(insert[s] - gi)
                d_tip = np.linalg.norm(tip[s] - gt)
                if max(d_ins, d_tip) > 250:
                    print(f'  {bird} shank {sh}: spreadsheet vs annotation differ '
                          f'by {d_ins:.0f} um at the insertion and {d_tip:.0f} um '
                          f'at the tip (keeping the spreadsheet)')

        bd['insert_coords'] = insert
        bd['tip_coords'] = tip
        bd['anatomy_source'] = source
    return data_dict


def lhy_track_bounds(rois, insert_coords, tip_coords, final_depth,
                     tol_um=TOL_UM, step_um=10.0, past_tip_um=1500.0,
                     hist_rad=0.0):
    '''
    Where each shank's track enters and leaves the LHy.

    This walks a dense set of virtual channels down each shank with
    probe_to_brain and tests each one with rois.distance -- the same
    positions and the same test used to flag real channels. The bounds and
    the per-channel flags therefore agree by construction: a channel is
    flagged in_lhy exactly when its DV falls inside one of the segments
    returned here, so the lines on a depth plot can be read as the in/out
    boundary without marking individual points.

    The walk continues past the traced tip so that sessions recorded deeper
    than the histology session are covered.

    Returns
    -------
    dvs      : (n_shanks, 2) [shallowest, deepest] DV the LHy reaches along
               each shank, in the same brain DV as lhy_ch_pos / lhy_cell_pos.
               NaN where the shank never enters
    travel   : (n_shanks, 2) the same two points as distance along the shank
               from the brain surface
    segments : list, one (n_segments, 4) array per shank:
               [dv_in, dv_out, travel_in, travel_out]. More than one row
               means the track leaves the LHy and re-enters, and only the
               rows -- not the overall dvs -- then describe where cells are
               in or out
    '''
    insert_coords = np.atleast_2d(np.asarray(insert_coords, float))
    tip_coords = np.atleast_2d(np.asarray(tip_coords, float))
    n_shanks = insert_coords.shape[0]
    dvs = np.full((n_shanks, 2), np.nan)
    travel = np.full((n_shanks, 2), np.nan)
    segments = [np.zeros((0, 4)) for _ in range(n_shanks)]

    traj_len = np.full(n_shanks, np.nan)
    for s in range(n_shanks):
        if np.isnan(insert_coords[s]).any() or np.isnan(tip_coords[s]).any():
            continue
        traj_len[s] = np.sqrt((insert_coords[s, 0] - tip_coords[s, 0]) ** 2
                              + (insert_coords[s, 1] - tip_coords[s, 1]) ** 2
                              + tip_coords[s, 2] ** 2)
    if not np.isfinite(traj_len).any():
        return dvs, travel, segments

    # a dense virtual probe: one column per shank, running from past the tip
    # (negative local dv) up to the brain surface. Driving it at the
    # histology depth makes local dv = 0 land on the traced tip.
    local_dv = np.arange(-float(past_tip_um),
                         np.nanmax(traj_len) + step_um, step_um)
    probe_coords = np.column_stack([
        np.repeat(np.arange(n_shanks) * 1000.0, len(local_dv)),
        np.tile(local_dv, n_shanks)])
    pos = get_probe_coords_lhy.probe_to_brain(
        insert_coords, tip_coords, final_depth, probe_coords,
        final_depth=final_depth, hist_rad=hist_rad)

    good = np.isfinite(pos).all(1)
    dist = np.full(len(pos), np.nan)
    if good.any():
        dist[good] = rois.distance(pos[good])['dist_um'].to_numpy()
    inside = np.isfinite(dist) & (dist <= tol_um)

    for s in range(n_shanks):
        sl = slice(s * len(local_dv), (s + 1) * len(local_dv))
        ins_s = inside[sl]
        if not ins_s.any():
            continue
        dv_s = pos[sl, 2]
        # distance along the shank from the surface
        trav_s = traj_len[s] - local_dv
        order = np.argsort(trav_s)          # surface -> deep
        ins_s, dv_s, trav_s = ins_s[order], dv_s[order], trav_s[order]

        edges = np.diff(np.concatenate(([0], ins_s.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1) - 1
        seg = np.column_stack([dv_s[starts], dv_s[stops],
                               trav_s[starts], trav_s[stops]])
        segments[s] = seg
        dvs[s] = [seg[:, :2].min(), seg[:, :2].max()]
        travel[s] = [seg[:, 2:].min(), seg[:, 2:].max()]
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

        final_depth = bird_data['final_depth']
        pitch_deg = np.ravel(bird_data.get('head_angle', [np.nan]))[-1]
        hist_rad = 0.0 if not np.isfinite(pitch_deg) else hist_rad_from_pitch(pitch_deg)

        # where the track crosses the LHy: a property of the bird, not the
        # session, so it can be drawn onto a depth axis like nucleus_dvs
        dvs, travel, segments = lhy_track_bounds(
            rois, insert, tip, final_depth, tol_um=tol_um, hist_rad=hist_rad)
        bird_data['lhy_dvs'] = dvs
        bird_data['lhy_travel'] = travel
        bird_data['lhy_segments'] = segments
        bird_data['lhy_tol_um'] = float(tol_um)
        for s, shank in enumerate(np.ravel(bird_data['shank_ids'])):
            if np.isnan(dvs[s]).all():
                print(f'  shank {shank}: the track never enters the LHy')
            else:
                print(f'  shank {shank}: LHy from DV {dvs[s, 0]:.0f} to '
                      f'{dvs[s, 1]:.0f} um')

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

            # channel and cell positions, but anchored to the ANNOTATED scar
            ch_pos, _, cell_pos, _ = get_probe_coords_lhy.get_channel_cell_pos(
                session_dir, ks_dir, ephys_dir, insert, tip,
                session_data['depth'], hist_rad=hist_rad,
                final_depth=final_depth)

            # cell-level arrays are masked by keep_cells, exactly as
            # save_cell_positions does, so 'in_lhy' lines up with cell_pos and
            # waveform_props. Channel-level arrays cover the whole probe.
            keep = get_probe_coords_lhy.get_keep_mask(data_dict, bird, session_id,
                                                      cell_pos.shape[0])
            if keep is None:
                print(f'  skipping {session_id}: keep_cells is stale '
                      f'(re-run flag_excluded_cells with overwrite=True)')
                continue
            cell_pos = cell_pos[keep]

            session_data['lhy_ch_pos'] = ch_pos
            session_data['lhy_cell_pos'] = cell_pos
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
