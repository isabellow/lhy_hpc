import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

import os 
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "methods"))
sys.path.append(os.path.join(script_dir, "..", "neural"))
import format_waveform_data, waveform_analysis
from estimate_target_coords import rotate_AP, rotate_ML, convert_head_to_stereo, probe_dir_brain

'''
Functions for localizing the probe in the brain for LHY recordings
'''
'''
Utility functions for channel mapping
'''
def remap_by_channel_map(values_by_row, channel_map, n_channels_total, fill_value=np.nan):
    '''
    Account for channels excluded during sorting.

    channel_positions.npy does not include channels excluded in the KS4 GUI
    channel_map.npy gives the original (pre-exclusion) channel index
    '''
    values_by_row = np.asarray(values_by_row, dtype=float)
    out_shape = (n_channels_total,) + values_by_row.shape[1:]
    full = np.full(out_shape, fill_value, dtype=float)
    for row, ch_id in enumerate(channel_map):
        full[ch_id] = values_by_row[row]
    return full

def get_channel_shank(probe_coords, n_shanks):
    '''
    Assigns each channel to a shank index (0, 1, ..., n_shanks-1), 
    generalized to an arbitrary number of shanks.
    '''
    # get the AP positions of each channel column
    probe_ap = probe_coords[:, 0]
    unique_ap, ap_rank = np.unique(probe_ap, return_inverse=True)

    # check to make sure there are an equal number of columns per shank
    if len(unique_ap) % n_shanks != 0:
        raise ValueError(
            f"N channel columns not consistent across shanks--check channel_positions.npy"
        )

    # cluster by shank
    n_groups_per_shank = len(unique_ap) // n_shanks
    shank_idx = ap_rank // n_groups_per_shank
    
    return shank_idx.astype(int)


'''
Anatomical functions
'''
def lhy_rel2abs(rel_ap):
    '''
    Given the tip position relative to the center of the lateral 
    hypothalamus, get the AP position relative to lambda (in microns).

    Positive is anterior of center, negative is posterior.

    The lateral hypothalamus center is roughly 1 mm anterior of lambda.
    '''
    return 1000 + rel_ap

def lhy_abs2rel(abs_ap):
    '''
    Given AP position relative to lamda, get positioning relative
    to lateral hypothalamus center.
    '''
    return abs_ap - 1000

def dmdl_rel2abs(rel_ap):
    '''
    Given hippocampus width, get AP position relative to lamda (in microns)

    DM/DL boundary follows roughly a 45 degree angle relative to the midline
    and is 900 um lateral at 3800 anterior - TODO check this!
    (so 0 L at 4700 A)
    '''
    return 4700 - rel_ap

def dmdl_abs2rel(abs_ap):
    '''
    Given AP position relative to lamda, get hippocampus width
    '''
    return 4700 - abs_ap


'''
Geometric functions
'''
def ml_to_ap_dist(ml_A, ml_B, shank_dist=150):
    '''
    Refine the triangulation of a multishank probe
    using the known difference between shanks.

    Given:
    - the ML position of each shank relative to the midline
    - the known distance between shanks

    Calculate the expected AP distance between shanks.
    '''
    ml_diff = ml_A - ml_B
    ap_dist = np.sqrt(shank_dist**2 - ml_diff**2)

    return ap_dist


def estimate_depth_hist(insert_coords, tip_coords):
    '''
    Estimate the probe travel distance from insertion to tip.

    Useful to cross-check against the experimentally measured
    final depth.

    Params
    ------
    insert_coords : ndarray, shape (n_shanks, 2)
        absolute [ML, AP] of each shank's insertion (um)
    tip_coords : ndarray, shape (n_shanks, 3)
        absolute [ML, AP, DV] of each shank's tip (um)
        DV is the tip's depth below the brain surface as 
        estimated from histology.

    Returns
    -------
    est_depth : float
        histology-only estimate of the final insertion depth 
        averaged across shanks (in um)
        (nan where the tip estimate, incl. DV, is incomplete)
    '''
    ml_offset = insert_coords[:, 0] - tip_coords[:, 0]
    ap_offset = insert_coords[:, 1] - tip_coords[:, 1]
    tip_dv = tip_coords[:, 2]
    est_depth = np.sqrt(ml_offset**2 + ap_offset**2 + tip_dv**2)
    return np.nanmean(est_depth)

'''
Localization functions
'''
def refine_ap(ml_um, approx_ap_um, shank_idx=None, shank_dist=150):
    '''
    Given per-shank ML/AP histology estimates (already in microns),
    re-centers the AP estimates using the known, fixed AP distance
    between shanks.

    If shanks are missing histology, they are excluded with shank_idx.
    '''
    n_shanks = len(ml_um)
    if shank_idx is None:
        shank_idx = np.arange(n_shanks)

    if n_shanks == 1:
        abs_ap = approx_ap_um
    else:
        ap_steps = np.asarray([
            ml_to_ap_dist(ml_um[i], ml_um[i + 1],
                            shank_dist=shank_dist * (shank_idx[i + 1] - shank_idx[i]))
            for i in range(n_shanks - 1)
        ])
        # cumulative AP offset of each shank relative to shank 0
        cum_ap_offset = np.concatenate([[0], np.cumsum(ap_steps)])
        # re-centered on the mean of the per-shank histology AP estimates
        abs_ap = np.nanmean(approx_ap_um) + cum_ap_offset - np.mean(cum_ap_offset)

    return abs_ap

def convert_tip_coords(raw_tip_coords, insert_coords, final_depth, head_angle, shank_dist=150):
    '''
    Converts relative values in mm to absolute (lamda-oriented) values in microns
    Takes AP estimate from histology and factors in known shank distance

    If there is no tip histology, this function estimates the tip location
    using the head angle during surgery, final depth of insertion, and
    insertion coords.

    Params
    ------
    raw_tip_coords : nparray, shape (n_shanks, 3)
        histology [ML, AP, DV] of each shank's tip (mm).
        AP here is distance from the LHY center.
        DV is optional.
    insert_coords : ndarray, shape (n_shanks, 2)
        absolute [ML, AP] insertion point of each shank (um), as returned
        by convert_insert_coords
    final_depth : float
        experimentally measured final insertion depth (um)
    head_angle : sequence of 2 floats
        [roll_deg, pitch_deg] measured during implant
    shank_dist : float
        known, fixed distance between shanks (um)    

    Returns
    -------
    tip_coords : ndarray, shape (n_shanks, 3)
        absolute [ML, AP, DV] of each shank's tip (um)
    '''
    # data params
    n_shanks = raw_tip_coords.shape[0]
    has_hist = ~np.isnan(raw_tip_coords[:, 0]) & ~np.isnan(raw_tip_coords[:, 1])

    # preallocate variables
    abs_ml = np.full(n_shanks, np.nan)
    abs_ap = np.full(n_shanks, np.nan)
    abs_dv = np.full(n_shanks, np.nan)

    # for shanks with histology, get the tip coordinates
    if np.any(has_hist):
        hist_idx = np.where(has_hist)[0]
        abs_ml[hist_idx] = raw_tip_coords[hist_idx, 0] * 1000
        approx_ap_hist = lhy_rel2abs(raw_tip_coords[hist_idx, 1] * 1000)
        abs_ap[hist_idx] = refine_ap(abs_ml[hist_idx], approx_ap_hist,
                                      shank_idx=hist_idx, shank_dist=shank_dist)
        abs_dv[hist_idx] = raw_tip_coords[hist_idx, 2] * 1000

        # fill in DV with geometric estimate as needed
        dv_missing = hist_idx[np.isnan(abs_dv[hist_idx])]
        if len(dv_missing) > 0:
            ml_off = insert_coords[dv_missing, 0] - abs_ml[dv_missing]
            ap_off = insert_coords[dv_missing, 1] - abs_ap[dv_missing]
            abs_dv[dv_missing] = np.sqrt(np.clip(final_depth**2 - ml_off**2 - ap_off**2, 0, None))

    # for shanks without histology, estimate the tip coordinates
    missing = ~has_hist
    if np.any(missing):
        # get the estimated probe track
        roll_deg, pitch_deg = head_angle
        hist_rad = np.deg2rad(pitch_deg - 37)
        v_brain = probe_dir_brain(hist_rad, np.deg2rad(roll_deg))  # [AP, ML, DV]
        u_ml, u_ap, u_dv = v_brain[1], v_brain[0], v_brain[2]

        # caculate the tip location along this track
        abs_ml[missing] = insert_coords[missing, 0] - final_depth * u_ml
        abs_ap[missing] = insert_coords[missing, 1] - final_depth * u_ap
        abs_dv[missing] = final_depth * u_dv

    return np.column_stack([abs_ml, abs_ap, abs_dv])


def convert_insert_coords(raw_insert_coords, raw_intended_coords, shank_dist=150):
    '''
    Converts relative values in mm to absolute (lamda-oriented) values in microns
    Takes AP estimate from histology and factors in known shank distance

    Params
    ------
    raw_insert_coords : nparray, shape (n_shanks, 2)
        histology [ML, AP] of each shank's insertion point (mm).
        AP here is HPC width at the insertion point.
        Nan where histology is missing.
    raw_intended_coords : ndarray, shape (n_shanks, 2)
        intended [ML, AP] insertion coordinates from surgery notes
        (mm, relative to midline/lambda), used as a fallback
    shank_dist : float
        known, fixed distance between shanks (um)

    Returns
    -------
    insert_coords : ndarray, shape (n_shanks, 2)
        absolute [ML, AP] of each shank's insertion point (um)
    '''
    # check for histology
    has_hist = ~np.isnan(raw_insert_coords[:, 0]) & ~np.isnan(raw_insert_coords[:, 1])

    # ML coordinates
    abs_ml = np.where(has_hist,
                        raw_insert_coords[:, 0] * 1000,
                        raw_intended_coords[:, 0] * 1000)

    # AP defaults to intended if no histology
    abs_ap = raw_intended_coords[:, 1] * 1000
    if np.any(has_hist):
        hist_idx = np.where(has_hist)[0]
        approx_ap_hist = dmdl_rel2abs(raw_insert_coords[hist_idx, 1] * 1000)
        abs_ap[hist_idx] = refine_ap(abs_ml[hist_idx], approx_ap_hist,
                                      shank_idx=hist_idx, shank_dist=shank_dist)

    return np.column_stack([abs_ml, abs_ap])


def probe_to_brain(insert_coords, tip_coords, probe_depth, probe_coords):
    '''
    Given a probe that is tilted towards the midline and tilted in the AP axis,
    convert from probe coordinates (as output by kilosort) to brain coordinates.

    Params
    ------
    insert_coords : nparray, shape (n_shanks, 2)
        absolute [ML, AP] of each shank's insertion point (um)
    tip_coords : nparray, shape (n_shanks, 3)
        absolute [ML, AP, DV] of each shank's (final/deepest) tip (um)
    probe_depth : float
        depth the probe was inserted on this session
    probe_coords : nparray, shape (n_channels, 2)
        local [ap, dv] coordinates along the probe for each channel
        from channel_positions.npy (dv = 0 at the probe's physical tip)
        
    Returns
    -------
    brain_coords : nparray, shape (n_channels, 3)
        ML, AP, and DV coordinates in the brain for each channel
    '''
    # data params
    n_shanks = insert_coords.shape[0]
    n_channels = probe_coords.shape[0]

    # channel coords on the probe
    probe_ap_local = probe_coords[:, 0]
    probe_dv_local = probe_coords[:, 1]
    shank_idx = get_channel_shank(probe_coords, n_shanks)

    # allocate variables
    brain_ml = np.full(n_channels, np.nan)
    brain_ap = np.full(n_channels, np.nan)
    brain_dv = np.full(n_channels, np.nan)

    # for each shank, get the channel positions
    for s in range(n_shanks):
        ch_mask = (shank_idx == s)
        if not np.any(ch_mask):
            continue

        # shank coordinates and trajectory in the brain
        insert_ml, insert_ap = insert_coords[s]
        tip_ml, tip_ap, tip_dv = tip_coords[s]
        ml_offset = insert_ml - tip_ml
        ap_offset = insert_ap - tip_ap
        traj_vector = np.asarray([ml_offset, ap_offset, tip_dv])
        traj_len = np.sqrt(ml_offset**2 + ap_offset**2 + tip_dv**2)
        u_ml, u_ap, u_dv = traj_vector / traj_len

        # channel positions on this shank
        ch_dv = probe_dv_local[ch_mask]
        ch_ap = probe_ap_local[ch_mask]

        # get the distances from the shank insertion point
        dist_from_insert = probe_depth - ch_dv

        # get the ap locations relative to shank tip
        tip_dv_local = np.min(ch_dv)
        ref_ap_local = np.mean(ch_ap[ch_dv==tip_dv_local])
        fine_ap_offset = ch_ap - ref_ap_local

        # get the 3D brain positions
        brain_ml[ch_mask] = insert_ml - dist_from_insert * u_ml
        brain_ap[ch_mask] = insert_ap - dist_from_insert * u_ap + fine_ap_offset
        brain_dv[ch_mask] = dist_from_insert * u_dv
    
    return np.column_stack((brain_ml, brain_ap, brain_dv))


def get_channel_cell_pos(session_dir, ks_dir, ephys_dir, insert_coords, tip_coords, depth):
    '''
    Triangulates probe channel positions in the brain
    Matches each cell's best channel to its brain position

    Returns
    -------
    ch_pos_brain : nparray, shape (n_channels_total, 3)
        ML, AP, DV brain coords for every channel on the probe
        (NaN for any channel excluded from kilosort4)
    ch_shank_idx : nparray, shape (n_channels_total,), int
        shank index for each channel (-1 for excluded channels)
    cell_pos : nparray, shape (n_cells, 3)
        brain coords of each good cell's best channel
    cell_shank_idx : nparray, shape (n_cells,), int
        shank index of each good cell's best channel
    '''
    # load and format the waveform data
    waveform_struct = format_waveform_data.load_wf_data(session_dir, ks_dir=ks_dir)
    wf_ids = waveform_struct['goodIDs']
    mean_waveforms, wf_channels, _, ch_names = format_waveform_data.sort_wf_by_channel('', waveform_struct,
                                                                                               data_dir=ephys_dir,
                                                                                               return_ch_names=True)       
    wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])

    # params
    n_cells = mean_waveforms.shape[0]
    n_channels_total = len(ch_names)
    n_shanks = insert_coords.shape[0]

    # channel positions (not inc. excluded channels) and mapping to all probe channels
    ch_pos_probe = np.load(f"{session_dir}{ks_dir}channel_positions.npy")
    channel_map = np.load(f"{session_dir}{ks_dir}channel_map.npy").squeeze().astype(int)

    # get the positions in the brain
    ch_pos_brain = probe_to_brain(insert_coords=insert_coords,
                                    tip_coords=tip_coords,
                                    probe_depth=depth,
                                    probe_coords=ch_pos_probe)

    # which shank is each channel on?
    ch_shank_idx = get_channel_shank(ch_pos_probe, n_shanks)

    # account for excluded channels
    ch_pos_brain = remap_by_channel_map(ch_pos_brain, channel_map, n_channels_total)
    ch_shank_idx = remap_by_channel_map(ch_shank_idx, channel_map, n_channels_total, fill_value=-1).astype(int)
    excluded_idx = np.where(np.isnan(ch_pos_brain[:, 0]))[0]
    excluded_names = [ch_names[i] for i in excluded_idx]
    print(f"  broken/excluded channels ({len(excluded_names)}): {excluded_names}")

    # get the position of each cell's best channel
    cell_pos = np.zeros((n_cells, 3))
    cell_shank_idx = np.zeros(n_cells, dtype=int)
    for cell, ch in enumerate(wf_ch_idx):
        cell_pos[cell] = ch_pos_brain[ch]
        cell_shank_idx[cell] = ch_shank_idx[ch]

    return ch_pos_brain, ch_shank_idx, cell_pos, cell_shank_idx


def get_raw_anatomy_info(session_info_file, data_dict):
    '''
    Load the anatomy info for each probe shank for each bird

    Also gets the estimated depth of insertion for each session

    Anatomy sheet in good sessions should have (per shank):

    Measured from histology:
    - 'insert ML' : insertion point distance from midline
    - 'insert AP' : HPC width at insertion
    - 'tip ML' : tip distance from midline
    - 'tip AP' : tip distance from LHY center
    - 'tip DV' : tip distance from brain surface (optional)
    - 'angle ML' : histologically measured angle relative to midline

    Measured experimentally:
    - final depth : experimentally measured tip distance from surface for histology scar
    - roll deg : head tilt ML
    - pitch deg : beak bar angle
    - intended AP : insertion AP from lambda (during implant surgery)
    - intended ML : insertion ML from midline (during implant surgery)
    '''
    # get the bird list
    bird_list = list(data_dict.keys())    

    # load the session info for each bird
    for bird in bird_list:
        session_list = data_dict[bird]['all_sessions']
        session_info = pd.read_excel(session_info_file, sheet_name=bird, header=1)
        session_info["id"] = session_info["date"].dt.strftime("%y%m%d")

        # get the approx probe depth per session
        for session_id in session_info['id']:
            if session_id in data_dict[bird].keys():
                probe_depth = session_info.loc[session_info["id"] == session_id,
                                               "approx. depth (um)"].iloc[0]
                data_dict[bird][session_id]['depth'] = probe_depth

    # get N shanks
    probe_info = pd.read_excel(session_info_file, sheet_name='Anatomy', header=0)
    shank_letter_to_idx = {}
    n_shanks_per_bird = {}
    for i, row in probe_info.iterrows():
        bird_shank = row['bird ID']
        if '_' in bird_shank:
            bird, shank = bird_shank.split(sep='_')
        else:
            bird, shank = bird_shank, 'A'
        shank_idx = shank_letter_to_idx.setdefault(shank, len(shank_letter_to_idx))
        n_shanks_per_bird[bird] = shank_idx + 1

    # to store anatomy info
    for bird in bird_list:
        n_shanks = n_shanks_per_bird.get(bird, 1)
        data_dict[bird]['raw_insert_coords'] = np.full((n_shanks, 2), np.nan)
        data_dict[bird]['raw_tip_coords'] = np.full((n_shanks, 3), np.nan)
        data_dict[bird]['raw_intended_coords'] = np.full((n_shanks, 2), np.nan)
        data_dict[bird]['probe_angle_ml'] = np.full(n_shanks, np.nan)
        data_dict[bird]['head_angle'] = np.zeros(2)
        data_dict[bird]['final_depth'] = np.nan

        
    # extract the raw coords
    for i, row in probe_info.iterrows():
        bird_shank = row['bird ID']
        if '_' in bird_shank:
            bird, shank = bird_shank.split(sep='_')
        else:
            bird, shank = bird_shank, 'A'
        if bird not in data_dict:
            continue
        shank_idx = shank_letter_to_idx[shank]

        # ---- measured from histology ----
        # insertion coordinates
        ins_ml, ins_ap = row['insert ML'], row['insert AP']
        data_dict[bird]['raw_insert_coords'][shank_idx] = [ins_ml, ins_ap]

        # tip coordinates
        tip_ml, tip_ap, tip_dv = row['tip ML'], row['tip AP'], row['tip DV']
        data_dict[bird]['raw_tip_coords'][shank_idx] = [tip_ml, tip_ap, tip_dv]
        
        # measured angle relative to midline
        data_dict[bird]['probe_angle_ml'][shank_idx] = np.deg2rad(row['angle ML'])

        # ---- measured experimentally ----
        # probe depth for histology scar
        data_dict[bird]['final_depth'] = row['final depth']

        # head angle during probe implant
        ml_diff = row['ML diff']
        ml_offset = row['ML offset']
        roll_deg = np.rad2deg(np.arctan2(ml_diff, 2*ml_offset))
        pitch_deg = 90 - row['pitch deg']
        data_dict[bird]['head_angle'] = [roll_deg, pitch_deg]

        # intended insertion coordinates -- fallback for insert_coords
        int_ml, int_ap = row['intended ML'], row['intended AP']
        data_dict[bird]['raw_intended_coords'][shank_idx] = [int_ml, int_ap]

    return data_dict

def convert_anatomy_info(data_dict, tol_frac=0.1):
    '''
    Converts each bird's raw anatomy measurements into absolute [ML, AP]
    insert_coords and [ML, AP, DV] tip_coords (um) in 3D brain space.

    Fallback behavior, applied independently per shank:
    - insert_coords: histology 'insert ML'/'insert AP' where available,
      else the intended ML/AP targeted during implant surgery.
    - tip_coords: histology 'tip ML'/'tip AP'/'tip DV' where available,
      else estimated from that shank's (real or fallback) insert_coords,
      the measured head pitch/roll, and the measured final insertion depth.

    Also cross-checks the experimentally measured final depth against the
    depth implied by histology alone, where histology exists.
    '''
    # get the bird list
    bird_list = list(data_dict.keys())  

    # convert the relative coords to absolute
    for bird in bird_list:
        # get anatomy data
        raw_insert = data_dict[bird]['raw_insert_coords']
        raw_intended = data_dict[bird]['raw_intended_coords']
        raw_tip = data_dict[bird]['raw_tip_coords']
        final_depth = data_dict[bird]['final_depth']
        head_angle = data_dict[bird]['head_angle']

        # convert the insertion coords
        insert_coords = convert_insert_coords(raw_insert, raw_intended)
        data_dict[bird]['insert_coords'] = insert_coords

        # convert the tip coords
        tip_coords = convert_tip_coords(raw_tip, insert_coords, final_depth, head_angle)
        data_dict[bird]['tip_coords'] = tip_coords

        # compare the experimentally and histologically measured depths
        expt_depth = data_dict[bird]['final_depth']
        hist_depth = estimate_depth_hist(insert_coords, tip_coords)
        if np.isnan(hist_depth):
            print('histology missing - could not check depth')
        else:
            pct_diff = np.abs(hist_depth - expt_depth) / expt_depth
            if pct_diff > tol_frac:
                print(f"  {bird}: noted final depth = {expt_depth:.0f} um vs histology depth = {hist_depth:.0f} um"
                      f"-- double-check histology measurements and insertion notes")

    return data_dict


def save_cell_positions(data_dict, root_dir):
    '''
    Adds the channel locations in brain space and the locations of
    the best channel for all good cells to the data dict
    '''
    for bird in data_dict.keys():
        print(f'\nlocalizing cells for {bird}')
        insert_coords = data_dict[bird]['insert_coords']
        tip_coords = data_dict[bird]['tip_coords']
        session_list = data_dict[bird]['all_sessions']

        for session_id in session_list:
            session_data = data_dict[bird][session_id]
            if 'ephys' not in session_data['preprocessed_data']:
                continue

            # set paths
            session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
            ephys_id = session_data['ephys_id']
            ks_id = session_data['ks_folder']
            ks_dir = f"{bird}_{ephys_id}/{ks_id}/"
            ephys_dir = f"{session_dir}{bird}_{ephys_id}/"

            # channel and cell positions in brain coordinates
            depth = session_data['depth']
            ch_pos, ch_shank_idx, cell_pos, cell_shank_idx = get_channel_cell_pos(
                session_dir, ks_dir, ephys_dir, insert_coords, tip_coords, depth
            )
            # save everything
            session_data['channel_pos'] = ch_pos
            session_data['cell_pos'] = cell_pos
            session_data['shank_idx'] = cell_shank_idx

    return data_dict