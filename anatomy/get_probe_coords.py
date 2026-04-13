import numpy as np
import csv
import pandas as pd

import os 
import sys
sys.path.append("..//utils/")
sys.path.append("..//neural/")
import format_waveform_data, waveform_analysis
import matplotlib.pyplot as plt

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

def ml_to_ap_dist(ml_A, ml_B, shank_dist=150):
    '''
    Given:
    - the ML position of each shank relative to the DM/DL boundary
    - the known distance between shanks (150um for H10 probes)

    Calculate the AP distance between shanks.

    Assumes that the DM/DL boundary runs at 45 deg relative to midline.
    '''
    # get the two angles
    alpha = np.deg2rad(135)
    delta = np.deg2rad(45)

    # get the terms for the distance equation
    Acos = ml_A * np.cos(delta)
    Asin = ml_A * np.sin(delta)
    Bcos = ml_B * np.cos(alpha)
    Bsin = ml_B * np.sin(alpha)

    # get the distance along the DM/DL boundary line
    dist_dmdl = Bcos + Acos + np.sqrt(shank_dist**2 - (Bsin - Asin)**2)

    # this is the hypotenuse of a 45/45/90 triangle with the AP distance
    ap_dist = dist_dmdl / np.sqrt(2)

    return ap_dist

def convert_coords(raw_coords):
    '''
    Takes the raw insertion coordinates and converts them to a more useful format

    Params
    ------
    raw_coords : nparray, shape (2, 3)
        For each shank (A, B):
        - ML distance of the probe from the DM/DL boundary
        - Width of the hippocampus (indicator of AP position along hippocampal axis)
        - Angle of probe insertion (in degrees)
    
    Converts relative values in mm to absolute (lamda-oriented) values in microns
    Takes AP estimate from histology and factors in known shank distance
    --> 150um for H10 probes, at an angle relative to midline
    Converts degrees to radians
    '''
    rel_ml = raw_coords[:, 0]
    rel_ap = raw_coords[:, 1]
    angle_deg = raw_coords[:, 2]

    # convert to microns
    rel_ml_um = rel_ml*1000
    rel_ap_um = rel_ap*1000
    
    # calculate ML distance from midline
    abs_ml = (rel_ap_um - rel_ml_um)

    # calculate the AP shank distance given the ML positions
    ap_shanks = ml_to_ap_dist(rel_ml_um[0], rel_ml_um[1])

    # adjust ap and convert to microns
    abs_ap_approx = dmdl_rel2abs(rel_ap_um)
    abs_ap = np.asarray([np.mean(abs_ap_approx)-ap_shanks/2, np.mean(abs_ap_approx)+ap_shanks/2])
    
    # convert degrees to radians
    angle_rad = np.deg2rad(angle_deg)
    
    return np.column_stack([abs_ml, abs_ap, angle_rad])


def probe_to_brain(insert_coords, probe_depth, probe_coords):
    '''
    Given a probe that is tilted towards the midline and not tilted in the AP axis,
    convert from probe coordinates (as output by kilosort) to brain coordinates.
    
    Params
    ------
    insert_coords : nparray, shape (2, 3)
        For each shank (A, B):
        ML and AP coordinates of the shank entry point into the brain
        the third value is the angle of probe insertion (in radians)
    probe_depth : float
        depth that probe was inserted
    probe_coords : nparray, shape (n_channels, 2)
        AP and DV coordinates along the probe for each channel
        
    Returns
    -------
    brain_coords : nparray, shape (n_channels, 3)
        ML, AP, and DV coordinates in the brain
    '''
    # format inputs
    insert_angle = insert_coords[0, 2]
    A_in_ml, A_in_ap = insert_coords[0, :2]
    B_in_ml, B_in_ap = insert_coords[1, :2]
    probe_ap = probe_coords[:, 0]
    probe_dv = probe_coords[:, 1]

    # get probe params
    n_channels = probe_coords.shape[0]
    shank_dist = probe_coords[n_channels//2, 0] - probe_coords[0, 0]
    probe_tip_ap = probe_coords[0, 0]

    # estimate the tip location for each shank
    A_tip_ml = A_in_ml - probe_depth*np.sin(insert_angle)
    A_tip_ap = A_in_ap
    B_tip_ml = B_in_ml - probe_depth*np.sin(insert_angle)
    B_tip_ap = B_in_ap
    tip_dv = probe_depth*np.cos(insert_angle)
    
    # account for shank
    A_idx = probe_ap < 100
    B_idx = probe_ap >= 100
    brain_ml = np.zeros_like(probe_ap)
    brain_ap = np.zeros_like(probe_ap)

    # convert the channel locations
    brain_ml[A_idx] = A_tip_ml + probe_dv[A_idx] * np.sin(insert_angle)
    brain_ml[B_idx] = B_tip_ml + probe_dv[B_idx] * np.sin(insert_angle)
    brain_ap[A_idx] = A_tip_ap + probe_ap[A_idx] - probe_tip_ap
    brain_ap[B_idx] = B_tip_ap + probe_ap[B_idx] - shank_dist - probe_tip_ap
    brain_dv = tip_dv - probe_dv * np.cos(insert_angle)
    
    return np.column_stack((brain_ml, brain_ap, brain_dv))

def get_anatomy_info(session_info_file, data_dict):
    # get the bird list
    bird_list = []
    for bird in data_dict.keys():
        bird_list.append(bird)

    # load the session info for each bird
    for bird in bird_list:
        session_list = data_dict[bird]['all_sessions']
        session_info = pd.read_excel(session_info_file, sheet_name=bird, header=1)
        session_info["id"] = session_info["date"].dt.strftime("%y%m%d")

        # get the approx probe depth per session
        for session_id in session_info['id']:
            if session_id in data_dict[bird].keys():
                # only calculate for new data
                if 'depth' in data_dict[bird][session_id].keys():
                    continue
                probe_depth = session_info.loc[session_info["id"] == session_id,
                                               "approx. depth (um)"].iloc[0]
                data_dict[bird][session_id]['depth'] = probe_depth

        # to store the insertion coordinates - ML, AP, angle for each shank
        data_dict[bird]['insert_coords'] = np.zeros((2, 3))

    # grab the raw coordinates
    probe_info = pd.read_excel(session_info_file, sheet_name='Anatomy', header=0)
    for i, row in probe_info.iterrows():
        # get the bird and shank IDs
        bird_shank = row['bird ID']
        bird, shank = bird_shank.split(sep='_')
        
        # extract the raw coords
        rel_ml = row['ML']
        rel_ap = row['AP']
        angle_deg = row['angle_deg']
        
        # store the raw values
        insert_coords = np.asarray([rel_ml, rel_ap, angle_deg])
        if shank=='A':
            data_dict[bird]['insert_coords'][0] = insert_coords
        else:
            data_dict[bird]['insert_coords'][1] = insert_coords

    # convert the relative coords to absolute
    for bird in bird_list:
        raw_coords = data_dict[bird]['insert_coords']
        abs_coords = convert_coords(raw_coords)
        data_dict[bird]['insert_coords'] = abs_coords 

    return data_dict

def get_cell_pos(session_dir, ks_dir, ephys_dir, insert_coords, depth):
    # load and format the waveform struct
    waveform_struct = format_waveform_data.load_wf_data(session_dir, ks_dir=ks_dir)
    wf_ids = waveform_struct['goodIDs']
    mean_waveforms, wf_channels, _, ch_names = format_waveform_data.sort_wf_by_channel('', waveform_struct,
                                                                                               data_dir=ephys_dir,
                                                                                               return_ch_names=True)       
    n_cells = mean_waveforms.shape[0]
    wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])

    # load the channel positions and convert to brain coords
    ch_pos_probe = np.load(f"{session_dir}{ks_dir}channel_positions.npy")
    ch_pos_brain = probe_to_brain(insert_coords=insert_coords,
                                    probe_depth=depth,
                                    probe_coords=ch_pos_probe)

    # get the position of each cell in the brain
    cell_pos = np.zeros((n_cells, 3))
    for cell, ch in enumerate(wf_ch_idx):
        cell_pos[cell] = ch_pos_brain[ch]
        
    return cell_pos

def get_channel_pos(session_dir, ks_dir, ephys_dir, insert_coords, depth):
    # load and format the waveform struct
    waveform_struct = format_waveform_data.load_wf_data(session_dir, ks_dir=ks_dir)
    wf_ids = waveform_struct['goodIDs']
    mean_waveforms, wf_channels, _, ch_names = format_waveform_data.sort_wf_by_channel('', waveform_struct,
                                                                                               data_dir=ephys_dir,
                                                                                               return_ch_names=True)       
    wf_ch_idx = np.asarray([ch_names.index(ch) for ch in wf_channels])

    # load the channel positions and convert to brain coords
    ch_pos_probe = np.load(f"{session_dir}{ks_dir}channel_positions.npy")
    ch_pos_brain = probe_to_brain(insert_coords=insert_coords,
                                    probe_depth=depth,
                                    probe_coords=ch_pos_probe)
        
    return ch_pos_brain


def define_dm_dl(ap_lims, n_pts=20):
    '''
    Gets rough coordinates for the DM/DL boundary
    Returns n_pts points along this line for plotting purposes
    '''
    # use AP lims to determine the ML coordinates
    min_ml, max_ml = dmdl_abs2rel(ap_lims)
    min_ap, max_ap = ap_lims

    # get the points
    bound_ml = np.linspace(min_ml, max_ml, n_pts)
    bound_ap = np.linspace(min_ap, max_ap, n_pts)

    return np.row_stack([bound_ml, bound_ap])


def save_cell_positions(data_dict, root_dir):
    '''
    Adds the channel locations in brain space and the locations of
    the best channel for all good cells to the data dict
    '''
    for bird in data_dict.keys():
        if bird == 'RBY94':
            continue
        print(f'\nlocalizing cells for {bird}')
        insert_coords = data_dict[bird]['insert_coords']
        session_list = data_dict[bird]['all_sessions']
        for session_id in session_list:
            # specify the file paths
            if 'ephys' in data_dict[bird][session_id]['preprocessed_data']:
                session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'
                for folder in sorted(os.listdir(session_dir)):
                    if f'{bird}_{session_id}' in folder:
                        ephys_id = folder[-13:]
                        for file in sorted(os.listdir(f"{session_dir}{bird}_{ephys_id}")):
                            if 'kilosort4' in file:
                                ks_dir = f"{bird}_{ephys_id}/{file}/"
                                ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"
                                depth = data_dict[bird][session_id]['depth']

                                # get the channel positions
                                ch_pos = get_channel_pos(session_dir, ks_dir, ephys_dir, insert_coords, depth)
                                data_dict[bird][session_id]['channel_pos'] = ch_pos

                                # get the cell positions
                                cell_pos = get_cell_pos(session_dir, ks_dir, ephys_dir, insert_coords, depth)
                                data_dict[bird][session_id]['cell_pos'] = cell_pos

    return data_dict