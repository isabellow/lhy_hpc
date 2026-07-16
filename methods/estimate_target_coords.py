'''
We have entry coordinates and ventral distance that successfully target a brain region at
a known pitch and zero roll.

Estimate the new roll and rotate to a different pitch.

Given these changes, estimate the new entry coordinates and ventral distance to hit the same target.
'''
import numpy as np

def estimate_roll(dv_left, dv_right, ml_offset):
    '''
    Given DV offset between 2 points equidistant from the midline
    estimate the roll in radians.
    '''
    ml_diff = np.round(dv_left-dv_right, 2)
    return np.arctan2(ml_diff, 2*ml_offset)


''' Functions to rotate along each axis '''
def rotate_AP(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.asarray([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    return rot_mat

def rotate_ML(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.asarray([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    return rot_mat

def rotate_DV(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.asarray([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return rot_mat

''' Functions to convert between brain and stereotaxic coordinates '''
def convert_head_to_stereo(pitch, roll):
    '''
    Given the pitch and roll of the head, map vectors from
    head reference frame to stereotaxic reference frame.
    
    Roll rotates about the AP axis (right/left side are not level)
    Pitch rotates about the ML axis (front/back are relatively offset)
    
    R is the rotation matric mapping vectors from head-frame to stereo-frame
    '''
    R = rotate_AP(roll) @ rotate_ML(pitch)
    return R

def probe_dir_brain(pitch, roll, v_stereo=np.asarray([0, 0, 1])):
    '''
    Given the pitch and roll of the head, map vectors from
    stereotaxic reference frame to head reference frame.
    
    Default is vertically mounted probe.
    '''
    R = convert_head_to_stereo(pitch, roll)
    
    # convert stereo to head vector
    v_brain = R.T @ v_stereo
    v_brain = v_brain / np.linalg.norm(v_brain)
    
    # make sure DV is positive
    if v_brain[2] < 0:
        v_brain = -v_brain
        
    return v_brain

def get_target_loc(AP_entry, ML_entry, DV_probe, pitch_deg, roll_rad):
    '''
    Calculate target location [AP, ML, DV] in 3D brain coordinates.
    Uses empirically-determined entry coordinates and depth.

    AP_entry, ML_entry : float
        in mm, AP and ML coordinates of entry point
    DV_probe : float
        in mm, depth of probe insertion (+ is ventral)
    pitch_deg : int
        beak bar angle
    roll_rad : float
        computed from ML difference using estimate_roll(...)
    '''
    # for clarity, convert beak bar angle to histology level
    hist_deg = pitch_deg - 37
    
    # convert head angle to radians
    hist_rad = np.deg2rad(hist_deg)
    
    # get entry point and vector direction to target
    E = np.asarray([AP_entry, ML_entry, 0.0])
    v_brain = probe_dir_brain(hist_rad, roll_rad)
    
    # get target location in the brain
    T = E + (DV_probe*v_brain)
    
    return T

def get_new_coords(target_loc, pitch_deg, roll_rad):
    '''
    Given a target location in the brain [AP, ML, DV]
    and a measured ML roll and AP pitch, calculate the 
    entry point and depth for (vertical) probe insertion.
    '''
    # convert beak bar angle to histology level
    hist_deg = pitch_deg - 37
    
    # convert head angle to radians
    hist_rad = np.deg2rad(hist_deg)
    
    # ensure target is properly formatted
    T = np.asarray(target_loc, dtype=float)
    
    # new vector direction from target to surface
    v_brain = probe_dir_brain(hist_rad, roll_rad)
    
    # calculate the vertical distance from the target to the surface
    surface_dist = T[2] / v_brain[2]
    
    # get the new entry vector
    E = T - (surface_dist*v_brain)
    
    return {
        "entry_AP": np.round(E[0], 2),
        "entry_ML": np.round(E[1], 2),
        "travel_DV": np.round(surface_dist, 2)
    }