'''
Helpers for targeting brain structures at new head angles.

We have entry coordinates and a ventral travel distance that successfully target a
brain region at a known head pitch and roll. Given a new pitch and roll, estimate the
new entry coordinates and travel distance to hit the same target -- or, given a desired
entry point, solve for the pitch and probe angle that still hit it.

CONVENTIONS (all angles in degrees unless the name says _rad)
------------------------------------------------------------
Brain frame  : [AP, ML, DV].  AP + = anterior, ML + = lateral, DV + = ventral (depth).
Stereo frame : [AP, ML, DV].  AP + = anterior, ML + = lateral, DV + = down.
               These two frames coincide when the head is at `histology_level`.

pitch_deg    : beak bar angle, degrees below horizontal.
               0 = beak level, positive = beak tip rotated down.
histology_level : the beak bar angle at which the brain frame coincides with the
               stereotaxic frame, i.e. the pitch equivalent to the plane the
               Applegate/histology brains were sliced at. A vertical probe inserted
               with the head at this pitch travels straight down the brain's DV axis.

hist_deg = histology_level - pitch_deg
               How far the head is rotated NOSE-UP relative to the atlas orientation.
               Positive (head pitched less than the atlas angle) makes a vertical probe
               travel posteriorly in the brain; negative makes it travel anteriorly.

roll_rad     : positive = the +ML side of the head sits lower (more ventral).
probe_angle  : positive = probe tip deflected toward -ML, i.e. toward the midline for a
               target in the +ML hemisphere.

NOTE: entry coordinates are solved in the BRAIN frame, but the manipulator dials move
along the STEREOTAXIC axes. Functions that return an entry point therefore also return
the equivalent dial offsets from the reference landmark ("dial_*"). The two differ
whenever hist_deg != 0.

NOTE: the brain surface is modeled as the flat plane DV = 0 through the reference
landmark. Real surface curvature will offset the DV zero at entry points far from the
landmark; the returned travel distance is a starting estimate, not a final depth.
'''
import numpy as np

# beak bar angle (deg below horizontal) equivalent to the histology / atlas slice plane
histology_level = 50


def estimate_roll(dv_left, dv_right, ml_offset):
    '''
    Given the DV offset between 2 points equidistant from the midline,
    estimate the roll in radians.
    '''
    ml_diff = np.round(dv_left - dv_right, 2)
    return np.arctan2(ml_diff, 2 * ml_offset)


def hist_rad_from_pitch(pitch_deg):
    '''
    Convert a beak bar angle (degrees below horizontal) into the head's rotation
    relative to the atlas orientation, in radians.
    '''
    return np.deg2rad(histology_level - float(pitch_deg))


def pitch_from_hist_rad(hist_rad):
    '''Inverse of hist_rad_from_pitch: returns the beak bar angle in degrees.'''
    return histology_level - np.rad2deg(hist_rad)


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
def convert_head_to_stereo(hist_rad, roll_rad):
    '''
    Given the head's rotation relative to the atlas orientation (hist_rad) and its
    roll, return the rotation matrix R mapping vectors from the head/brain frame to
    the stereotaxic frame.

    Pitch is intrinsic (about the head's own ML axis, i.e. the ear bar axis);
    roll is applied in the stereotaxic frame, hence R = rotate_AP @ rotate_ML.
    '''
    R = rotate_AP(roll_rad) @ rotate_ML(hist_rad)
    return R


def probe_dir_brain(hist_rad, roll_rad, v_stereo=np.asarray([0., 0., 1.])):
    '''
    Map the probe's travel direction from the stereotaxic frame into the brain frame.
    Default is a vertically mounted probe.
    '''
    R = convert_head_to_stereo(hist_rad, roll_rad)

    # convert stereo vector to brain vector
    v_brain = R.T @ v_stereo
    v_brain = v_brain / np.linalg.norm(v_brain)

    # a descending probe must have a ventral component; anything else means the head
    # orientation or the probe angle is out of range, so fail loudly rather than
    # silently mirroring the direction
    if v_brain[2] <= 0:
        raise ValueError(
            f"probe direction is not ventral in the brain (v_brain = {np.round(v_brain, 3)}); "
            "check pitch, roll and probe angle"
        )

    return v_brain


def entry_to_dials(E_brain, hist_rad, roll_rad):
    '''
    Convert an entry point expressed in brain coordinates into offsets along the
    stereotaxic axes (i.e. what the manipulator dials should read), measured from the
    reference landmark with the head at its current pitch and roll.

    Returns [AP, ML, DV] in the stereo frame. The DV term is how far the entry point
    sits below (positive) or above (negative) the landmark; it is informative only,
    since DV is re-zeroed on the brain surface at the entry point.
    '''
    R = convert_head_to_stereo(hist_rad, roll_rad)
    return R @ np.asarray(E_brain, dtype=float)


def get_target_loc(AP_entry, ML_entry, DV_probe, pitch_deg, roll_rad):
    '''
    Calculate target location [AP, ML, DV] in 3D brain coordinates.
    Uses empirically-determined entry coordinates and depth.

    AP_entry, ML_entry : float
        in mm, AP and ML coordinates of the entry point (brain frame)
    DV_probe : float
        in mm, distance travelled by the probe (+ is ventral)
    pitch_deg : float
        beak bar angle, degrees below horizontal
    roll_rad : float
        computed from the ML difference using estimate_roll(...)
    '''
    hist_rad = hist_rad_from_pitch(pitch_deg)

    # get entry point and vector direction to target
    E = np.asarray([float(AP_entry), float(ML_entry), 0.0])
    v_brain = probe_dir_brain(hist_rad, roll_rad)

    # get target location in the brain
    T = E + (float(DV_probe) * v_brain)

    return T


def get_target_loc_angled(AP_entry, ML_entry, DV_probe, pitch_deg, roll_rad, probe_angle_rad):
    '''
    Same as get_target_loc, but for a probe angled toward the midline rather than
    mounted vertically. Useful for double-checking the output of get_new_angles
    below (forward pass).

    DV_probe is the distance travelled ALONG the probe axis.
    '''
    hist_rad = hist_rad_from_pitch(pitch_deg)

    # calculate v_stereo given angled probe
    v_stereo = rotate_AP(probe_angle_rad) @ np.asarray([0., 0., 1.])

    # get entry point and vector direction to target
    E = np.asarray([float(AP_entry), float(ML_entry), 0.0])
    v_brain = probe_dir_brain(hist_rad, roll_rad, v_stereo=v_stereo)

    # get target location in the brain
    T = E + (float(DV_probe) * v_brain)

    return T


def get_new_coords(target_loc, pitch_deg, roll_rad):
    '''
    Given a target location in the brain [AP, ML, DV] and a measured ML roll and beak
    bar pitch, calculate the entry point and travel distance for a vertically mounted
    probe.

    entry_AP / entry_ML are in the brain frame; dial_AP / dial_ML are the equivalent
    offsets along the stereotaxic axes from the reference landmark, which is what you
    actually set on the manipulator.
    '''
    hist_rad = hist_rad_from_pitch(pitch_deg)

    # ensure target is properly formatted
    T = np.asarray(target_loc, dtype=float)

    # vector direction from the surface to the target
    v_brain = probe_dir_brain(hist_rad, roll_rad)

    # distance travelled along the probe to reach the target from the DV = 0 plane
    surface_dist = T[2] / v_brain[2]

    # get the new entry point
    E = T - (surface_dist * v_brain)
    dials = entry_to_dials(E, hist_rad, roll_rad)

    return {
        "entry_AP": np.round(E[0], 2),
        "entry_ML": np.round(E[1], 2),
        "dial_AP": np.round(dials[0], 2),
        "dial_ML": np.round(dials[1], 2),
        "dial_DV_offset": np.round(dials[2], 2),
        "travel_DV": np.round(surface_dist, 2)
    }


def get_new_angles(target_loc, AP_entry, ML_entry, roll_rad):
    '''
    Given a known target location in the brain [AP, ML, DV], a desired entry point
    (AP_entry, ML_entry) on the brain surface, and the current (measured) roll, solve
    for the pitch and probe angle needed so that the probe still hits the target,
    along with the resulting travel distance.

    This is the inverse of get_target_loc_angled: there, pitch, roll and probe_angle
    are known and entry point + travel distance are solved for; here, entry point and
    roll are known and pitch + probe_angle are solved for.

    target_loc : array-like [AP, ML, DV]
        known target location in the brain (e.g. from get_target_loc)
    AP_entry, ML_entry : float
        desired entry coordinates on the brain surface (brain frame)
    roll_rad : float
        current roll, e.g. from estimate_roll(...)

    travel_probe is the distance along the (tilted) probe axis; travel_vertical is the
    same path expressed as pure vertical drop, for a rig whose DV drive stays vertical
    while the probe is angled.
    '''
    T = np.asarray(target_loc, dtype=float)
    E = np.asarray([float(AP_entry), float(ML_entry), 0.0], dtype=float)

    # vector from desired entry point to the target, in brain coordinates
    d = T - E
    travel_probe = np.linalg.norm(d)
    d_hat = d / travel_probe

    # solve for the head rotation that zeroes out the AP component of d_hat once
    # rotated into the stereotaxic frame (the probe can only tilt in DV/ML, so its
    # stereo-frame AP component must be zero)
    hist_rad = np.arctan2(-d_hat[0], d_hat[2])
    d_pitched = rotate_ML(hist_rad) @ d_hat

    # apply the known roll; whatever's left in the ML-DV plane determines the probe
    # angle. rotate_AP preserves the zeroed AP component, so the solution is exact.
    v_stereo = rotate_AP(roll_rad) @ d_pitched
    probe_angle_rad = np.arctan2(-v_stereo[1], v_stereo[2])

    # convert the head rotation back to a beak bar angle
    pitch_deg = pitch_from_hist_rad(hist_rad)
    dials = entry_to_dials(E, hist_rad, roll_rad)

    return {
        "pitch_deg": np.round(pitch_deg, 2),
        "probe_angle_rad": probe_angle_rad,
        "probe_angle_deg": np.round(np.rad2deg(probe_angle_rad), 2),
        "travel_probe": np.round(travel_probe, 2),
        "travel_vertical": np.round(travel_probe * np.cos(probe_angle_rad), 2),
        "dial_AP": np.round(dials[0], 2),
        "dial_ML": np.round(dials[1], 2),
        "dial_DV_offset": np.round(dials[2], 2)
    }
