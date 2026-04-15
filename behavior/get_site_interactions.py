import numpy as np
from matplotlib.path import Path
from scipy.signal import medfilt
from scipy.io import loadmat, savemat
'''
Used ChatGPT (checked and modified by IL) to convert and modify SC
countHexInteractions and parts of runSiteIntGUI from RigControl/arena alignment 
'''
''' Set root directory '''
root_dir = "Z:/Isabel/data/hpc_implants/" # locker
# root_dir = "C:/Users/ilow1/Documents/code/bird_pose_tracking/model_output/" # local - update as needed
bird_id = 'AMB154'
session_id = '241202'
pred_id = '260413'
session_root = f"{root_dir}{bird_id}/{bird_id}_{session_id}/"


''' Define paths to models, smoothed keypoints, arena info '''
behavior_folder = f"{session_root}/behavior_data/"
pred_file = f'{pred_id}_posture_2stage_face.npy'
pred_path = f"{behavior_folder}{pred_file}"
pos_file = 'posture_pos_smooth.npy'
vel_file = 'posture_vel_smooth.npy'
use_raw_pos = False

arena_dir = 'C:/Users/Isabel/Documents/code/il_rig_control/arena_alignment/'
arena_items_file = 'arena_items_2.mat'

''' Define mat file to save params and data '''
save_file = "seed_struct.mat"
save_path = f"{behavior_folder}{save_file}"


''' Helper functions '''
def detect_stateChanges_selfmerge(state_matrix):
    '''
    Determines the start and end time for each state change.
    Merges state changes when there is no state transition.

    Params
    ------
    state_matrix : bool, shape (n_frames, n_arena_objects)

    Returns
    -------
    onset_times : ndarray, shape (n_states, )
        state onset times (in frames)
    offset_times : ndarray, shape (n_states, )
        state offset times (in frames)
    obj_num : ndarray, shape (n_states, )
        arena object index
    '''
    # Check state matrix shape
    if len(state_matrix.shape) == 1:
        state_matrix = state_matrix[:, None]
    assert len(state_matrix.shape) == 2, "State matrix must be at most 2D"


    # Count state changes, including start and final
    state_vector = np.any(state_matrix, axis=1)
    state_vector_padded = np.concatenate(([False], state_vector, [False])).astype(int)
    onset_times = np.where(np.diff(state_vector_padded) > 0.5)[0]
    offset_times = np.where(np.diff(state_vector_padded) < -0.5)[0]
    if offset_times[-1] == state_matrix.shape[0]:
        offset_times[-1] = offset_times[-1] - 1
        if offset_times[-1] == onset_times[-1]:
            onset_times[-1] = onset_times[-1] - 1
    
    # Check the number of states and the onset/offset times
    assert np.all(np.count_nonzero(state_matrix, axis=1) <= 1), "State includes more than one object at once!"
    obj_num = np.argmax(state_matrix[onset_times], axis=1)
    assert len(obj_num) == len(onset_times), "Each state must correspond to an object"
    assert np.all(offset_times - onset_times > 0.5), "Offset times precede onset times!"
    
    # Merge states that do not transition between different objects
    same_obj = np.where(np.diff(obj_num) == 0)[0]
    onset_times = np.delete(onset_times, same_obj + 1) # Remove the next entry (no state transition)
    offset_times = np.delete(offset_times, same_obj)  # Remove this exit (no state transition)
    
    # Check the number of states
    obj_num = np.argmax(state_matrix[onset_times], axis=1)
    assert len(obj_num) == len(onset_times), "Each state must correspond to an object"

    return onset_times, offset_times, obj_num

def detect_stateChanges_othermerge(state_matrix, other_times, dur_thresh):
    '''
    Determines the start and end time for each state change.

    Params
    ------
    state_matrix : bool, shape (n_frames, n_arena_objects)
    other_times
    dur_thresh : float
        minimum number of frames for events to be considered separate

    Returns
    -------
    onset_times : ndarray, shape (n_states, )
        state onset times (in frames)
    offset_times : ndarray, shape (n_states, )
        state offset times (in frames)
    obj_num : ndarray, shape (n_states, )
        state index number
    '''
    # Check state matrix shape
    if len(state_matrix.shape) == 1:
        state_matrix = state_matrix[:, None]
    assert len(state_matrix.shape) == 2, "State matrix must be at most 2D"

    # Count state changes, including start and final
    state_vector = np.any(state_matrix, axis=1)
    state_vector_padded = np.concatenate(([False], state_vector, [False])).astype(int)
    onset_times = np.where(np.diff(state_vector_padded) > 0.5)[0]
    offset_times = np.where(np.diff(state_vector_padded) < -0.5)[0]
    if offset_times[-1] == state_matrix.shape[0]:
        offset_times[-1] = offset_times[-1] - 1
        if offset_times[-1] == onset_times[-1]:
            onset_times[-1] = onset_times[-1] - 1
    inter_event_dur = onset_times[1:] - offset_times[:-1]

    # Check the number of states and the onset/offset times
    obj_num = np.argmax(state_matrix[onset_times], axis=1)
    assert len(obj_num) == len(onset_times), "Each state must correspond to an object"
    assert np.all(offset_times - onset_times > 0.5), "Offset times precede onset times!"

    # Look for transitions between objects
    same_obj = np.where(np.diff(obj_num) == 0)[0]
    merge_same = np.ones(len(same_obj), dtype=bool)
    for i, n_int in enumerate(same_obj):
        this_onset = onset_times[n_int]
        next_onset = onset_times[n_int + 1]
        this_interval = inter_event_dur[n_int]
        # If there is an interposing event and minimum inter-event interval, do not merge
        if (np.any((other_times > this_onset) & (other_times < next_onset)) & (this_interval > dur_thresh)):
            merge_same[i] = False
        else:
            merge_same[i] = True
    merge_interactions = same_obj[merge_same]

    # Merge interactions for when the bird did not transition to another object
    onset_times = np.delete(onset_times, merge_interactions + 1)  # Remove the next entry (no state transition)
    offset_times = np.delete(offset_times, merge_interactions)  # Remove this exit (no state transition)
    
    # Check the number of states
    obj_num = np.argmax(state_matrix[onset_times], axis=1)
    assert len(obj_num) == len(onset_times), "Each state must correspond to an object"

    return onset_times, offset_times, obj_num


''' Main function for "primal" action detection '''
def count_arena_interactions(smooth_pts, foot_speed, body_reproj_error, arena_data):
    '''
    Params
    ------
    smooth_pts : ndarray, shape (n_frames, 3, n_keypoints) todo check this!
        predicted keypoint locations
        output from kalman filter (bird_pose_tracking/pkm_utils.kf_smooth_preds)
        in normalized arena coords (1 = 13 in = 13*2.54 cm)
    foot_speed : ndarray, shape (n_frames,)
        avg foot speed in norm coords / sec
    body_reproj : ndarray, shape (n_frames,)
        3D body location from COM model
    arena_data : dict
        coordinates of arena objects
        output from il_rig_control/arena_alignment/sort_arena_items.m

    Returns
    -------
    data : dict of "primal" arena interactions
    '''
    ''' Set interaction params '''
    # todo:
    #   check height thresholds and adjust as needed
    #   check cache tolerance params and adjust/discard --> not using for now
    params = {
        "reproj_thresh": 10,  # Maximum reprojection error to count as a valid frame (pixels)
        "speed_thresh": 1/2,  # Threshold for feet 'not moving' (normalized units/second, e.g., output of Kalman filter)
        "cache_height_thresh": 0.023,  # Threshold for beak low enough for site interaction
        "merge_dur_thresh": 50,  # Threshold below which events at the same site *must* be merged (in frames at 50 fps)
        "feeder_height_thresh": 0.05,  # Threshold for beak low enough for feeder interaction
        "water_height_thresh": 0.06,  # Threshold for beak low enough for water interaction
        "water_radius_thresh": 0.625 / 13,  # Threshold for beak close enough to the center for water dish interaction
        "beak_foot_dist_thresh": 0.03,  # Distance threshold to count beak and feet near enough for eating
        # "cache_radius_tol": 1.005,  # Scale factor to adjust cache site locations (>1 means further from arena center) to better match the center of beak interactions
        "state_median_win": 5,  # Median window for filtering state status to prevent transient blips
    }

    ''' Calculate temporary variables '''
    # beak and foot positions
    beak_pos = np.mean(smooth_pts[:, [0, 1]], axis=1)  # avg beak position
    foot_pos = np.mean(smooth_pts[:, [10, 14]], axis=1)  # avg foot position
    beak_foot_dist = np.sqrt(np.sum((beak_pos - foot_pos)**2, axis=1))  # Euclidean dist beak to foot

    # data params
    n_frames = beak_pos.shape[0]  # Number of frames
    n_cache_sites = len(arena_data["caches"])  # Number of cache sites
    n_perches = len(arena_data["perches"])  # Number of perches
    n_feeders = len(arena_data["feeder_perches"])  # Number of feeder perches

    ''' Filter variables '''
    valid_frames = body_reproj_error < params["reproj_thresh"]
    feet_still = foot_speed < params["speed_thresh"]
    beak_low_cache = beak_pos[:, 2] < params["cache_height_thresh"]
    beak_low_feeder = beak_pos[:, 2] < params["feeder_height_thresh"]
    beak_low_water = beak_pos[:, 2] < params["water_height_thresh"]
    beak_low_feet = beak_foot_dist < params["beak_foot_dist_thresh"]
    beak_radius_water = np.sqrt(np.sum(beak_pos[:, :2]**2, axis=1)) < params["water_radius_thresh"]
    # beak_radius_feeder = np.sqrt(np.sum(beak_pos[:, :2]**2, axis=1)) < params["feeder_radius_thresh"]

    ''' Detect interactions '''
    print('Detecting perch interactions...')
    # Feet on perches
    feet_on_perch = np.zeros((n_frames, n_perches + n_feeders), dtype=bool)
    
    for n in range(n_perches):
        convex_hull = arena_data["perches"][n]["ConvexHull"]
        path = Path(convex_hull)
        tmp = path.contains_points(foot_pos[:, :2])  # Check if points are in polygon
        feet_on_perch[:, n] = tmp & feet_still & valid_frames

    for n in range(n_feeders):
        convex_hull = arena_data["feeder_perches"][n]["ConvexHull"]
        path = Path(convex_hull)
        tmp = path.contains_points(foot_pos[:, :2])
        feet_on_perch[:, n_perches + n] = tmp & feet_still & valid_frames

    # Beak on feet
    print('Detecting beak-feet interactions...')
    # todo: why is this not just feet_on_perch & beak_low_feet?
    beak_on_feet = np.zeros((n_frames, n_perches + n_feeders), dtype=bool)

    for n in range(n_perches):
        convex_hull = arena_data["perches"][n]["ConvexHull"]
        path = Path(convex_hull)
        tmp = path.contains_points(beak_pos[:, :2])
        beak_on_feet[:, n] = tmp & beak_low_feet & feet_still & valid_frames

    for n in range(n_feeders):
        convex_hull = arena_data["feeder_perches"][n]["ConvexHull"]
        path = Path(convex_hull)
        tmp = path.contains_points(beak_pos[:, :2])
        beak_on_feet[:, n_perches + n] = tmp & beak_low_feet & feet_still & valid_frames

    # Beak on cache sites
    print('Detecting beak-cache interactions...')
    beak_on_cache = np.zeros((n_frames, n_cache_sites), dtype=bool)

    for n in range(n_cache_sites):
        convex_hull = np.array(arena_data["caches"][n]["ConvexHull"]) # * params["cache_radius_tol"]
        path = Path(convex_hull)
        tmp = path.contains_points(beak_pos[:, :2])
        beak_on_cache[:, n] = tmp & beak_low_cache & feet_still & valid_frames

    # Beak on feeders
    print('Detecting feeder and water interactions...')
    beak_on_feeder = np.zeros((n_frames, n_feeders), dtype=bool)

    for n in range(n_feeders):
        convex_hull = arena_data["feeders"][n]["ConvexHull"]
        path = Path(convex_hull)
        tmp = path.contains_points(beak_pos[:, :2])
        beak_on_feeder[:, n] = tmp & beak_low_feeder & feet_still & valid_frames

    # Beak on water
    beak_on_water = beak_low_water & beak_radius_water & feet_still & valid_frames

    # Median filter interactions
    state_median_win = params["state_median_win"]
    print(f"Median state filtering with {state_median_win} frames")
    feet_on_perch = medfilt(feet_on_perch.astype(float), kernel_size=(state_median_win, 1)).astype(bool)
    beak_on_feet = medfilt(beak_on_feet.astype(float), kernel_size=(state_median_win, 1)).astype(bool)
    beak_on_cache = medfilt(beak_on_cache.astype(float), kernel_size=(state_median_win, 1)).astype(bool)
    beak_on_feeder = medfilt(beak_on_feeder.astype(float), kernel_size=(state_median_win, 1)).astype(bool)
    beak_on_water = medfilt(beak_on_water.astype(float), kernel_size=state_median_win).astype(bool)


    ''' Merge operations '''
    print('Merging interactions...')
    # Define padded arrays to include start and end times
    beak_on_feet_padded = np.concatenate(([False], np.any(beak_on_feet, axis=1), [False]))
    beak_on_cache_padded = np.concatenate(([False], np.any(beak_on_cache, axis=1), [False]))
    beak_on_feeder_padded = np.concatenate(([False], np.any(beak_on_feeder, axis=1), [False]))
    beak_on_water_padded = np.concatenate(([False], beak_on_water, [False]))

    # Define exclusion times
    exc_eat = np.where(np.diff(beak_on_feet_padded) > 0.5)[0]
    exc_site = np.where(np.diff(beak_on_cache_padded) > 0.5)[0]
    exc_feed = np.where(np.diff(beak_on_feeder_padded) > 0.5)[0]
    exc_water = np.where(np.diff(beak_on_water_padded) > 0.5)[0]

    # Always merge perch interactions at the same site
    new_perch, end_perch, perch_num = detect_stateChanges_selfmerge(feet_on_perch)

    # Merge feeder/water interactions unless the bird does something else in between
    tmp = np.concatenate((new_perch, exc_eat, exc_site, exc_water))
    new_feeder, end_feeder, feeder_num = detect_stateChanges_othermerge(
        beak_on_feeder, tmp, params["merge_dur_thresh"]
    )

    tmp = np.concatenate((new_perch, new_feeder, exc_eat, exc_site))
    new_water, end_water, water_num = detect_stateChanges_othermerge(
        beak_on_water, tmp, params["merge_dur_thresh"]
    )

    # Merge eating interactions
    tmp = np.concatenate((new_perch, new_feeder, new_water, exc_site))
    new_beak_perch, end_beak_perch, beak_perch_num = detect_stateChanges_othermerge(
        beak_on_feet, tmp, params["merge_dur_thresh"]
    )

    # Merge cache site interactions
    tmp = np.concatenate((new_perch, new_water, new_feeder, new_beak_perch))
    new_site, end_site, site_num = detect_stateChanges_othermerge(
        beak_on_cache, tmp, params["merge_dur_thresh"]
    )   


    ''' collect everything into a dictionary '''
    data = {
        "newPerch": new_perch,
        "endPerch": end_perch,
        "perchNum": perch_num,
        "newSite": new_site,
        "endSite": end_site,
        "siteNum": site_num,
        "newFeeder": new_feeder,
        "endFeeder": end_feeder,
        "feederNum": feeder_num,
        "newWater": new_water,
        "endWater": end_water,
        "waterNum": water_num,
        "newBeakPerch": new_beak_perch,
        "endBeakPerch": end_beak_perch,
        "beakPerchNum": beak_perch_num,
        "params": params,
    }

    return data

''' load the model output and get the body reprojection error '''
print('Getting reprojection error...')
results_dict = np.load(pred_path, allow_pickle=True).item()
results = results_dict['results']
body_reproj_error = results['com_rep_err'][:, 1]

''' load the raw/smoothed keypoints and calculate the foot speed '''
print('Loading smoothed postural keypoints and velocity...')
if use_raw_pos:
    smooth_pts = results['posture_preds']
    foot_vel = np.diff(smooth_pts[:, [10, 14]], axis=0)*50
    foot_speed = np.sqrt(np.sum(np.mean(foot_vel, axis=1)**2, axis=1))
    foot_speed = np.insert(foot_speed, 0, foot_speed[0])
else:
    smooth_pts = np.load(f"{behavior_folder}{pos_file}") # n_frames, n_keypoints, 3
    smooth_vel = np.load(f"{behavior_folder}{vel_file}") # n_frames, n_keypoints, 3
    foot_speed = np.sqrt(np.sum(np.mean(smooth_vel[:, [10, 14]], axis=1)**2, axis=1))


''' load the arena objects '''
print('Getting arena objects...')
arena_data = loadmat(f'{arena_dir}{arena_items_file}', squeeze_me=True)
arena_data["perches"] = arena_data["perch_w_site"]
arena_data["feeder_perches"] = arena_data["perch_no_site"]


''' Set the seed detection params for the matlab siteInteractionGUI '''
# "Primal action" detection
print('\nPrimal action detection')
seed_struct = {}
seed_struct["countData"] = count_arena_interactions(smooth_pts,
                                                    foot_speed,
                                                    body_reproj_error,
                                                    arena_data)

# Parameters and thresholds
seed_struct["bk_height_seedDetect"] = 0.02  # Height threshold for site interactions
seed_struct["smSeedWindow"] = 11  # Frame width for median-filtering seed detection
seed_struct["gainThresh"] = 0.9  # Threshold to count as gain
seed_struct["loseThresh"] = 0.1  # Threshold to count as loss
seed_struct["minLoseDur"] = 12  # Losses followed by gain within this timeframe are ignored
seed_struct["validFrames"] = np.convolve(body_reproj_error, np.ones(27) / 27, mode="same") < 13 # todo check this
seed_struct["seedIntTol"] = 0  # Tolerance for overlap of site-interaction-end and seed Loss/Gain
seed_struct["path"] = session_root  # Session path

# Average top and bottom beak positions
beak_pos = np.mean(smooth_pts[:, [0, 1]], axis=1)
seed_struct["beakPos"] = beak_pos

# Model predictions for seed/no seed
sm_seed = results["face_preds"].copy()
seed_struct["smSeed"] = np.squeeze(sm_seed)

# Discard predictions during site interactions
seed_struct["smSeed"][beak_pos[:, 2] < seed_struct["bk_height_seedDetect"]] = np.nan

# Median filter the seed predictions
seed_struct["smSeed"] = medfilt(seed_struct["smSeed"], kernel_size=seed_struct["smSeedWindow"])

# Save as a matlab struct
savemat(save_path, {'seedStruct':seed_struct})