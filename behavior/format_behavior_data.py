import numpy as np
from scipy import stats
import pandas as pd
import warnings
import sys
sys.path.append("../utils/")
from load_matlab_data import loadmat_sbx
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy import stats
from scipy.signal import medfilt
from matplotlib.path import Path

"""
INDEXING NOTE
------------
runSiteIntGUI_2.m increments siteNum, feederNum, waterNum and beakPerchNum
when it writes annotatedSeeds.mat, but leaves perchNum alone. So in every
count_data loaded here:

    siteNum, feederNum, waterNum, beakPerchNum : 1-INDEXED
    perchNum                                   : 0-INDEXED

TODO update this somewhere at the root to correct indexing confusion
(currently everything is patched to work downstream)
"""


''' Load and format behavior data '''
def load_behavior_data(data_dir):
    seed_struct = loadmat_sbx(f'{data_dir}annotatedSeeds.mat')['annotatedSeeds']
    count_data = seed_struct['countData']
    return seed_struct, count_data


''' Basic behavior data '''
def get_position(data_dir):
    # load the posture tracking data
    smooth_posture_preds = np.load(f'{data_dir}posture_pos_smooth.npy') # time x keypoints x xyz

    # get the rough head position (between the eyes)
    eye_xy = smooth_posture_preds[:, [7, 11], :2] # left and right eye xy positions
    avg_head_xy = np.mean(eye_xy, axis=1)

    return avg_head_xy

def get_speed(pos_2d, smooth_sig=50, fps=50):
    smooth_pos = gaussian_filter(pos_2d, sigma=smooth_sig, axes=0)
    abs_speed_xy  = np.sqrt(
        np.diff(smooth_pos[:, 0]) ** 2 +
        np.diff(smooth_pos[:, 1]) ** 2
    ) * fps
    abs_speed_xy  = np.append(abs_speed_xy, np.nan)  # pad to match n_frames

    return abs_speed_xy


''' Classify arena interactions '''
def get_cache_ints(count_data, seed_struct, return_site_idx=False):
    '''
    Caches are site interactions where a seed is added

    Params
    ------
    return_site_idx : bool
        also return the cache site of each event.  Default False, so the
        two-value call signature used by existing scripts is unchanged.

    Returns
    -------
    cache_onsets, cache_offsets : arrays, shape (n_caches,)
    cache_site_idx : int array, shape (n_caches,)   only if return_site_idx
        0-INDEXED site, i.e. siteNum - 1, matching get_checks_raw and the
        event_site_idx expected by get_site_occupancy / get_site_status.
        NOTE this differs from get_caches_refined, which returns the raw
        1-indexed siteNum (see module INDEXING NOTE).
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # caches = add a seed
    cache_onsets = all_int_start[all_int_changes > 0]
    cache_offsets = all_int_end[all_int_changes > 0]

    if return_site_idx:
        all_site_num = np.asarray(count_data['siteNum']).astype(int)
        cache_site_idx = all_site_num[all_int_changes > 0] - 1  # siteNum is 1-indexed
        return cache_onsets, cache_offsets, cache_site_idx
    return cache_onsets, cache_offsets

def get_retrieve_ints(count_data, seed_struct, return_site_idx=False):
    '''
    Retrievals are site interactions where a seed is removed

    Params
    ------
    return_site_idx : bool
        also return the cache site of each event.  Default False, so the
        two-value call signature used by existing scripts is unchanged.

    Returns
    -------
    ret_onsets, ret_offsets : arrays, shape (n_retrievals,)
    ret_site_idx : int array, shape (n_retrievals,)   only if return_site_idx
        0-INDEXED site, i.e. siteNum - 1, matching get_checks_raw and the
        event_site_idx expected by get_site_occupancy / get_site_status.
        NOTE this differs from get_retrievals_refined, which returns the raw
        1-indexed siteNum (see module INDEXING NOTE).
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # caches = add a seed
    ret_onsets = all_int_start[all_int_changes < 0]
    ret_offsets = all_int_end[all_int_changes < 0]

    if return_site_idx:
        all_site_num = np.asarray(count_data['siteNum']).astype(int)
        ret_site_idx = all_site_num[all_int_changes < 0] - 1  # siteNum is 1-indexed
        return ret_onsets, ret_offsets, ret_site_idx
    return ret_onsets, ret_offsets

def get_caches_refined(count_data, seed_struct, n_total_frames, dt=0.02):
    '''
    Caches are site interactions where a seed is added

    Returns cache onset and offset times, 
    as well as the perch ID for each cache
    
    Define a cache window as in SC, EM 2024
    - 240 ms before cache onset to 240 ms after cache offset
    - truncated to avoid other interactions
    - caches > 2 sec, only include 1 sec after onset and 1 sec before offset
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_site_idx = count_data['siteNum']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # caches = add a seed
    cache_onsets_raw = all_int_start[all_int_changes > 0]
    cache_offsets_raw = all_int_end[all_int_changes > 0]
    cache_perch_idx = all_site_idx[all_int_changes > 0]

    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    n_perches = all_perch_start.shape[0]

    # +/-240 ms window around cache, avoiding other events
    t_window = 0.24/dt
    cache_onsets = np.asarray([])
    cache_offsets = np.asarray([])
    for cs, ce in zip(cache_onsets_raw, cache_offsets_raw):
        # create the time window
        cache_start = cs - t_window
        cache_end = ce + t_window

        # get the perch index for this cache event
        temp_perch_start = all_perch_start.copy()
        temp_perch_start[temp_perch_start > cs] = 0
        cache_idx = np.argmin(cs-temp_perch_start)

        # check for overlap with other events
        if cache_idx > 0:
            if all_perch_end[cache_idx-1] >= cache_start:
                cache_start = all_perch_end[cache_idx-1]
        if cache_idx < n_perches-1:
            if all_perch_start[cache_idx+1] <= cache_end:
                cache_end = all_perch_start[cache_idx+1]

        # check session ends
        if cache_start < 0:
            cache_start = 0
        if cache_end > n_total_frames:
            cache_end = n_total_frames

        cache_onsets = np.append(cache_onsets, cache_start)
        cache_offsets = np.append(cache_offsets, cache_end)

    # cache_perch_idx is siteNum, i.e. 1-INDEXED (see module INDEXING NOTE)
    return cache_onsets.astype(int), cache_offsets.astype(int), cache_perch_idx.astype(int)

def get_retrievals_refined(count_data, seed_struct, n_total_frames, dt=0.02):
    '''
    Retrievals are site interactions where a seed is removed

    Returns retrieval onset and offset times, 
    as well as the perch ID for each retrieval
    
    Define a retrieval window as in SC, EM 2024
    - 240 ms before retrieval onset to 240 ms after retrieval offset
    - truncated to avoid other interactions
    - retrievals > 2 sec, only include 1 sec after onset and 1 sec before offset
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_site_idx = count_data['siteNum']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # retrievals = remove a seed
    ret_onsets_raw = all_int_start[all_int_changes < 0]
    ret_offsets_raw = all_int_end[all_int_changes < 0]
    ret_perch_idx = all_site_idx[all_int_changes < 0]

    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    n_perches = all_perch_start.shape[0]

    # +/-240 ms window around retrieval, avoiding other events
    t_window = 0.24/dt
    ret_onsets = np.asarray([])
    ret_offsets = np.asarray([])
    for cs, ce in zip(ret_onsets_raw, ret_offsets_raw):
        # create the time window
        ret_start = cs - t_window
        ret_end = ce + t_window

        # get the perch index for this retrieval event
        temp_perch_start = all_perch_start.copy()
        temp_perch_start[temp_perch_start > cs] = 0
        ret_idx = np.argmin(cs-temp_perch_start)

        # check for overlap with other events
        if ret_idx > 0:
            if all_perch_end[ret_idx-1] >= ret_start:
                ret_start = all_perch_end[ret_idx-1]
        if ret_idx < n_perches-1:
            if all_perch_start[ret_idx+1] <= ret_end:
                ret_end = all_perch_start[ret_idx+1]

        # check session ends
        if ret_start < 0:
            ret_start = 0
        if ret_end > n_total_frames:
            ret_end = n_total_frames

        ret_onsets = np.append(ret_onsets, ret_start)
        ret_offsets = np.append(ret_offsets, ret_end)

    # ret_perch_idx is siteNum, i.e. 1-INDEXED (see module INDEXING NOTE)
    return ret_onsets.astype(int), ret_offsets.astype(int), ret_perch_idx.astype(int)

def get_checks_refined(count_data, seed_struct, n_total_frames, dt=0.02,
                        return_keep=False):
    '''
    Checks are site interactions where the contents are unchanged

    Returns check onset and offset times, 
    as well as the perch ID for each check
    
    Define a check window as in SC, EM 2024
    - 250 ms before check onset to 250 ms after check offset
    - truncated to avoid other interactions
    - exclude checks > 1.5 sec (approximation of other non-check interactions)

    return_keep : bool
        also return a boolean mask over the RAW checks (site interactions with
        no seed change) marking which ones survived the duration cut.  Needed
        to line other per-check quantities - occupancy, in particular - up with
        the returned events.
    '''
    # get all site interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    all_site_idx = count_data['siteNum']
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    n_interactions = all_int_start.shape[0]

    # checks = seed unchanged
    check_onsets_raw = all_int_start[all_int_changes == 0]
    check_offsets_raw = all_int_end[all_int_changes == 0]
    check_perch_idx = all_site_idx[all_int_changes == 0]

    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    n_perches = all_perch_start.shape[0]

    # +/-240 ms window around check, avoiding other events
    t_window = 0.24/dt
    check_onsets = np.asarray([])
    check_offsets = np.asarray([])
    # checks longer than 1.5 s are dropped below, so track which raw checks
    # survive: check_perch_idx was previously returned unfiltered, so it was
    # longer than the onsets/offsets and every perch ID after the first
    # dropped check belonged to a different event
    check_keep = np.zeros(check_onsets_raw.shape[0], dtype=bool)
    for i, (cs, ce) in enumerate(zip(check_onsets_raw, check_offsets_raw)):
        # create the time window
        check_start = cs - t_window
        check_end = ce + t_window

        # get the perch index for this retrieval event
        temp_perch_start = all_perch_start.copy()
        temp_perch_start[temp_perch_start > cs] = 0
        check_idx = np.argmin(cs-temp_perch_start)

        # check for overlap with other events
        if check_idx > 0:
            if all_perch_end[check_idx-1] >= check_start:
                check_start = all_perch_end[check_idx-1]
        if check_idx < n_perches-1:
            if all_perch_start[check_idx+1] <= check_end:
                check_end = all_perch_start[check_idx+1]

        # check session ends
        if check_start < 0:
            check_start = 0
        if check_end > n_total_frames:
            check_end = n_total_frames

        # check duration
        if (check_end - check_start) > (1.5/dt):
            continue

        check_keep[i] = True
        check_onsets = np.append(check_onsets, check_start)
        check_offsets = np.append(check_offsets, check_end)

    # check_perch_idx is siteNum, i.e. 1-INDEXED (see module INDEXING NOTE)
    if return_keep:
        return (check_onsets.astype(int), check_offsets.astype(int),
                check_perch_idx[check_keep].astype(int), check_keep)
    return (check_onsets.astype(int), check_offsets.astype(int),
            check_perch_idx[check_keep].astype(int))

def get_checks_raw(count_data, seed_struct, max_check_dur=1.5, dt=0.02):
    '''
    Checks are site interactions where the contents are unchanged, lasting no
    more than max_check_dur seconds.

    Mirrors get_visits_raw / get_cache_ints / get_retrieve_ints: bare event
    onset and offset with no padding or truncation, for aligning to the real
    event boundaries (a raster, an onset/offset PSTH).

    Returns
    -------
    check_onsets, check_offsets : int arrays, shape (n_checks,)
    check_site_idx : int array, shape (n_checks,)
        0-indexed cache site for each check
    '''
    all_int_start = np.asarray(count_data['newSite']).astype(int)
    all_int_end = np.asarray(count_data['endSite']).astype(int)
    all_site_num = np.asarray(count_data['siteNum']).astype(int)
    all_int_changes = np.sum(np.atleast_2d(np.asarray(seed_struct['seedChanges'])), axis=1)

    is_check = all_int_changes == 0
    short_enough = (all_int_end - all_int_start) <= (max_check_dur / dt)
    keep = is_check & short_enough

    check_onsets = all_int_start[keep]
    check_offsets = all_int_end[keep]
    check_site_idx = all_site_num[keep] - 1        # siteNum is 1-indexed

    return check_onsets.astype(int), check_offsets.astype(int), check_site_idx.astype(int)


''' Individual beak cache site touches (unmerged) '''
# defaults mirror count_arena_interactions in get_site_interactions.py.  Only
# the params that feed the beak-on-cache state matrix are needed here.
BEAK_TOUCH_PARAMS = dict(
    reproj_thresh=10,            # max reprojection error for a valid frame (px)
    speed_thresh=1 / 2,          # feet 'not moving' (norm units/s)
    cache_height_thresh=0.023,   # beak low enough for a site interaction
    state_median_win=5,          # median filter on the state, to kill blips
)


def get_beak_site_touches(data_dir, arena_data, pred_file=None, params=None,
                          count_data=None, pos_file='posture_pos_smooth.npy',
                          vel_file='posture_vel_smooth.npy',
                          body_reproj_error=None, min_dur_frames=1,
                          return_state=False, warn=True):
    '''
    Every individual beak-on-cache-site touch, WITHOUT merging.

    Recomputes the beak-on-cache state matrix exactly as the
    "Beak on cache sites" section of get_site_interactions.py does — same
    keypoints, same thresholds, same median filter — then returns each
    contiguous run of that state as its own event.  get_site_interactions.py
    instead hands the state matrix to detect_stateChanges_othermerge, which
    glues consecutive touches at one site into a single "site interaction"
    unless the bird did something else in between; those merged interactions
    are what ends up in count_data as newSite / endSite / siteNum, and the
    individual touches inside them cannot be recovered from that, which is why
    this has to go back to the pose data.

    Runs are detected per site (one column at a time), so a touch that moves
    straight from one site to the next comes back as two events rather than
    one, and nothing is ever merged across a gap.

    Params
    ------
    data_dir : str
        session behavior_data folder, as passed to load_behavior_data
    arena_data : dict
        arena objects, e.g. loadmat(f'{arena_dir}{arena_items_file}',
        squeeze_me=True).  Only arena_data['caches'] is used.
    pred_file : str or None
        name of the pose model output in data_dir
        ('<pred_date>_posture_2stage_face.npy'), used for the body
        reprojection error.  None looks for a single *posture_2stage_face.npy
        in data_dir.  Ignored if body_reproj_error is given.
    params : dict or None
        thresholds, keys as BEAK_TOUCH_PARAMS.  Anything missing falls back to
        count_data['params'] if count_data is given, then to
        BEAK_TOUCH_PARAMS.  Pass count_data to reproduce the thresholds this
        session's saved interactions were actually detected with.
    count_data : dict or None
        as loaded by load_behavior_data, only read for count_data['params']
    pos_file, vel_file : str
        smoothed keypoint position / velocity files in data_dir
    body_reproj_error : array (n_frames,) or None
        skips loading pred_file if you already have it
    min_dur_frames : int
        drop touches shorter than this many frames.  1 keeps everything with a
        non-zero duration; the median filter has already removed brief blips.
    return_state : bool
        also return the (n_frames, n_cache_sites) bool state matrix

    Returns
    -------
    touch_onsets, touch_offsets : int arrays, shape (n_touches,)
        first frame of the touch, and the frame after the last one, in the
        same half-open convention as count_data newSite / endSite.  Sorted by
        onset.
    touch_site_idx : int array, shape (n_touches,)
        0-INDEXED cache site, i.e. matching get_checks_raw and the
        event_site_idx expected by get_site_status / get_expectation_status,
        NOT the 1-indexed count_data siteNum (see module INDEXING NOTE).
    beak_on_cache : bool array (n_frames, n_cache_sites)   only if return_state
    '''
    # resolve thresholds: explicit params, then the session's own, then defaults
    p = dict(BEAK_TOUCH_PARAMS)
    if count_data is not None:
        saved = count_data['params'] if 'params' in count_data else None
        if saved is not None:
            for key in BEAK_TOUCH_PARAMS:
                try:
                    val = saved[key]
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                if val is not None and np.size(val) == 1:
                    p[key] = float(np.asarray(val).item())
    if params is not None:
        p.update(params)

    # beak position and foot speed, as in count_arena_interactions
    smooth_pts = np.load(f'{data_dir}{pos_file}')      # frames x keypoints x xyz
    smooth_vel = np.load(f'{data_dir}{vel_file}')
    beak_pos = np.mean(smooth_pts[:, [0, 1]], axis=1)  # avg of the two beak pts
    foot_speed = np.sqrt(np.sum(np.mean(smooth_vel[:, [10, 14]], axis=1) ** 2,
                                axis=1))
    n_frames = beak_pos.shape[0]

    # body reprojection error, for the valid-frame mask
    if body_reproj_error is None:
        if pred_file is None:
            from glob import glob
            hits = sorted(glob(f'{data_dir}*posture_2stage_face.npy'))
            if len(hits) != 1:
                raise FileNotFoundError(
                    f'expected one *posture_2stage_face.npy in {data_dir}, '
                    f'found {len(hits)}; pass pred_file')
            pred_path = hits[0]
        else:
            pred_path = f'{data_dir}{pred_file}'
        results = np.load(pred_path, allow_pickle=True).item()['results']
        body_reproj_error = results['com_rep_err'][:, 1]
    body_reproj_error = np.asarray(body_reproj_error)

    # frame filters
    valid_frames = body_reproj_error < p['reproj_thresh']
    feet_still = foot_speed < p['speed_thresh']
    beak_low_cache = beak_pos[:, 2] < p['cache_height_thresh']

    # beak inside each cache site's hull
    n_cache_sites = len(arena_data['caches'])
    beak_on_cache = np.zeros((n_frames, n_cache_sites), dtype=bool)
    for n in range(n_cache_sites):
        convex_hull = np.array(arena_data['caches'][n]['ConvexHull'])
        path = Path(convex_hull)
        tmp = path.contains_points(beak_pos[:, :2])
        beak_on_cache[:, n] = tmp & beak_low_cache & feet_still & valid_frames

    # same median filter on the state as get_site_interactions.py
    state_median_win = int(p['state_median_win'])
    if state_median_win > 1:
        beak_on_cache = medfilt(beak_on_cache.astype(float),
                                kernel_size=(state_median_win, 1)).astype(bool)

    # contiguous runs, one site at a time, so nothing is merged across sites
    onsets, offsets, sites = [], [], []
    for n in range(n_cache_sites):
        col = np.concatenate(([False], beak_on_cache[:, n], [False])).astype(int)
        starts = np.flatnonzero(np.diff(col) > 0.5)
        ends = np.flatnonzero(np.diff(col) < -0.5)
        if starts.shape[0] == 0:
            continue
        # a run that reaches the last frame: keep it in range, as
        # detect_stateChanges_* does
        if ends[-1] == n_frames:
            ends[-1] = n_frames - 1
            if ends[-1] == starts[-1]:
                starts[-1] = starts[-1] - 1
        onsets.append(starts)
        offsets.append(ends)
        sites.append(np.full(starts.shape[0], n, dtype=int))

    if not onsets:
        empty = np.asarray([], dtype=int)
        if return_state:
            return empty, empty, empty, beak_on_cache
        return empty, empty, empty

    # combine everything across sites
    touch_onsets = np.concatenate(onsets)
    touch_offsets = np.concatenate(offsets)
    touch_site_idx = np.concatenate(sites)

    # drop anything too short
    keep = (touch_offsets - touch_onsets) >= max(int(min_dur_frames), 1)
    touch_onsets = touch_onsets[keep]
    touch_offsets = touch_offsets[keep]
    touch_site_idx = touch_site_idx[keep]

    # sort by time in session
    order = np.argsort(touch_onsets, kind='stable')
    touch_onsets = touch_onsets[order].astype(int)
    touch_offsets = touch_offsets[order].astype(int)
    touch_site_idx = touch_site_idx[order].astype(int)

    if return_state:
        return touch_onsets, touch_offsets, touch_site_idx, beak_on_cache
    return touch_onsets, touch_offsets, touch_site_idx


def get_visits_raw(count_data, exclude_feeders=True,
                    feeder_perches=np.asarray([84, 85, 86, 87])):
    '''
    Visits are perch interactions without eating or site interaction

    Params
    ------
    exclude_feeders : bool
        exclude visits to feeder perches
    feeder_perches : array of ints
        feeder perch ID numbers (as in get_site_interactions.py)

    Returns
    -------
    visit_onsets, visit_offsets : int arrays, shape (n_visits,)
    visit_site_idx : int array, shape (n_visits,)
        the returned IDs are 0-indexed
    '''
    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    all_perch_idx = count_data['perchNum']
    n_perches = all_perch_start.shape[0]

    # collect non-visit interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    n_interactions = all_int_start.shape[0]
    eat_onsets = count_data['newBeakPerch']
    eat_offsets = count_data['endBeakPerch']
    all_non_visit_start = np.append(eat_onsets, all_int_start)

    # get visits by excluding other interactions
    visits = np.full(n_perches, 1).astype(bool)
    for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
        this_perch = all_perch_idx[i]
        if exclude_feeders and (this_perch in feeder_perches):
            visits[i] = False
            continue
        start_idx = all_non_visit_start > ps
        end_idx = all_non_visit_start < pe
        if any(start_idx & end_idx):
            visits[i] = False        
    visit_onsets = all_perch_start[visits]
    visit_offsets = all_perch_end[visits]
    visit_site_idx = all_perch_idx[visits]

    return (visit_onsets.astype(int), visit_offsets.astype(int),
            visit_site_idx.astype(int))

def get_visits_refined(count_data, n_total_frames, dt=0.02,
                        exclude_feeders=True, feeder_perches=np.asarray([84, 85, 86, 87])):
    '''
    Visits are perch interactions without eating or site interaction

    Returns visit onset and offset times, 
    as well as the perch ID for each visit

    Define a visit window as in SC, EM 2024
    +/- 500 ms from perch arrival, truncated to avoid other interactions
    '''
    # get all perch interactions
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    all_perch_idx = count_data['perchNum']
    n_perches = all_perch_start.shape[0]

    # collect non-visit interactions
    all_int_start = count_data['newSite']
    all_int_end = count_data['endSite']
    n_interactions = all_int_start.shape[0]
    eat_onsets = count_data['newBeakPerch']
    eat_offsets = count_data['endBeakPerch']
    all_non_visit_start = np.append(eat_onsets, all_int_start)

    # get visits by excluding other interactions
    visits = np.full(n_perches, 1).astype(bool)
    for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
        start_idx = all_non_visit_start > ps
        end_idx = all_non_visit_start < pe
        this_perch = all_perch_idx[i]
        if (this_perch in feeder_perches) & exclude_feeders:
            visits[i] = False
        elif any(start_idx & end_idx):
            visits[i] = False        

    # 1000ms window around visit onset, avoiding other events
    t_window = 0.5/dt
    visit_onsets = np.asarray([])
    visit_offsets = np.asarray([])
    for i, visit_t in enumerate(all_perch_start):
        if visits[i]:
            visit_start = visit_t - t_window
            visit_end = visit_t + t_window

            # check for overlap with other events
            if i > 0:
                if all_perch_end[i-1] >= visit_start:
                    visit_start = all_perch_end[i-1]
            if i < n_perches-1:
                if all_perch_end[i] <= visit_end:
                    visit_end = all_perch_end[i]

            # check session ends
            if visit_start < 0:
                visit_start = 0
            if visit_end > n_total_frames:
                visit_end = n_total_frames

            visit_onsets = np.append(visit_onsets, visit_start)
            visit_offsets = np.append(visit_offsets, visit_end)

    return visit_onsets.astype(int), visit_offsets.astype(int), all_perch_idx[visits].astype(int)


def get_site_occupancy(count_data, seed_struct, event_onsets, event_site_idx,
                        use_init_counts=True):
    '''
    Was site occupied at the time of each event?

    Params
    ------
    count_data : dict
        interaction data from annotatedSeeds.mat
    seed_struct : dict
        the annotated seed struct ('seedChanges', 'initSeedCounts')
    event_onsets : array, shape (n_events,)
        onset frame of each event, e.g. from get_checks_raw / get_visits_raw
    event_site_idx : array, shape (n_events,)
        0-indexed cache site for each event
    use_init_counts : bool
        Include initSeedCounts, so a site that was already baited at the start
        of the session and never touched counts as occupied.
        Set False to score occupancy from within-session caching only.

    Returns
    -------
    occupied : bool array, shape (n_events,)
    '''
    seed_changes = np.atleast_2d(np.asarray(seed_struct['seedChanges']))
    init_counts = np.atleast_1d(np.asarray(seed_struct['initSeedCounts'], dtype=float))
    n_sites = seed_changes.shape[1]

    # cumulative seed count in every site as of each site interaction
    seeds_in_sites = np.cumsum(seed_changes, axis=0).astype(float)
    if use_init_counts:
        seeds_in_sites = seeds_in_sites + init_counts
    else:
        init_counts = np.zeros(n_sites)

    all_int_start = np.asarray(count_data['newSite']).astype(int)
    occupied = np.zeros(event_onsets.shape[0], dtype=bool)

    # rank each event against every site interaction's start time, once, rather
    # than once per event. side='right' places an event that IS itself a site
    # interaction (a check, whose own onset equals one of these times) after
    # its own row, so its own zero-change contribution is correctly included.
    order = np.argsort(all_int_start)
    int_start_sorted = all_int_start[order]
    k = np.searchsorted(int_start_sorted, event_onsets, side='right') - 1

    for i, (site, k_i) in enumerate(zip(event_site_idx, k)):
        if not (0 <= site < n_sites):
            continue
        if k_i < 0:
            occupied[i] = init_counts[site] > 0
        else:
            occupied[i] = seeds_in_sites[order[k_i], site] > 0

    return occupied


''' Seed provenance: baited vs cached '''
# status codes returned by get_site_status.  0/1/2/3 so they can index a list
# of colours or names directly; -1 marks an event we cannot score at all.
SITE_EMPTY, SITE_BAITED, SITE_CACHED, SITE_MIXED = 0, 1, 2, 3
SITE_UNKNOWN = -1
SITE_STATUS_NAMES = {SITE_EMPTY: 'empty', SITE_BAITED: 'baited',
                     SITE_CACHED: 'cached', SITE_MIXED: 'mixed',
                     SITE_UNKNOWN: 'unknown'}


def _seed_provenance_timeline(count_data, seed_struct, use_init_counts=True,
                              removal_rule='cached_first', warn=True):
    '''
    Baited and cached seed counts in every site, after every site interaction.

    seedChanges only records how many seeds moved, not which ones, so the two
    pools are tracked as a running ledger: every positive change adds to the
    cached pool (the bird put it there), every negative change draws down the
    pools in the order set by removal_rule.

    removal_rule only matters for a site holding both kinds at once, which is
    only possible if the bird cached into a baited site.
    'cached_first' assumes the bird takes its own seed back first; the
    alternative is 'baited_first'.  Either way the site is reported as
    SITE_MIXED for as long as both pools are non-empty, so nothing downstream
    has to trust the rule unless it chooses to keep mixed events.

    Note on clamping: a pool is never allowed below zero.  With
    use_init_counts=False the baited pool starts empty, so retrievals of
    baited seeds have nothing to draw from and are silently clamped (this is
    exactly the case where get_site_occupancy, whose cumulative sum can go
    negative, may disagree with the ledger).

    Returns
    -------
    dict with
        int_start_sorted : int array (n_interactions,)  onset frames, sorted
        baited_after, cached_after : float arrays (n_interactions, n_sites)
            pool sizes immediately AFTER each interaction, rows in the same
            (sorted) order as int_start_sorted
        init_baited : float array (n_sites,)  state before the first interaction
        n_sites : int
        removal_rule : str
    '''
    # load the data
    seed_changes = np.atleast_2d(np.asarray(seed_struct['seedChanges'], dtype=float))
    init_counts = np.atleast_1d(np.asarray(seed_struct['initSeedCounts'], dtype=float))
    all_int_start = np.asarray(count_data['newSite']).astype(int)
    n_interactions, n_sites = seed_changes.shape

    if removal_rule not in ('cached_first', 'baited_first'):
        raise ValueError("removal_rule must be 'cached_first' or 'baited_first'")

    # ensure interactions are in chronological order
    order = np.argsort(all_int_start, kind='stable')
    int_start_sorted = all_int_start[order]
    changes_sorted = seed_changes[order]

    # initialize baited vs. cached tallies
    init_baited = init_counts.copy() if use_init_counts else np.zeros(n_sites)
    baited = init_baited.copy()
    cached = np.zeros(n_sites)
    baited_after = np.zeros((n_interactions, n_sites))
    cached_after = np.zeros((n_interactions, n_sites))
    unaccounted = 0.0

    # walk the interactions in time order
    for j in range(n_interactions):
        row = changes_sorted[j]
        for s in np.flatnonzero(row):
            d = row[s]
            if d > 0:
                cached[s] += d
                continue
            take = -d
            pools = (cached, baited) if removal_rule == 'cached_first' else (baited, cached)
            for pool in pools:
                drawn = min(take, pool[s])
                pool[s] -= drawn
                take -= drawn
            unaccounted += take           # removal from an already-empty site
        baited_after[j] = baited
        cached_after[j] = cached

    if warn and use_init_counts and unaccounted > 0:
        warnings.warn(
            f"{unaccounted:g} seed removal(s) came from sites the ledger had "
            "already scored as empty, and were clamped at zero. Check "
            "initSeedCounts against the annotation for this session.")

    return dict(int_start_sorted=int_start_sorted, baited_after=baited_after,
                cached_after=cached_after, init_baited=init_baited,
                n_sites=n_sites, removal_rule=removal_rule)


def get_site_seed_counts(count_data, seed_struct, event_onsets, event_site_idx,
                         include_own_change=False, use_init_counts=True,
                         removal_rule='cached_first', timeline=None):
    '''
    How many baited and how many cached seeds were in the site at each event?

    Params
    ------
    count_data, seed_struct : as loaded by load_behavior_data
    event_onsets : array, shape (n_events,)
        onset frame of each event, e.g. from get_checks_raw / get_retrieve_ints
    event_site_idx : array, shape (n_events,)
        0-indexed cache site for each event
    include_own_change : bool
        False (default) scores the site as the bird arrives, i.e. strictly
        BEFORE this event's own seed change.
        True reproduces the convention used by get_site_occupancy.
    use_init_counts : bool
        Count the baited seeds (initSeedCounts).  False
        scores within-session caching only and no event can come back baited.
    removal_rule : 'cached_first' | 'baited_first'
        see _seed_provenance_timeline
    timeline : dict or None
        a pre-built _seed_provenance_timeline, to avoid rebuilding the ledger
        when scoring several event types from one session

    Returns
    -------
    n_baited, n_cached : float arrays, shape (n_events,)
        nan for events whose site index falls outside the arena
    '''
    # data params
    if timeline is None:
        timeline = _seed_provenance_timeline(
            count_data, seed_struct, use_init_counts=use_init_counts,
            removal_rule=removal_rule)
    event_onsets = np.asarray(event_onsets).astype(int)
    event_site_idx = np.asarray(event_site_idx).astype(int)
    n_sites = timeline['n_sites']

    # rank every event against the interaction start times once.
    # side='left' excludes the event's own seed change
    # side='right' includes it.
    side = 'right' if include_own_change else 'left'
    k = np.searchsorted(timeline['int_start_sorted'], event_onsets, side=side) - 1

    # tally baited vs cached
    n_baited = np.full(event_onsets.shape[0], np.nan)
    n_cached = np.full(event_onsets.shape[0], np.nan)
    for i, (site, k_i) in enumerate(zip(event_site_idx, k)):
        if k_i < 0:                       # before the first site interaction
            n_baited[i] = timeline['init_baited'][site]
            n_cached[i] = 0.0
        else:
            n_baited[i] = timeline['baited_after'][k_i, site]
            n_cached[i] = timeline['cached_after'][k_i, site]
    return n_baited, n_cached


def get_site_status(count_data, seed_struct, event_onsets, event_site_idx,
                    include_own_change=False, use_init_counts=True,
                    removal_rule='cached_first', timeline=None):
    '''
    Three-way version of get_site_occupancy: was the site empty, holding a
    baited seed, or holding a seed the bird cached?

    Params are as get_site_seed_counts.

    Returns
    -------
    status : int array, shape (n_events,)
        SITE_EMPTY 0 | SITE_BAITED 1 | SITE_CACHED 2 | SITE_MIXED 3
        SITE_UNKNOWN -1 for events whose site index falls outside the arena.
        SITE_MIXED means the site held both kinds at once and the event cannot
        be attributed; callers usually drop these.
        (status > 0) reproduces the occupancy flag for checks and visits.
    '''
    n_baited, n_cached = get_site_seed_counts(
        count_data, seed_struct, event_onsets, event_site_idx,
        include_own_change=include_own_change, use_init_counts=use_init_counts,
        removal_rule=removal_rule, timeline=timeline)

    status = np.full(n_baited.shape[0], SITE_UNKNOWN, dtype=int)
    known = np.isfinite(n_baited) & np.isfinite(n_cached)
    has_baited = known & (n_baited > 0)
    has_cached = known & (n_cached > 0)
    status[known & ~has_baited & ~has_cached] = SITE_EMPTY
    status[has_baited & ~has_cached] = SITE_BAITED
    status[has_cached & ~has_baited] = SITE_CACHED
    status[has_baited & has_cached] = SITE_MIXED
    return status


def get_expectation_status(count_data, seed_struct, event_onsets, event_site_idx,
                           timeline=None, use_init_counts=True,
                           removal_rule='cached_first', discovered_bait='cached'):
    '''
    For each event, classify the site at that event as either:
        empty | novel bait | expected seed.

    Params
    ------
    count_data, seed_struct : as loaded by load_behavior_data
    event_onsets : int array, shape (n_events,)
    event_site_idx : int array, shape (n_events,)   0-indexed
    timeline : dict or None
        a pre-built _seed_provenance_timeline.  NOTE get_site_seed_counts
        ignores use_init_counts and removal_rule when a timeline is passed, so
        build the timeline with the same settings you pass here.
    discovered_bait : 'cached' | 'drop'
        what to do with an interaction at a site that still holds a bait the
        bird has already found.  'cached' pools it with the bird's own caches
        (both are seeds it should expect); 'drop' marks it SITE_UNKNOWN so the
        caller can leave it out.

    Returns
    -------
    status : int array, shape (n_events,)
        SITE_EMPTY | SITE_BAITED | SITE_CACHED, SITE_UNKNOWN where the site is
        outside the arena or the event was dropped by discovered_bait
    is_first : bool array, shape (n_events,)
        first-encounter flag, returned so callers can report it
    '''
    #check inputs
    if discovered_bait not in ('cached', 'drop'):
        raise ValueError("discovered_bait must be 'cached' or 'drop'")
    if timeline is None:
        timeline = _seed_provenance_timeline(count_data, seed_struct,
                                             use_init_counts=use_init_counts,
                                             removal_rule=removal_rule)
    
    # data params
    n_sites = timeline['n_sites']
    onsets = np.asarray(event_onsets).astype(int)
    sites = np.asarray(event_site_idx).astype(int)
    max_frame = np.max(onsets)

    # get baits vs. cached seeds
    n_baited, n_cached = get_site_seed_counts(
        count_data, seed_struct, onsets, sites,
        include_own_change=False, use_init_counts=use_init_counts,
        removal_rule=removal_rule, timeline=timeline)
    known = np.isfinite(n_baited) & np.isfinite(n_cached)
    has_baited = known & (n_baited > 0)
    has_seed = known & ((n_baited > 0) | (n_cached > 0))

    # was this the first encounter?
    first = np.full(n_sites, max_frame+1000, dtype=np.int64)
    np.minimum.at(first, sites, onsets)
    is_first = onsets == first[sites]

    # set the status of each interaction
    status = np.full(onsets.shape[0], SITE_UNKNOWN, dtype=int)
    status[known & ~has_seed] = SITE_EMPTY
    status[has_seed] = SITE_CACHED                    # a seed the bird expects
    status[has_baited & is_first] = SITE_BAITED       # a seed it doesn't expect

    # optionally exclude discovered but not retrieved baits
    if discovered_bait == 'drop':
        status[has_baited & ~is_first] = SITE_UNKNOWN

    return status, is_first


def get_n_seeds(seed_struct):
    '''
    Get the number of seeds in the arena (roughly n cached seeds)
    at the time of each interaction
    '''
    count_data = seed_struct['countData']
    n_init = np.sum(seed_struct['initSeedCounts'])
    all_int_changes = np.sum(seed_struct['seedChanges'], axis=1)
    all_int_start = count_data['newSite']
    n_seeds_arena = np.cumsum(all_int_changes) + n_init
    if all_int_start[0] > 0:
        n_seeds_arena = np.insert(n_seeds_arena, 0, n_init)
    return n_seeds_arena


''' Barcode-related '''
def dist_binned_mean_sem(vector_correlations, vector_distances, distance_bin_edges):
    '''
    Given correlations and distances, process the data and
    return distance-binned means and SEMs
    '''
    n_bins = int(distance_bin_edges.shape[0] - 1)

    # remove nans
    keep_idx = np.abs(np.isnan(vector_correlations)-1).astype(bool)
    vector_correlations = vector_correlations[keep_idx]
    vector_distances = vector_distances[keep_idx]

    # compute the average and sem
    dist_bin_idx = np.digitize(vector_distances, distance_bin_edges)-1
    avg_correlations = np.zeros(n_bins)
    sem_correlations = np.zeros(n_bins)
    for b_idx in range(n_bins):
        avg_correlations[b_idx] = np.mean(vector_correlations[dist_bin_idx==b_idx])
        sem_correlations[b_idx] = stats.sem(vector_correlations[dist_bin_idx==b_idx]) 

    return  avg_correlations, sem_correlations

def dist_binned_mean_sem_boot(vector_correlations, vector_distances, session_ids,
                              distance_bin_edges, n_boot=1000, seed=354512,
                              min_sessions=3, return_n_sessions=False):
    '''
    Distance-binned means, with SEMs from bootstrapping over sessions,
    as in Chettih, Mackevicius et al. 2024.

    Drop-in replacement for dist_binned_mean_sem() with one extra argument.
    The returned means are identical; only the SEMs change.

    Parameters
    ----------
    vector_correlations : array, one correlation per event pair. 2D cdist
                          output is fine; it gets raveled.
    vector_distances    : array, same shape, physical distance for that pair.
    session_ids         : array, same shape, which session each pair came from.
                          Must be unique per session when pooling across birds
                          (e.g. f'{bird}_{session_id}', or a running counter).
                          Pass bird IDs instead to bootstrap over birds.
    distance_bin_edges  : bin edges, as in dist_binned_mean_sem().
    n_boot              : number of resamples. Chettih et al. used 100; 1000
                          costs little here and gives a steadier SEM.
    seed                : RNG seed, so error bars don't move between runs.
    min_sessions        : bins with fewer contributing sessions get SEM = nan.
                          See the note below -- do not set this to 1.
    return_n_sessions   : also return how many sessions contribute to each bin.

    Returns
    -------
    avg_correlations : (n_bins,) bin means, identical to dist_binned_mean_sem()
    sem_correlations : (n_bins,) SD of the resampled bin means, i.e. the
                       bootstrap SEM
    n_sessions_per_bin : (n_bins,) only if return_n_sessions=True

    Note on min_sessions
    --------------------
    If every pair in a bin comes from one session, the resampled mean is the
    same no matter how many times that session is drawn, so the bootstrap would
    return SEM exactly 0. Such bins are returned as nan instead, with a warning.
    '''
    vector_correlations = np.asarray(vector_correlations, dtype=float).ravel()
    vector_distances = np.asarray(vector_distances, dtype=float).ravel()
    session_ids = np.asarray(session_ids).ravel()

    if not (vector_correlations.shape == vector_distances.shape == session_ids.shape):
        raise ValueError(
            f"correlations {vector_correlations.shape}, distances "
            f"{vector_distances.shape} and session_ids {session_ids.shape} "
            "must all have one entry per event pair")

    n_bins = int(distance_bin_edges.shape[0] - 1)

    # remove nans, keeping all three arrays aligned
    keep_idx = ~np.isnan(vector_correlations)
    vector_correlations = vector_correlations[keep_idx]
    vector_distances = vector_distances[keep_idx]
    session_ids = session_ids[keep_idx]

    # bin every pair once, dropping anything outside the bin edges
    dist_bin_idx = np.digitize(vector_distances, distance_bin_edges) - 1
    in_range = (dist_bin_idx >= 0) & (dist_bin_idx < n_bins)

    # Per-session sum and count in each bin. A resampled bin mean is then
    # sum(sums of drawn sessions) / sum(counts of drawn sessions), so the
    # resampling loop never has to touch the (very large) pair arrays again.
    sessions, session_idx = np.unique(session_ids, return_inverse=True)
    n_sessions = sessions.shape[0]
    flat_idx = session_idx[in_range] * n_bins + dist_bin_idx[in_range]
    n_cells = n_sessions * n_bins
    bin_counts = np.bincount(
        flat_idx, minlength=n_cells).astype(float).reshape(n_sessions, n_bins)
    bin_sums = np.bincount(
        flat_idx, weights=vector_correlations[in_range],
        minlength=n_cells).reshape(n_sessions, n_bins)

    # observed means, pooling all pairs exactly as dist_binned_mean_sem() does
    total_counts = bin_counts.sum(axis=0)
    avg_correlations = np.full(n_bins, np.nan)
    np.divide(bin_sums.sum(axis=0), total_counts, out=avg_correlations,
              where=total_counts > 0)

    n_sessions_per_bin = (bin_counts > 0).sum(axis=0)

    # resample sessions with replacement
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_sessions, size=(n_boot, n_sessions))
    boot_sums = bin_sums[draws].sum(axis=1)       # (n_boot, n_bins)
    boot_counts = bin_counts[draws].sum(axis=1)   # (n_boot, n_bins)
    boot_means = np.full((n_boot, n_bins), np.nan)
    np.divide(boot_sums, boot_counts, out=boot_means, where=boot_counts > 0)

    # the bootstrap SEM is the spread of the resampled means
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sem_correlations = np.nanstd(boot_means, axis=0, ddof=1)

    # bins too thinly supported for the bootstrap to say anything
    undersupported = n_sessions_per_bin < min_sessions
    if undersupported.any():
        sem_correlations[undersupported] = np.nan
        warnings.warn(
            f"bins {list(np.flatnonzero(undersupported))} have fewer than "
            f"{min_sessions} sessions "
            f"(n = {list(n_sessions_per_bin[undersupported])}); their SEM is "
            "not estimable and is returned as nan.")

    if return_n_sessions:
        return avg_correlations, sem_correlations, n_sessions_per_bin
    return avg_correlations, sem_correlations

def spikes_by_cache(spike_frame, cache_onsets, cache_offsets, cache_window=20, dt=0.02):
    '''
    For cache offset aligned rasters, returns a matrix of spikes by cache
    sorted by duration

    cache_window : time in seconds to take around cache offset
    '''
    # params
    n_cells = spike_frame.shape[0]
    n_cache = cache_offsets.shape[0]
    cache_t_points = np.arange(-cache_window//2, cache_window//2 + dt, dt)
    n_t_pts = cache_t_points.shape[0]
    fr_halfwidth = n_t_pts//2

    # sort caches by duration
    cache_dur = cache_offsets - cache_onsets
    duration_idx = np.argsort(cache_dur)

    # matrix of spike times by cache
    cache_mat = np.zeros((n_cells, n_cache, n_t_pts), dtype=bool)
    for c_idx in range(n_cells):
        spk_times = spike_frame[c_idx]
        for i, cache_idx in enumerate(duration_idx):  
            start_idx = (cache_offsets[cache_idx] - fr_halfwidth).astype(int)
            end_idx = (cache_offsets[cache_idx] + fr_halfwidth + 1).astype(int)
            cache_mat[c_idx, i] = spk_times[start_idx:end_idx]

    # save cache onset times for plotting
    t_zero_idx = int(cache_window//2//dt + 1)
    onset_idx = np.clip(t_zero_idx - cache_dur[duration_idx], 0, len(cache_t_points) - 1)
    cache_ons = cache_t_points[onset_idx]

    return cache_mat, cache_t_points, cache_ons


''' Eating '''
def get_eating_bouts(count_data):
    '''
    Beak on feet

    Real onset/offset per bout (no padding or truncation)
    No site index or occupancy

    Returns
    -------
    eat_onsets, eat_offsets : int arrays, shape (n_eating_bouts,)
    '''
    eat_onsets = np.asarray(count_data['newBeakPerch']).astype(int)
    eat_offsets = np.asarray(count_data['endBeakPerch']).astype(int)
    return eat_onsets, eat_offsets

''' Classify feeder interactions '''
def get_feeder_ints(count_data, use_beak=True, feeder_perches=np.asarray([84, 85, 86, 87])):
    '''
    Parses count_data to extract feeder interactions.

    Params
    ------
    count_data : dict
        dict of interaction data from get_site_interactions.py
    use_beak : bool
        if True, feeder interactions are defined by the beak near the feeder
        else, feeder interactions are defined by the feet on the feeder perch
    feeder_perches : array of ints
        feeder perch ID numbers (as defined in get_site_interactions.py)

    Returns
    -------
    feeder_int_start/end : array, shape (n_feeder_int,)
        feeder interaction start/end frame numbers
        interaction is classified as the beak getting close to the feeder
        (as defined in get_site_interactions.py)
        or feet on the feeder perch, determined by the use_beak param
    feeder_idx : array of ints, shape (n_feeders,)
        the feeder ID number associated with each interaction
    '''
    if use_beak:
        ''' beak near feeder '''
        # get all feeder interactions
        feeder_int_start = count_data['newFeeder']
        feeder_int_end = count_data['endFeeder']
        feeder_idx = count_data['feederNum']
        
    else:
        ''' feet on feeder perch '''
        # get all perch interactions
        all_perch_start = count_data['newPerch']
        all_perch_end = count_data['endPerch']
        all_perch_idx = count_data['perchNum']
        n_perches = all_perch_start.shape[0]

        # get all feeder interactions
        feeder_int_start = []
        feeder_int_end = []
        feeder_idx = []
        
        for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
            this_perch = all_perch_idx[i]
            if this_perch in feeder_perches:
                feeder_int_start.append(ps)
                feeder_int_end.append(pe)
                feeder_idx.append(np.where(feeder_perches==this_perch)[0][0] + 1)
        feeder_int_start = np.asarray(feeder_int_start)
        feeder_int_end = np.asarray(feeder_int_end)
        feeder_idx = np.asarray(feeder_idx)

    return feeder_int_start, feeder_int_end, feeder_idx


def get_foot_angle(data_dir, posture_file):
    # get a vector between the two feet
    smooth_posture_preds = np.load(f'{data_dir}{posture_file}') # time x keypoints x xyz
    feet_xy = smooth_posture_preds[:, [10, 14], :2]
    feet_vector = feet_xy[:, 0] - feet_xy[:, 1]

    # compute the angle between adjacent feet vectors
    v1 = feet_vector[:-1]
    v2 = feet_vector[1:]
    mag_v1 = np.sqrt(v1[:, 0]**2 + v1[:, 1]**2)
    mag_v2 = np.sqrt(v2[:, 0]**2 + v2[:, 1]**2)
    dot = np.sum(v1*v2, axis=1)

    # regularize
    mag_v1[mag_v1 < 0.01] = 0.01
    mag_v2[mag_v2 < 0.01] = 0.01
    
    cos_theta = dot / (mag_v1*mag_v2)
    cos_theta[cos_theta < -1] = -1
    cos_theta[cos_theta > 1] = 1

    # compute the angle
    feet_angle = np.arccos(cos_theta)
    feet_angle = np.append(feet_angle, 0)
    
    return feet_angle

def get_feeder_departure_bounds(count_data, feeder_perches=np.asarray([84, 85, 86, 87])):
    ''' get feeder visits '''
    # all perches
    all_perch_start = count_data['newPerch']
    all_perch_end = count_data['endPerch']
    all_perch_idx = count_data['perchNum']
    n_perches = all_perch_end.shape[0]
    
    # feeder departures
    feeder_depart_start = []
    feeder_depart_end = []
    feeder_idx = []
    for i, (ps, pe) in enumerate(zip(all_perch_start, all_perch_end)):
        this_perch = all_perch_idx[i]
        if this_perch in feeder_perches:
            feeder_depart_start.append(ps)
            feeder_depart_end.append(pe)
            feeder_idx.append(np.where(feeder_perches==this_perch)[0][0] + 1)
    feeder_depart_start = np.asarray(feeder_depart_start)
    feeder_depart_end = np.asarray(feeder_depart_end)
    feeder_idx = np.asarray(feeder_idx)
    
    ''' refine by beak on feeder (if relevant) '''
    # last time the beak was near the feeder
    end_beak_on_feeder = count_data['endFeeder']
    for i, (fs, fe) in enumerate(zip(feeder_depart_start, feeder_depart_end)):
        after_start = end_beak_on_feeder > fs
        before_end = end_beak_on_feeder < fe
        if any(after_start & before_end):
            this_beak_end = np.argmax(np.cumsum(after_start & before_end))
            feeder_depart_start[i] = end_beak_on_feeder[this_beak_end]
    
    ''' refine to exclude eating '''
    eat_offsets = count_data['endBeakPerch']
    for i, (fs, fe) in enumerate(zip(feeder_depart_start, feeder_depart_end)):
        after_start = eat_offsets > fs
        before_end = eat_offsets < fe
        if any(after_start & before_end):
            this_offset = np.argmax(np.cumsum(after_start & before_end))
            feeder_depart_start[i] = eat_offsets[this_offset]
    assert np.sum((feeder_depart_end - feeder_depart_start) < 0) == 0

    return feeder_depart_start, feeder_depart_end, feeder_idx

def get_feeder_periods(session_info_file, bird, session_id):
    '''
    Extracts feeder open/close times from the session info spreadsheet
    and converts them into numpy arrays.
    '''
    # get the session IDs
    session_info = pd.read_excel(session_info_file, sheet_name=bird, header=1)
    session_info["id"] = session_info["date"].dt.strftime("%y%m%d")
    
    # get the feeder times for this session
    feeder_times_raw = session_info.loc[session_info["id"] == session_id,
                                                   "feeder open times"].iloc[0]
    
    # convert to numpy arrays of open and close times
    feeder_ranges = feeder_times_raw.split(sep=', ')
    feeder_open_list = []
    feeder_close_list = []
    for fr in feeder_ranges:
        times = fr.split(sep='-')
        feeder_open_list.append(times[0])
        feeder_close_list.append(times[1])
    feeder_open_times = [int(t) for t in feeder_open_list]
    feeder_close_times = [int(t) for t in feeder_close_list]

    return np.asarray(feeder_open_times), np.asarray(feeder_close_times)



def classify_feeder_ints(feeder_int_start, feeder_int_end, 
                            feeder_open_times, feeder_close_times, frame_rate=50):
    '''
    For each feeder interaction, was the feeder open or closed?

    Params
    ------
    feeder_int_start/end : array, shape (n_feeder_int,)
        feeder interaction start/end frame numbers
    feeder_open/close_times : array, shape (n_feeder_periods,)
        times in minutes that feeders opened/closed
    frame_rate : int
        video frames per second (default is 50Hz)

    Returns
    -------
    feeder_status : array of floats, shape (n_feeder_int,)
        feeder status for each interaction
        either 0 (closed the entire interaction), 1 (open the entire interaction),
        or 0.5 (closed or opened during the interaction)
    '''
    n_feeder_int = feeder_int_start.shape[0]

    # convert feeder times to frames
    feeder_open_frames = feeder_open_times*60*frame_rate
    feeder_close_frames = feeder_close_times*60*frame_rate

    # classify each interaction as open vs. closed
    # NOTE: this must be float. It used to be an int array, so the 0.5 written
    # for an interaction that spanned an open/close transition was truncated to
    # 0 and the visit was silently counted as 'closed' rather than partial.
    feeder_status = np.zeros(n_feeder_int, dtype=float)
    for i, (fs, fe) in enumerate(zip(feeder_int_start, feeder_int_end)):
        start_status = 0
        end_status = 0
        for fo, fc in zip(feeder_open_frames, feeder_close_frames):
            if (fs > fo) & (fs < fc):
                start_status = 1
            if (fe > fo) & (fe < fc):
                end_status = 1
        feeder_status[i] = np.mean((start_status, end_status))

    return feeder_status