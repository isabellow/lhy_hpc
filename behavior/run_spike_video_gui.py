#!/usr/bin/env python
"""
run_spike_video_gui.py
======================

Set the parameters below and run:

    python run_spike_video_gui.py

Keys
----
    space           play / pause
    left / right    step one frame back / forward   (+shift = 10 frames)
    up / down       shrink / grow the heatmap time window
    c / r / k / e / f   jump to next cache / retrieve / check / eating bout /
                        feeder visit   (+shift = previous one)
    + / -           zoom the video in / out
    h               grab the video (move the mouse to drag it); h again to drop
    z               back to full frame
    d               switch between the spike raster and the rate heatmap
    n               toggle across-cell normalization (heatmap only)
    click a row     select that cell (ID in the status bar); click again to clear
    [ / ]           slower / faster playback
    q or esc        quit
"""

import numpy as np

from spike_video_gui import launch, sort_cells_by_peak_time  # noqa: F401


# --------------------------------------------------------------------------- #
#  PARAMETERS                                                                  #
# --------------------------------------------------------------------------- #

bird = 'LMN86'
session_id = '260831'
root_dir = 'Z:/Isabel/data/lhy_implants/'
session_dir = f'{root_dir}{bird}/{bird}_{session_id}/'

# path to the behavior videos
vid_dir = f'{session_dir}'

# path to the behavior data and aligned_spikes.npy
data_dir = f'{session_dir}behavior_data/'

# what sorting to use
sort_to_use = 'rastermap' # or try 'waveform', 'depth'

# which camera to show -- expects <vid_dir>/<cam_id>.avi
cam_id = 'blue_cam'

# 'raster' draws every spike as a tick the height of its cell's row, jittered
# within its video frame; 'heatmap' draws normalized firing rate.  Switch live
# with 'd'.  The normalization parameters below only apply to the heatmap, and
# are only computed if you ask for it.
display_mode = 'raster'

# rescale every cell to its own robust range (True) so low-SD cells are not
# washed out, or leave the population-vector normalization on one shared scale
# (False).  Heatmap only; toggle live with 'n'.
norm_across_cells = True

# frames of firing rate to show on each side of the current frame
# (at 50 Hz, 250 frames = 5 s each side, 10 s total)
spike_t_window = 250

# SD of the Gaussian display smoothing, in video frames; None = no smoothing
# (at 50 Hz, 5 frames = 100 ms SD).  Applied after normalization, so it does
# not change the units.  Heatmap only -- the raster shows every spike.
smoothing = None

# phy cluster IDs to plot; None = all rows of aligned_spikes.npy
# cell_ids = None
cell_ids = [42, 15, 18, 34, 53, 61]

# indices that sort the rows of aligned_spikes.npy into the desired order;
# None = keep the order they are stored in
cell_sort = None
if sort_to_use == 'rastermap':
    sort_path = f'{data_dir}aligned_spikes_embedding.npy'
    _rm = np.load(sort_path, allow_pickle=True).item()
    cell_sort = np.asarray(_rm['isort']).ravel()

# cell_sort = np.load(f'{data_dir}depth_sort.npy')


# --------------------------------------------------------------------------- #
#  OPTIONAL EXTRAS  (fine to leave alone)                                      #
# --------------------------------------------------------------------------- #

# lhy_hpc checkout, so format_behavior_data / helpers can be imported.
# None = infer it from where spike_video_gui.py lives (works if this script
# sits in the repo, e.g. in behavior/).
repo_root = None

# kilosort folder holding cluster_group.tsv, for the phy cluster IDs.
# None = look for <data_dir>/../*/*kilosort*/cluster_group.tsv
ks_dir = None
# ks_dir = f'{session_dir}{bird}_{ephys_id}/kilosort4/'

# must match the only_good used to build aligned_spikes.npy
only_good = True

fps = 50

# population-vector normalization, as in collect_population_vectors:
#   norm_fr = (inst_fr - moving_avg(inst_fr, baseline_window)) / (SD + std_reg)
# set pop_norm=False to show plain Hz instead
pop_norm = True
std_reg = 0.6                 # 0.6 in build_data_dict, 1e-2 in pop_normalize
baseline_window = 30          # running baseline, see baseline_units
baseline_units = 'minutes'    # 'minutes' or 'frames'
# when helpers is importable, call helpers.moving_avg(row, window=baseline_window)
# verbatim so the display matches your population vectors exactly.  Set False to
# use the local uniform_filter1d baseline with the units above.
use_repo_moving_avg = True

# False = raw event bounds (get_cache_ints / get_checks_raw / ...)
# True  = the SC/EM 2024 analysis windows (get_*_refined, +/-250 ms and
#         truncated against neighbouring interactions)
refined_events = False

playback_speed = 1.0
video_downsample = 1          # 2 halves a 1896x640 frame if playback stutters

# override the event loader entirely, e.g. to add feeder open/closed splits:
#   import format_behavior_data as fbd
#   seed_struct, count_data = fbd.load_behavior_data(data_dir)
#   events = {'cache': fbd.get_cache_ints(count_data, seed_struct), ...}
events = None

# phy cluster ID per row of aligned_spikes.npy; None = read the tsv
cluster_ids = None


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    launch(vid_dir=vid_dir,
           data_dir=data_dir,
           cam_id=cam_id,
           norm_across_cells=norm_across_cells,
           spike_t_window=spike_t_window,
           smoothing=smoothing,
           cell_ids=cell_ids,
           cell_sort=cell_sort,
           repo_root=repo_root,
           ks_dir=ks_dir,
           only_good=only_good,
           fps=fps,
           pop_norm=pop_norm,
           std_reg=std_reg,
           baseline_window=baseline_window,
           baseline_units=baseline_units,
           use_repo_moving_avg=use_repo_moving_avg,
           refined_events=refined_events,
           display_mode=display_mode,
           playback_speed=playback_speed,
           video_downsample=video_downsample,
           events=events,
           cluster_ids=cluster_ids)
