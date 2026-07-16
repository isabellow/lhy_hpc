import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt

''' Position coding 
Based on place coding analysis in Payne et al. 2021
'''
def get_firing_by_pos(pos_xy, spike_fr, inclusion_mask, dt=1/50):
    '''
    Compute the firing rate by position bin for each cell

    Params
    ------
    pos_xy : ndarray, shape (n_frames, 2)
        xy position for each video frame
    spike_fr : ndarray, shape (n_cells, n_frames)
        spikes per video frame for each cell
    inclusion_mask : boolean array, shape (n_frames)
        True for frames that should be included in the calculation

    Returns
    ------- 
    smooth_pos_fr : ndarray, shape (n_cells, n_pos_bins, n_pos_bins)
        smoothed firing rate in each xy position bin for each cell
    smooth_pos_time : ndarray, shape (n_pos_bins, n_pos_bins)
        smoothed occupancy in each spatial bin (seconds)
    pos_edges : ndarray, shape (n_pos_bins,)
        position bin edges in normalized arena coordinates
    centers : ndarray, shape (n_pos_bins,)
        position bin centers in normalized arena coordinates
    '''
    # params
    n_pos_bins = 40 # 40 x 40 grid
    pos_max = 12/13 # arena edges in normalized coords (2 ft arena)
    pos_min = -12/13
    sm_pos_sig    = 50 # frames, i.e., 1 second
    sm_map_sigma = 3.6 # gaussian smoothing sigma in bins (xy map)
    min_occupancy = 1 # minimum seconds in spatial bin to be included

    # bin edges and centers
    pos_edges = np.linspace(pos_min, pos_max, n_pos_bins + 1)
    centers   = (pos_edges[:-1] + pos_edges[1:]) / 2

    # masked and smoothed position
    smooth_pos = gaussian_filter(pos_xy, sigma=sm_pos_sig, axes=0)
    pos_x = smooth_pos[inclusion_mask, 0]
    pos_y = smooth_pos[inclusion_mask, 1]

    # seconds per position bin
    occupancy, _, _ = np.histogram2d(pos_y, pos_x, bins=pos_edges)
    pos_time = dt * occupancy
    smooth_pos_time  = gaussian_filter(pos_time, sigma=sm_map_sigma)
    smooth_pos_time[smooth_pos_time < min_occupancy] = np.nan

    # position bin for each video frame
    x_idx = np.digitize(pos_x, pos_edges) - 1
    x_idx = np.clip(x_idx, 0, n_pos_bins - 1)
    y_idx = np.digitize(pos_y, pos_edges) - 1
    y_idx = np.clip(y_idx, 0, n_pos_bins - 1)
    unique_x = np.unique(x_idx)
    unique_y = np.unique(y_idx)

    # spikes per position bin
    n_cells = spike_fr.shape[0]
    masked_spikes = spike_fr[:, inclusion_mask]
    spk_count = np.zeros((n_cells, n_pos_bins, n_pos_bins))
    for i in unique_x:
        for j in unique_y:
            pos_idx = (x_idx == i) & (y_idx == j)
            spk_count[:, i, j] = np.sum(masked_spikes[:, pos_idx], axis=1)
    smooth_spk_count = gaussian_filter(spk_count, sigma=sm_map_sigma, axes=[1, 2])

    # firing rate per position bin
    smooth_pos_fr = smooth_spk_count / smooth_pos_time

    return smooth_pos_fr, smooth_pos_time, pos_edges, centers

''' Spatial information '''
def get_spatial_info(smooth_pos_fr, smooth_pos_time):
    '''
    Compute spatial information (bits/spike) for each cell.
    Uses the Skaggs et al. 1993 formula, as in Payne et al. 2021:
        SI = Σ_i  p_i * (λ_i / λ̄ ) * log₂(λ_i / λ̄ )

    where p_i is the fractional occupancy of bin i, λ_i is the
    mean firing rate in bin i, and λ̄ is the overall mean firing
    rate (weighted by occupancy).

    Parameters
    ----------
    smooth_pos_fr : ndarray, shape (n_cells, n_pos_bins, n_pos_bins)
        Smoothed firing rate (spikes/s) in each position bin.
    smooth_pos_time : ndarray, shape (n_pos_bins, n_pos_bins)
        Smoothed occupancy (seconds) per bin.

    Returns
    -------
    spatial_info : ndarray, shape (n_cells,)
        Spatial information in bits/spike for each cell.
        Silent cells receive a value of 0.
    '''
    n_cells = smooth_pos_fr.shape[0]

    # probability of occupying each bin (sums to 1 over occupied bins)
    total_time = np.sum(smooth_pos_time)
    p_i = smooth_pos_time / total_time

    # occupancy-weighted mean firing rate for each cell  →  λ̄
    mean_fr = np.sum(smooth_pos_fr * p_i[None, :, :], axis=(1, 2)) # (n_cells,)

    # spatial information for each cell
    spatial_info = np.zeros(n_cells)
    for c in range(n_cells):
        if mean_fr[c] == 0:
            continue
        fr_ratio = smooth_pos_fr[c] / mean_fr[c] # λ_i / λ̄
        
        # restrict to bins that are both occupied and have a positive firing rate
        valid = (p_i > 0) & (fr_ratio > 0)
        spatial_info[c] = np.sum(p_i[valid] * fr_ratio[valid] * np.log2(fr_ratio[valid]))

    return spatial_info

''' Filtering functions '''
def make_speed_mask(abs_speed, speed_threshold=5, stationary_dur=5.0, dt=1/50):
    """
    Returns a boolean mask that is True during epochs of active movement.
    Long stationary periods (gaps between moving frames > stationary_dur)
    are excluded.

    speed_threshold : float
        minimum speed (cm/s) to be considered moving
    stationary_dur : float
        minimum number of seconds of slowness to be considered stationary
    dt : float
        seconds per video frame
    """
    n_frames = abs_speed.shape[0]

    # convert from cm/s to arena coords
    speed_thresh_norm = (1 / 33.02) * speed_threshold 

    # find and mask long stationary periods
    moving_idx  = np.where(abs_speed >= speed_thresh_norm)[0]
    speed_mask = np.ones(n_frames, dtype=bool)

    # stationary at start
    if moving_idx[0] * dt > stationary_dur:
        speed_mask[:moving_idx[0]] = False

    # other long stationary periods
    t_still = np.diff(moving_idx) * dt
    long_starts = moving_idx[:-1][t_still > stationary_dur]
    long_ends = moving_idx[1:][t_still > stationary_dur]
    for start, end in zip(long_starts, long_ends):
        speed_mask[start+1:end] = False

    # stationary at end
    if (n_frames - 1 - moving_idx[-1]) * dt > stationary_dur:
        speed_mask[moving_idx[-1] + 1:] = False
    
    return speed_mask

def make_event_exclusion_mask(onsets, offsets, n_frames):
    """
    Returns a boolean mask that excludes any event (e.g., caching)
    in the provided onset/offset lists.

    Parameters
    ----------
    onsets      : 1D array, event onset times in seconds
    offsets     : 1D array, event offset times in seconds
                    (paired with onsets)
    n_frames    : int

    Returns
    -------
    event_mask : (n_frames,) bool
        False = frame is part of an event, exclude it
    """
    event_mask = np.ones(n_frames, dtype=bool)
    for onset, offset in zip(onsets, offsets):
        event_mask[onset:offset] = False
    
    return event_mask

''' Visualization '''
# universal fig params
title_size = 14
axis_label = 12
tick_label = 9

def plot_place_maps(smooth_pos_fr, spatial_info, centers,
                    n_cols=10, cmap='hot', figsize_per_cell=(1.4, 1.6),
                    normalize_to_peak=True):
    '''
    Plot 2D spatial firing-rate maps for all cells, sorted by spatial
    information (highest first), in a grid reminiscent of Fig. 2a of
    Payne et al. 2021.

    Parameters
    ----------
    smooth_pos_fr : ndarray, shape (n_cells, n_pos_bins, n_pos_bins)
        Smoothed firing rate in each xy position bin (spikes/s).
    spatial_info : ndarray, shape (n_cells,)
        Spatial information (bits/spike) for each cell.
    centers : ndarray, shape (n_pos_bins,)
        Position bin centers in normalized arena coordinates.
    n_cols : int
        Number of subplot columns.
    cmap : str
        Matplotlib colormap for the firing-rate maps (default 'hot').
    figsize_per_cell : tuple of float
        (width, height) in inches allocated to each subplot cell.
    normalize_to_peak : bool
        If True, each cell's map is normalized to its own peak firing
        rate so cells can be compared visually regardless of overall rate.
        The title shows peak FR in Hz.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (n_rows, n_cols)
    sort_idx : ndarray, shape (n_cells,)
        Indices that sort cells from highest to lowest spatial information,
    '''
    # data params
    n_cells  = smooth_pos_fr.shape[0]
    sort_idx = np.argsort(spatial_info)[::-1]   # descending SI order

    # fig params
    n_rows  = int(np.ceil(n_cells / n_cols))
    fig_w   = figsize_per_cell[0] * n_cols
    fig_h   = figsize_per_cell[1] * n_rows

    # set the colormap value for nans (low occupancy bins)
    cmap = plt.cm.get_cmap(cmap).copy()  # avoid mutating the original
    cmap.set_bad(color='w')

    # define figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten()

    # extent for imshow: [left, right, bottom, top] in arena coords
    half_bin = (centers[1] - centers[0]) / 2
    extent   = [centers[0] - half_bin, centers[-1] + half_bin,
                 centers[0] - half_bin, centers[-1] + half_bin]

    # plot the spatial firing rate map for each cell
    for plot_idx, cell_idx in enumerate(sort_idx):
        ax = axes_flat[plot_idx]
        fr_map = smooth_pos_fr[cell_idx] # (n_pos_bins, n_pos_bins)
        peak_fr = np.nanmax(fr_map)

        if normalize_to_peak and peak_fr > 0:
            display_map = fr_map / peak_fr
            vmax = 0.95
        else:
            display_map = fr_map
            vmax = 0.95*peak_fr if peak_fr > 0 else 1.0

        ax.imshow(display_map, origin='lower', extent=extent,
                  cmap=cmap, vmin=0, vmax=vmax,
                  aspect='equal', interpolation=None)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f'{peak_fr:.1f} Hz',
            fontsize=axis_label, 
            pad=2, loc='right'
        )

    # hide unused axes in the last row
    for ax in axes_flat[n_cells:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)

    return fig, axes, sort_idx

def suptitle_fixed_pad(fig, title, pad_inches=0.5):
    """
    Add a super title at a consistent spacing above the figure
    """
    y = 1.0 + pad_inches / fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=title_size, y=y)