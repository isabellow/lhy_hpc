import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# for log scaling the FR axis
def log_tick_formatter(val, pos=None):
    return f"$10^{{{val:g}}}$"

def plot_wf_clusters(asymm, width, log_fr, clu1_idx, clu2_idx):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')

    # all excitatory cells
    ax.scatter(asymm[clu1_idx],
               width[clu1_idx],
               log_fr[clu1_idx],
               c='xkcd:scarlet',
               alpha=0.4, lw=0, s=15, zorder=0)

    # all inhibitory cells
    ax.scatter(asymm[clu2_idx],
               width[clu2_idx],
               log_fr[clu2_idx],
               c='xkcd:cobalt blue', 
               alpha=0.4, lw=0, s=10, zorder=0)

    # labels
    ax.set_xlabel('spike asymmetry')
    ax.set_ylabel('spike width (ms)')
    ax.set_zlabel('firing rate (Hz)')

    # log scale the z-axis
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # limits
    ax.set_xlim([-0.75, 0.75])
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])

    # background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('xkcd:grey')
    ax.yaxis.pane.set_edgecolor('xkcd:grey')
    ax.zaxis.pane.set_edgecolor('xkcd:grey')
    ax.xaxis.pane.set_color('xkcd:light grey')
    ax.yaxis.pane.set_color('xkcd:light grey')
    ax.zaxis.pane.set_color('xkcd:light grey')

    ax.view_init(azim=45, elev=15)
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.show()

    return fig, ax



def plot_cum_fr(individual_rates, individual_bins,
                combined_rates, combined_bins, 
                colors=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    if colors == None:
        colors = []
        for i in range(len(combined_rates)):
            colors.append('k') 

    # plot individual rates
    for norm_rates, bin_vals in zip(individual_rates, individual_bins):
        ax.plot(bin_vals, norm_rates, lw=0.8, c='k', alpha=0.1)

    # plot the combined rates
    for i, (norm_rates, bin_vals) in enumerate(zip(combined_rates, combined_bins)):
        color = colors[i]
        ax.plot(bin_vals, norm_rates, lw=2, c=color, alpha=1)
    
    # log scale the x-axis
    ax.set_xlim(-3, 2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # limits
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 1)

    # labels
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('cumulative probability')

    plt.show()

    return fig, ax

def plot_fr_by_pos(cell_pos, cell_fr, dmdl_bound, cmap='jet'):
    '''
    TODO update slightly for any generic heatmap (not just firing rate)
    '''
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')

    # fig params
    n_cells = cell_pos.shape[0]
    jit = np.random.randn(2, n_cells) * 3

    # plot cell positions
    sc = ax.scatter(cell_pos[:, 0]+jit[0], cell_pos[:, 1]+jit[1], cell_pos[:, 2],
                    c=cell_fr, cmap=cmap, 
                    s=10, lw=0, zorder=1, alpha=0.7)

    # plot DM/DL boundary
    ax.scatter(dmdl_bound[0], dmdl_bound[1], np.zeros_like(dmdl_bound[0]),
                c='k', marker='.', s=1)

    # labels
    ax.set_xlabel('ML (um)')
    ax.set_ylabel('AP (um)')
    ax.set_zlabel('DV (um)')

    # set axis limits
    ax.axis('equal')
    z_lims = ax.get_zlim()
    ax.set_zlim(z_lims[1], z_lims[0])

    # background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('xkcd:grey')
    ax.yaxis.pane.set_edgecolor('xkcd:grey')
    ax.zaxis.pane.set_edgecolor('xkcd:grey')
    ax.xaxis.pane.set_color('xkcd:light grey')
    ax.yaxis.pane.set_color('xkcd:light grey')
    ax.zaxis.pane.set_color('xkcd:light grey')

    # add colorbar in the top-right corner
    cax = fig.add_axes([0.55, 0.95, 0.005, 0.18])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('log firing rate')
    max_fr = np.round(np.nanmax(cell_fr), 1)
    min_fr = np.round(np.nanmin(cell_fr), 1)
    cbar.set_ticks([sc.norm.vmin, sc.norm.vmax])
    cbar.set_ticklabels([rf'$10^{{{min_fr}}}$', rf'$10^{{{max_fr}}}$'])

    ax.view_init(azim=-50, elev=45)
    ax.set_box_aspect(aspect=None, zoom=1)
    plt.show()

    return fig, ax


def plot_bool_by_pos(cell_pos, bool_idx, dmdl_bound, 
                        labels, colors=['xkcd:scarlet', 'xkcd:cobalt blue']):
    '''
    Makes a scatter plot of N cells at cell_pos in 3D brain space,
    colored by the boolian bool_idx.

    For context, plots a dashed line at the DM/DL boundary defined by dmdl_bound.

    Adds bird/shank labels to a 3D plot at the given insertion coords

    colors : list of strings, len (2)
        color of each set of points
    labels : list of strings, len (2)
        identity of each set of points
    '''
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')

    # fig params
    n_cells = cell_pos.shape[0]
    jit = np.random.randn(2, n_cells) * 3


    # get the indices for the rest of the data
    if bool_idx.dtype == bool:
        not_bool_idx = np.abs(bool_idx-1).astype(bool)
    else:
        all_indices = np.arange(n_cells, dtype=int)
        not_bool_idx = np.setdiff1d(all_indices, bool_idx)

    # split the data
    cell_pos_A = cell_pos[bool_idx]
    cell_pos_B = cell_pos[not_bool_idx]
    jit_A = jit[:, bool_idx]
    jit_B = jit[:, not_bool_idx]

    # plot cell positions, split by bool
    sc = ax.scatter(cell_pos_A[:, 0]+jit_A[0], cell_pos_A[:, 1]+jit_A[1], cell_pos_A[:, 2],
                    c=colors[0], label=labels[0],
                    s=10, lw=0, zorder=1, alpha=0.6)
    sc = ax.scatter(cell_pos_B[:, 0]+jit_B[0], cell_pos_B[:, 1]+jit_B[1], cell_pos_B[:, 2],
                    c=colors[1],  label=labels[1],
                    s=10, lw=0, zorder=1, alpha=0.3)

    # plot DM/DL boundary
    ax.scatter(dmdl_bound[0], dmdl_bound[1], np.zeros_like(dmdl_bound[0]),
                c='k', marker='.', s=2)

    # labels
    ax.set_xlabel('ML (um)')
    ax.set_ylabel('AP (um)')
    ax.set_zlabel('DV (um)')

    # set axis limits
    ax.axis('equal')
    z_lims = ax.get_zlim()
    ax.set_zlim(z_lims[1], z_lims[0])

    # background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('xkcd:grey')
    ax.yaxis.pane.set_edgecolor('xkcd:grey')
    ax.zaxis.pane.set_edgecolor('xkcd:grey')
    ax.xaxis.pane.set_color('xkcd:light grey')
    ax.yaxis.pane.set_color('xkcd:light grey')
    ax.zaxis.pane.set_color('xkcd:light grey')

    # add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), markerscale=2)

    ax.view_init(azim=-78, elev=34)
    ax.set_box_aspect(aspect=None, zoom=1)
    plt.show()

    return fig, ax


def add_bird_labels(fig, ax, bird_shank_list, insertion_coords):
    '''
    Add bird/shank labels to a 3D plot at the given insertion coords
    '''
    assert len(bird_shank_list) == insertion_coords.shape[0]

    for i, bird_shank in enumerate(bird_shank_list):
        ml_insert, ap_insert = insertion_coords[i]
        dv_insert = 0
        ax.text(ml_insert, ap_insert, dv_insert, bird_shank,
                size='x-small', weight='semibold',
                ha='left', va='bottom'
                # zdir=(0, 200, -200),
                # transform_rotates_text=False
                )
    return fig, ax
