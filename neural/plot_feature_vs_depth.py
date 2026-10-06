import numpy as np

import os
import sys
sys.path.append("..//utils/")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# reuse the feature list, pooling and categorical coloring from the feature grid
from plot_feature_grid import FEATURES, FEATURE_KEY, gather, groups_for

'''
Version of plot_fr_vs_depth.py using the v2 waveform features.

One figure per feature in PLOT_FEATURES: cells plotted by depth (y) and v2
waveform width (x), one column per shank, columns sorted posterior -> anterior.
Categorical features ('wf_cluster', 'polarity') are colored discretely (same
colors/legend as the feature grid); signed features in SIGNED are colored with
bwr, centered on 0; everything else is colored with viridis.

Width = 'width_ms': dominant extremum -> largest opposite lobe, so it is
defined the same way for negative- and positive-dominant units.
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs_dir = f"../figures/basic_neural_analysis/unit_features/"
data_file = f"{root_dir}good_session_data.npy"

''' Features to plot (one figure each) '''
# 'wf_cluster' (GMM cluster ID), 'polarity', or any key in LABELS below
PLOT_FEATURES = ['wf_cluster', 'pk_trough_ratio', 'polarity', 'stable_rate', 'frac_stable',
                 'burst_index', 'asymmetry', 'weighted_dist_um']
X_KEY = 'width_ms'

# key: (colorbar label, log color scale); grid features plus the stability measures
# (stable_rate is already in FEATURES). Add any other per-cell key here to plot it.
STABILITY = [('frac_stable', 't stable / t total', False),
             ('amp_drop', 'amp. drop', False),
             ('rate_amp_rho', 'rate-amp. corr. (rho)', False),
             ('presence_ratio', 'presence ratio', False)]
LABELS = {k: (lab, lg) for k, lab, lg in FEATURES + STABILITY}
LABELS['rate_change'] = ('log2 rate change', False)   # last / first third of session
CATEGORICAL = ['wf_cluster', 'polarity']
# signed features: bwr, symmetric about 0 (the last four aren't plotted by default)
SIGNED = ['rate_change', 'rate_amp_rho', 'asymmetry', 'deriv_ratio',
          'trough_peak_lag_ms', 'asymmetry_raw', 'deriv_ratio_raw', 'main_offset_ms']

''' Fig params '''
title_size = 14
axis_label = 12
tick_label = 9
ylims = [7010, 3990]
xlims = [0, 1.2]          # width_ms is searched up to 1.2 ms after the extremum
alpha_pts = 0.8
size_pts = 6

''' Load the data dictionary '''
data_dict = np.load(data_file, allow_pickle=True).item()
save_dir = f"{save_figs_dir}/"
os.makedirs(save_dir, exist_ok=True)

''' Pool features across birds/sessions, with matching position / shank / rate-change arrays '''
feats, labels, birds, sessions = gather(data_dict, return_index=True)
if labels is not None:
    feats['wf_cluster'] = labels
assert X_KEY in feats, f"'{X_KEY}' not in the features (re-run collect_waveform_data?)"

cell_pos, shank_idx, rate_change = [], [], []
for bird, session_id in dict.fromkeys(zip(birds, sessions)):   # unique, in pooled order
    session_data = data_dict[bird][session_id]
    # position of each cell (ML, est AP, DV)
    cell_pos.append(session_data['lhy_cell_pos'] if 'lhy_cell_pos' in session_data
                    else session_data['cell_pos'])
    shank_idx.append(np.asarray(session_data['shank_idx']).astype(int))

    # rate change over the session, from the per-chunk rates unit_stability saved
    # ('stability_chunks'; chunks tile the session in time order, >= 60 s each):
    # log2(mean rate in the last third of chunks / mean rate in the first third).
    # nan with < 2 chunks, no chunk data, or no spikes in either third
    for ch in session_data[FEATURE_KEY]['stability_chunks']:
        r = np.array([]) if ch is None else np.asarray(ch['rate'], dtype=float)
        k = max(len(r) // 3, 1)
        early, late = (r[:k].mean(), r[-k:].mean()) if len(r) >= 2 else (0, 0)
        rate_change.append(np.log2(late / early) if early > 0 and late > 0 else np.nan)
cell_pos = np.vstack(cell_pos)
shank_idx = np.concatenate(shank_idx)
feats['rate_change'] = np.asarray(rate_change)
assert len(birds) == len(shank_idx) == len(feats['rate_change']), 'cell positions and features are misaligned'

# jitter, drawn once so every feature's figure has the same point positions
rng = np.random.default_rng(0)
jit_dv = rng.standard_normal(len(birds)) * 2
jit_w = rng.standard_normal(len(birds)) * (2 / 600)

''' One column per (bird, shank), sorted posterior -> anterior '''
columns = []
for bird in dict.fromkeys(birds):   # unique, in order
    in_bird = birds == bird
    n_shanks = int(data_dict[bird].get('n_shanks', shank_idx[in_bird].max() + 1))
    exclude = np.asarray(data_dict[bird].get('exclude_shank', np.zeros(n_shanks, dtype=bool)))

    # shanks with no probe track in histology: positions come from the intended
    # surgery coordinates, not a measured track
    raw_tip = data_dict[bird].get('raw_tip_coords')
    no_hist = (np.zeros(n_shanks, dtype=bool) if raw_tip is None
               else np.isnan(np.asarray(raw_tip)[:, :2]).any(axis=1))
    lhy_dvs = data_dict[bird].get('lhy_dvs', np.full((n_shanks, 2), np.nan))

    for sh in range(n_shanks):
        mask = in_bird & (shank_idx == sh)
        if exclude[sh] or not mask.any():
            continue
        dv = cell_pos[mask, -1]
        columns.append(dict(bird=bird, mask=mask, ap=cell_pos[mask, 1].mean(),
                            dv=dv - dv.min() if dv.min() < 0 else dv,   # min DV on shank -> 0
                            no_hist=no_hist[sh], lhy=lhy_dvs[sh]))
columns.sort(key=lambda c: (np.isnan(c['ap']), c['ap']))
n_cols = len(columns)

''' Plot each feature by width/depth '''
for key in PLOT_FEATURES:
    if key not in feats:
        print(f"skipping '{key}': not in the features (or no wf_cluster labels for every session)")
        continue
    label, log = LABELS.get(key, (key, False))

    if key in CATEGORICAL:
        # (legend label with n, mask over all cells, color) -- same as the feature grid
        layers = groups_for(feats, feats['wf_cluster'] if key == 'wf_cluster' else None)
    else:
        v = np.asarray(feats[key], dtype=float).copy()
        if log:
            v[v <= 0] = np.nan                  # not drawable on a log scale
        lo, hi = np.nanpercentile(v, [0.5, 99.5])   # central 99%, same for every column
        if key in SIGNED:                       # symmetric, so 0 is white
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        norm = (LogNorm if log else Normalize)(lo, hi)
        cmap = 'bwr' if key in SIGNED else 'viridis'

    gs_kw = dict(hspace=0.1, wspace=0.3)
    f, ax = plt.subplots(1, n_cols, figsize=(0.85*n_cols, 5),
                         sharey=True, gridspec_kw=gs_kw)
    ax = np.atleast_1d(ax)  # subplots returns a bare Axes when n_cols == 1

    for col, c in enumerate(columns):
        m = c['mask']
        x = feats[X_KEY][m] + jit_w[m]
        y = c['dv'] + jit_dv[m]
        if key in CATEGORICAL:
            for name, lay_mask, color in layers:    # background layers first
                k = lay_mask[m]
                ax[col].scatter(x[k], y[k], c=color, alpha=alpha_pts,
                                s=size_pts, lw=0, zorder=1)
        else:
            ok = np.isfinite(v[m])                  # cells missing this feature: hollow gray
            ax[col].scatter(x[~ok], y[~ok], facecolors='none', edgecolors='xkcd:gray',
                            alpha=alpha_pts, s=size_pts, lw=0.4, zorder=1)
            # thin outline on signed (bwr) plots, so values near 0 (white) stay visible
            ax[col].scatter(x[ok], y[ok], c=v[m][ok], cmap=cmap, norm=norm,
                            edgecolors='xkcd:gray', lw=0.2 if key in SIGNED else 0,
                            alpha=alpha_pts, s=size_pts, zorder=1)

        # plot the LHy boundaries relative to the probe location
        ax[col].hlines(c['lhy'], *xlims, colors='xkcd:scarlet', linestyles='dashed', lw=0.5)

        # add a reference line and title
        ax[col].vlines(0.5, ylims[1], ylims[0], colors='xkcd:gray', linestyles='dashed', lw=0.5)
        ax[col].set_title(f"{c['bird']}", fontsize=axis_label)

    # universal formatting
    for col in range(n_cols):
        ax[col].set_xlim(xlims)
        ax[col].spines['right'].set_visible(False)
        ax[col].spines['top'].set_visible(False)
        ax[col].spines['left'].set_bounds(ylims[0]-10, ylims[1]+10)
        ax[col].set_xticks([0, 0.5, 1])
        ax[col].set_xticklabels(['0', '0.5', '1'])
        ax[col].tick_params(labelsize=tick_label)
    ax[0].set_ylim(ylims)
    ax[0].set_ylabel('depth from surface (um)', fontsize=axis_label)
    f.supxlabel('spike width (ms)', fontsize=axis_label, y=0.02)
    f.suptitle(f"{label}: shanks sorted " + r"posterior $\rightarrow$ anterior",
               fontsize=axis_label, y=0.98)

    # colorbar (continuous) or legend (categorical)
    if key in CATEGORICAL:
        handles = [Line2D([], [], marker='o', ls='', color=color, label=name)
                   for name, _, color in layers]
        f.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.91, 0.9),
                 fontsize=9, frameon=False)
    else:
        cax = f.add_axes([0.93, 0.8, 0.008, 0.1])
        cbar = f.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        # ticks at min, center and max of the color range (center = 0 for signed,
        # geometric mean for log); label reads top -> bottom, facing the bar
        ticks = norm.inverse([0, 0.5, 1])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.3g}' for t in ticks])
        cbar.minorticks_off()
        cbar.set_label(label, rotation=-90, va='bottom')

    # flag any shank with no probe track in histology.
    # adjacent flagged columns share a single box + label, so the labels don't
    # collide when several shanks in a row are missing histology
    # flag_cols = [col for col, c in enumerate(columns) if c['no_hist']]
    # col_groups = []
    # for col in flag_cols:
    #     if col_groups and col == col_groups[-1][-1] + 1:
    #         col_groups[-1].append(col)
    #     else:
    #         col_groups.append([col])

    # pad_x, pad_y = 0.006, 0.01
    # for grp in col_groups:
    #     bbox_first = ax[grp[0]].get_position()
    #     bbox_last = ax[grp[-1]].get_position()
    #     rect = Rectangle(
    #         (bbox_first.x0 - pad_x, bbox_first.y0 - pad_y),
    #         (bbox_last.x1 - bbox_first.x0) + 2 * pad_x, bbox_first.height + 2 * pad_y,
    #         transform=f.transFigure, fill=False,
    #         edgecolor='xkcd:scarlet', linewidth=1, clip_on=False,
    #     )
    #     f.add_artist(rect)
    #     # wrap onto 2 lines over a single column, where one line doesn't fit
    #     text = 'no histology' if len(grp) > 1 else 'no\nhistology'
    #     f.text((bbox_first.x0 + bbox_last.x1) / 2, bbox_first.y1 - pad_y,
    #            text, color='xkcd:scarlet', ha='center', va='top',
    #            fontsize=axis_label, transform=f.transFigure)

    f.savefig(f'{save_dir}width_by_depth_{key}.png', dpi=400, bbox_inches='tight')
    plt.close(f)
