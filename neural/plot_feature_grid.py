import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

import os
import sys
sys.path.append("..//utils/")
import waveform_analysis

'''
Pairwise scatter plots of waveform and spike-train features, in a grid, to
see which axes separate cells and how the GMM clusters fall on each.

Cells are pooled across every bird and session in the data dict. Colors come
from the saved 'wf_cluster' labels (cluster_waveform_features in
build_data_dict), or from polarity alone if COLOR_BY = 'polarity' or the
labels are missing. Lower triangle: scatter for each feature pair; diagonal:
histogram of each feature per group.
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Params '''
FEATURE_KEY = 'wf_features_v2'
COLOR_BY = 'wf_cluster'          # or 'polarity'
# (key, axis label, log scale); any per-cell key of the features dict
FEATURES = [
    ('stable_rate', 'rate (Hz)', True),
    # ('inv_median_isi_hz', '1 / median ISI (Hz)', True),
    ('cv2', 'CV2', False),
    ('burst_index', 'burst index', False),
    # ('acg_tau_rise_ms', 'ACG tau rise (ms)', True),
    ('width_ms', 'width (ms)', False),
    # ('half_width_ms', 'half-width (ms)', False),
    ('deriv_ratio', 'deriv. ratio (log)', False),
    ('asymmetry', 'asymmetry', False),
    # ('lobe_log_ratio', 'lobe ratio', False), # (log, after/before)
    ('pk_trough_ratio', 'peak / |trough|', True),
    ('weighted_dist_um', 'spread (um)', False),
]
SHOW_UNCLASSIFIED = True         # draw cells with code -1 in light gray
PANEL_IN = 1.45                  # inches per panel

CLUSTER_COLORS = ['xkcd:saffron', 'xkcd:cobalt blue', 'xkcd:grass green', 'xkcd:scarlet', 'xkcd:purple',
                  'xkcd:teal', 'xkcd:magenta', 'xkcd:mustard', 'xkcd:sky blue']
CODE_STYLE = {   # code: (legend label, color), drawn under the clusters
    -1: ('unclassified', 'xkcd:light gray'),
    -2: ('positive', 'xkcd:saffron'),
    -3: ('ambiguous', 'xkcd:dark gray'),
    -5: ('outlier', 'k'),
    -6: ('noise (spread)', 'xkcd:light brown'),
}


def gather(data_dict, feature_key=FEATURE_KEY, return_index=False):
    '''
    Pool per-cell features and wf_cluster labels across birds and sessions.
    Labels are None if any session with features lacks them.

    With return_index=True, also returns (bird, session_id) arrays with one
    entry per pooled cell, so cells can be matched back to their session.
    '''
    pooled, labels, have_labels = {}, [], True
    n_sessions = 0
    index = []   # (bird, session_id, n_cells) per pooled session
    for bird in data_dict:
        if not isinstance(data_dict[bird], dict) or 'all_sessions' not in data_dict[bird]:
            continue
        for session_id in data_dict[bird]['all_sessions']:
            session_data = data_dict[bird][session_id]
            f = session_data.get(feature_key)
            if f is None:
                continue
            n = len(f['polarity'])
            for k, v in f.items():
                if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
                    pooled.setdefault(k, []).append(v)
            lab = session_data.get('wf_cluster')
            if lab is None or len(lab) != n:
                have_labels = False
            else:
                labels.append(np.asarray(lab))
            n_sessions += 1
            index.append((bird, session_id, n))
    feats = {k: np.concatenate(v) for k, v in pooled.items() if len(v) == n_sessions}
    feats = waveform_analysis.add_derived_features(feats)   # e.g. lobe_log_ratio
    labels = np.concatenate(labels) if have_labels and labels else None
    if not return_index:
        return feats, labels
    n_cells = [n for _, _, n in index]
    birds = np.repeat(np.array([b for b, _, _ in index], dtype=object), n_cells)
    sessions = np.repeat(np.array([s for _, s, _ in index], dtype=object), n_cells)
    return feats, labels, birds, sessions


def groups_for(feats, labels):
    '''(legend label with n, mask, color) layers, drawn in order (background first)'''
    pol = np.asarray(feats['polarity'])
    if labels is None:
        layers = [('trough-dominant', pol == -1, 'xkcd:cobalt blue'),
                  ('ambiguous', pol == 0, CODE_STYLE[-3][1]),
                  ('positive', pol == 1, CODE_STYLE[-2][1])]
    else:
        layers = []
        for code in (-1, -6, -3, -2, -5):
            if code == -1 and not SHOW_UNCLASSIFIED:
                continue
            name, color = CODE_STYLE[code]
            layers.append((name, labels == code, color))
        for k in range(labels.max() + 1):
            layers.append((f'cluster {k}', labels == k, CLUSTER_COLORS[k % len(CLUSTER_COLORS)]))
    return [(f'{name} (n={int(m.sum())})', m, color) for name, m, color in layers if m.any()]


def plot_feature_grid(feats, labels, features=FEATURES):
    features = [f for f in features if f[0] in feats]
    n = len(features)
    layers = groups_for(feats, labels)
    vals = []
    for key, _, lg in features:
        v = np.asarray(feats[key], dtype=float).copy()
        if lg:
            v[v <= 0] = np.nan                     # not drawable on a log axis
        vals.append(v)

    fig, axs = plt.subplots(n, n, figsize=(PANEL_IN * n, PANEL_IN * n))
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if j > i:
                ax.axis('off')
                continue
            xk, xlab, xlog = features[j]
            yk, ylab, ylog = features[i]
            if i == j:
                # histogram per group, on shared bins
                v = vals[i][np.isfinite(vals[i])]
                if v.size:
                    lo, hi = np.percentile(v, [0.5, 99.5])
                    bins = (np.logspace(np.log10(lo), np.log10(hi), 40) if xlog
                            else np.linspace(lo, hi, 40))
                    for name, m, color in layers:
                        x = vals[i][m & np.isfinite(vals[i])]
                        if x.size:
                            ax.hist(x, bins=bins, color=color, alpha=0.6, histtype='stepfilled', lw=0)
                ax.set_yticks([])
            else:
                for name, m, color in layers:
                    ok = m & np.isfinite(vals[i]) & np.isfinite(vals[j])
                    ax.scatter(vals[j][ok], vals[i][ok], s=2, c=color, lw=0, alpha=0.7,
                               label=name, rasterized=True)
                if ylog:
                    ax.set_yscale('log')
            if xlog:
                ax.set_xscale('log')
            # axis limits from the central 99% of cells, so a few extremes don't squash the rest
            for v, lg, setter in ((vals[j], xlog, ax.set_xlim),) + (((vals[i], ylog, ax.set_ylim),) if i != j else ()):
                fin = v[np.isfinite(v)]
                if fin.size:
                    lo, hi = np.percentile(fin, [0.5, 99.5])
                    pad = (hi / lo) ** 0.05 if lg else 0.05 * (hi - lo)
                    setter((lo / pad, hi * pad) if lg else (lo - pad, hi + pad))
            # log axes: plain-number labels at decades only, no minor labels
            plain = FuncFormatter(lambda v, p: f'{v:g}')
            axes_log = [(ax.xaxis, xlog)] + ([(ax.yaxis, ylog)] if i != j else [])
            for axis, lg in axes_log:
                if lg:
                    axis.set_major_locator(LogLocator(base=10, numticks=4))
                    axis.set_major_formatter(plain)
                    axis.set_minor_formatter(NullFormatter())
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
            ax.tick_params(which='both', labelsize=6, length=2)
            if i == n - 1:
                ax.set_xlabel(xlab, fontsize=8)
            else:
                ax.tick_params(which='both', labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(ylab, fontsize=8)
            elif j > 0:
                ax.tick_params(which='both', labelleft=False)

    handles, names = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               fontsize=9, frameon=False, markerscale=4)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    return fig


def main(data_file=data_file, save_figs=save_figs):
    data_dict = np.load(data_file, allow_pickle=True).item()
    feats, labels = gather(data_dict, FEATURE_KEY)
    if not feats:
        raise RuntimeError(f"no '{FEATURE_KEY}' in the data dict")
    if COLOR_BY == 'polarity' or labels is None:
        if COLOR_BY != 'polarity':
            print("no 'wf_cluster' labels for every session: coloring by polarity")
        labels = None
    missing = [k for k, _, _ in FEATURES if k not in feats]
    if missing:
        print(f'not in {FEATURE_KEY} (re-run collect_waveform_data?): {missing}')
    print(f"{len(feats['polarity'])} cells")
    fig = plot_feature_grid(feats, labels)
    os.makedirs(save_figs, exist_ok=True)
    suffix = '_v2' if FEATURE_KEY.endswith('_v2') else '_v1'
    fig.savefig(f'{save_figs}feature_grid{suffix}.png', dpi=300, bbox_inches='tight')
    return feats, labels


if __name__ == '__main__':
    main()
