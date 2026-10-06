import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator, ScalarFormatter

import os
import sys
sys.path.append("..//utils/")
sys.path.append("..//stim/")
import waveform_analysis

'''
Classify units by waveform polarity, then cluster the trough-dominant units
in (firing rate, waveform width, spread across the probe).

Needs data_dict[bird][session]['wf_features'] (original waveFormsMean) and,
for FEATURE_KEY = 'wf_features_v2', ['wf_features_v2'] (waveFormsMean_v2:
high-pass only, re-aligned), both from build_data_dict.collect_waveform_data.

Figures (suffix _v1 or _v2 from FEATURE_KEY, so both versions can be kept)
-------
waveform_polarity_diagnostics.png
    legacy width vs. peak/trough ratio (the ~0.5 ms pile-up should sit in
    the ambiguous band if it comes from polarity flips), legacy vs. new
    width, and where the dominant extremum sits relative to the KS spike time.
waveform_props_clusters.png
    3D scatter plus the three 2D projections, positive and ambiguous units
    drawn first, trough-dominant units coloured by cluster.
'''
''' File Paths '''
root_dir = "Z:/Isabel/data/lhy_implants/"
save_figs = f"../figures/basic_neural_analysis/"
data_file = f"{root_dir}good_session_data.npy"

''' Params '''
# which waveform version to plot: 'wf_features' (original waveFormsMean) or
# 'wf_features_v2' (high-pass only, re-aligned)
FEATURE_KEY = 'wf_features_v2'
FIG_SUFFIX = '_v2' if FEATURE_KEY.endswith('_v2') else '_v1'

# the three features to cluster on and plot: any per-unit key of the features
# dict. FEATURES[0] is the vertical axis of the 3D plot and of the first two
# 2D panels. Examples:
#   ['stable_rate', 'width_ms', 'weighted_dist_um']   rate, width, spread
#   ['stable_rate', 'width_ms', 'asymmetry']          like the original plot
FEATURES = ['pk_trough_ratio', 'width_ms', 'asymmetry']

# label, log10 scale, fixed limits (in real units) for each feature. Keys not
# listed here are plotted on a linear scale with automatic limits.
FEATURE_INFO = {
    'stable_rate':      ('firing rate (Hz)', True, (0.01, 300)),
    'mean_rate':        ('firing rate (Hz)', True, (0.01, 300)),
    'width_ms':         ('width (ms)', False, (0, 1.1)),
    'half_width_ms':    ('half-width (ms)', False, (0, 0.6)),
    'asymmetry':        ('spike asymmetry', False, (-1, 1.02)),
    'decay_um':         ('spread, decay length (um)', True, (2, 300)),
    'weighted_dist_um': ('spread, weighted distance (um)', True, (2, 300)),
    'extent_um':        ('spread, extent (um)', False, (0, 300)),
    'n_ch_above':       ('channels above threshold', False, (0, 30)),
    'pk_trough_ratio':  ('peak / |trough|', True, (0.05, 20)),
}
# fixed axis limits from FEATURE_INFO, so the v1 and v2 figures line up side
# by side (False = automatic). Units outside are not drawn; main() says how many.
USE_FIXED_LIMITS = True

N_CLUSTERS = 2
METHOD = 'gmm'               # or 'kmeans'
MIN_FRAC_STABLE = 0.05       # drop units well recorded for < 5% of the session

COLORS = {'positive': 'xkcd:saffron', 'ambiguous': 'xkcd:dark gray',
          'unclustered': 'xkcd:light gray',
          'clusters': ['xkcd:cobalt blue', 'xkcd:scarlet', 'xkcd:grass green', 'xkcd:purple']}


def gather_features(data_dict, feature_key='wf_features'):
    '''pool feature_key across birds and sessions; returns (feats, source)'''
    pooled, source = {}, []
    for bird in data_dict.keys():
        for session_id in data_dict[bird].get('all_sessions', []):
            f = data_dict[bird][session_id].get(feature_key)
            if f is None:
                continue
            n = len(f['polarity'])
            for k, v in f.items():
                if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
                    pooled.setdefault(k, []).append(v)
            source += [(bird, session_id)] * n
    feats = {k: np.concatenate(v) for k, v in pooled.items()}
    return feats, source


def _groups(feats):
    pol = feats['polarity']
    g = np.full(pol.shape, 'unclustered', dtype=object)
    g[pol == 1] = 'positive'
    g[pol == 0] = 'ambiguous'
    return g


def plot_diagnostics(feats, ambiguous=(0.8, 1.25)):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    r = feats['pk_trough_ratio']
    ax = axs[0]
    ax.axvspan(*ambiguous, color='xkcd:light gray', zorder=0)
    ax.scatter(r, feats['legacy_width_ms'], s=6, c='k', alpha=0.5)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ''))
    ax.set_xlabel('peak / |trough| near spike time')
    ax.set_ylabel('legacy width (ms)')
    ax.set_title('pile-up should sit in the grey band')

    ax = axs[1]
    ax.scatter(feats['legacy_width_ms'], feats['width_ms'], s=6,
               c=feats['polarity'], cmap='coolwarm', vmin=-1, vmax=1)
    lim = [0, np.nanmax(feats['legacy_width_ms'])]
    ax.plot(lim, lim, 'k:', lw=0.8)
    ax.set_xlabel('legacy width (ms)')
    ax.set_ylabel('new width (ms)')
    ax.set_title(f"{np.sum(np.isnan(feats['width_ms']))} units: no opposite lobe")

    ax = axs[2]
    bins = np.arange(-0.5, 0.51, 0.02)
    off = feats['main_offset_ms']
    for p, lab, c in [(-1, 'trough-dominant', 'xkcd:cobalt blue'),
                      (0, 'ambiguous', 'xkcd:gray'), (1, 'peak-dominant', 'orange')]:
        m = (feats['polarity'] == p) & np.isfinite(off)
        ax.hist(off[m], bins=bins, color=c, alpha=0.7, label=lab)
    ax.set_xlabel('dominant extremum - KS spike time (ms)')
    ax.set_ylabel('units')
    ax.set_title('spread within a group: check alignment')
    ax.legend(fontsize=7, frameon=False)
    for a in axs:
        for s in ['top', 'right']:
            a.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def feature_info(key):
    '''(label, log10, limits) for a feature key; linear and automatic if unlisted'''
    return FEATURE_INFO.get(key, (key, False, None))


def feature_limits(keys):
    '''per-feature axis limits in the units of res['X'] (log10 where used), or None'''
    lims = []
    for key in keys:
        _, lg, lim = feature_info(key)
        if not USE_FIXED_LIMITS or lim is None:
            lims.append(None)
        else:
            lims.append(tuple(np.log10(lim)) if lg else tuple(lim))
    return lims


def plot_clusters(feats, res):
    X, labels, ok = res['X'], res['labels'], res['clustered']
    keys, logs = res['keys'], res['log']
    names = [feature_info(k)[0] for k in keys]
    lims = feature_limits(keys)
    groups = _groups(feats)

    fig = plt.figure(figsize=(18, 4.4))
    # narrow empty column after the 3D axes keeps its vertical-axis label
    # clear of the first 2D panel's y-label
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 0.15, 1, 1, 1], wspace=0.45)
    ax3 = fig.add_subplot(gs[0], projection='3d')
    axs = [fig.add_subplot(gs[i]) for i in (2, 3, 4)]
    pairs = [(1, 0), (2, 0), (1, 2)]          # (x, y) feature index per 2D panel

    def draw(ax, mask, color, label, pair=None, s=8, alpha=0.8):
        if not mask.any():
            return
        if pair is None:
            ax.scatter(X[mask, 1], X[mask, 2], X[mask, 0], s=s, c=color, lw=0,
                       alpha=alpha, label=label, depthshade=False)
        else:
            ax.scatter(X[mask, pair[0]], X[mask, pair[1]], s=s, c=color, lw=0,
                       alpha=alpha, label=label)

    fin = np.all(np.isfinite(X), axis=1)
    layers = [(fin & (groups == 'positive'), COLORS['positive'], 'positive'),
              (fin & (groups == 'ambiguous'), COLORS['ambiguous'], 'ambiguous'),
              (fin & (groups == 'unclustered') & ~ok, COLORS['unclustered'], 'excluded')]
    for k in range(labels.max() + 1):
        layers.append((labels == k, COLORS['clusters'][k % 4], f'cluster {k} (n={np.sum(labels == k)})'))

    for m, c, lab in layers:
        draw(ax3, m, c, lab)
        for ax, pr in zip(axs, pairs):
            draw(ax, m, c, lab, pair=pr)

    # limits first, so tick placement can use them
    setters3 = {1: ax3.set_xlim, 2: ax3.set_ylim, 0: ax3.set_zlim}
    for d, lim in enumerate(lims):
        if lim is not None:
            setters3[d](lim)
    for ax, (dx, dy) in zip(axs, pairs):
        if lims[dx] is not None:
            ax.set_xlim(lims[dx])
        if lims[dy] is not None:
            ax.set_ylim(lims[dy])

    # log-scaled features are stored as log10: put ticks at 1-2-5 values
    def fmt(axis, dim, label):
        if logs[dim] and (lims[dim] is not None or fin.any()):
            lo, hi = lims[dim] if lims[dim] is not None else (
                np.min(X[fin, dim]), np.max(X[fin, dim]))
            cand = np.array([m * 10.0 ** e for e in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1)
                             for m in (1, 2, 5)])
            cand = cand[(np.log10(cand) >= lo - 1e-9) & (np.log10(cand) <= hi + 1e-9)]
            if cand.size > 6:                    # wide range: decades only
                cand = cand[np.isclose(np.log10(cand) % 1, 0)]
            axis.set_ticks(np.log10(cand))
            axis.set_ticklabels([f'{v:g}' for v in cand])
        axis.set_label_text(label)

    fmt(ax3.xaxis, 1, names[1]); fmt(ax3.yaxis, 2, names[2]); fmt(ax3.zaxis, 0, names[0])
    for ax, (dx, dy) in zip(axs, pairs):
        fmt(ax.xaxis, dx, names[dx]); fmt(ax.yaxis, dy, names[dy])
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

    # legend outside the rightmost panel, aligned with its top edge
    axs[-1].legend(fontsize=7, frameon=False, loc='upper left',
                   bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    return fig


def main(data_file=data_file, save_figs=save_figs):
    data_dict = np.load(data_file, allow_pickle=True).item()
    feats, source = gather_features(data_dict, FEATURE_KEY)
    if not feats:
        raise RuntimeError(f"no '{FEATURE_KEY}' in the data dict: re-run "
                           "build_data_dict.collect_waveform_data(overwrite=True)")
    print(f'\nplotting {FEATURE_KEY}')
    n = len(feats['polarity'])
    pol = feats['polarity']
    print(f'\n{n} units: {np.sum(pol == -1)} trough-dominant, '
          f'{np.sum(pol == 1)} peak-dominant, {np.sum(pol == 0)} ambiguous')

    res = waveform_analysis.cluster_negative_units(
        feats, features=[(k, feature_info(k)[1]) for k in FEATURES],
        n_clusters=N_CLUSTERS, method=METHOD, min_frac_stable=MIN_FRAC_STABLE)
    for key, n_bad in res['n_nonpos'].items():
        if n_bad:
            print(f'{n_bad} trough-dominant units have {key} <= 0 and cannot be log-scaled: '
                  f'left out of the clustering')
    if res['bic']:
        best = min(res['bic'], key=res['bic'].get)
        print('GMM BIC by k: ' + ', '.join(f'{k}: {v:.0f}' for k, v in res['bic'].items())
              + f'  (lowest at k={best})')
    for k in range(res['labels'].max() + 1):
        m = res['labels'] == k
        desc = []
        for d, (key, lg) in enumerate(zip(res['keys'], res['log'])):
            med = np.median(res['X'][m, d])
            desc.append(f'{key} {10 ** med:.3g}' if lg else f'{key} {med:.3g}')
        print(f'cluster {k}: n={m.sum()}, median ' + ', '.join(desc))

    lims = feature_limits(res['keys'])
    X = res['X']
    fin = np.all(np.isfinite(X), axis=1)
    inside = np.ones(len(X), bool)
    for d, lim in enumerate(lims):
        if lim is not None:
            inside &= (X[:, d] >= lim[0]) & (X[:, d] <= lim[1])
    n_out = int(np.sum(fin & ~inside))
    if n_out:
        print(f'{n_out} units fall outside the fixed limits and are not drawn')

    os.makedirs(save_figs, exist_ok=True)
    fig = plot_diagnostics(feats)
    fig.savefig(f'{save_figs}waveform_polarity_diagnostics{FIG_SUFFIX}.png', dpi=300, bbox_inches='tight')
    fig = plot_clusters(feats, res)
    fig.savefig(f'{save_figs}waveform_props_clusters{FIG_SUFFIX}.png', dpi=600, bbox_inches='tight')
    return feats, res, source


if __name__ == '__main__':
    main()
