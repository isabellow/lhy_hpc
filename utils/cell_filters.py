'''
Explicit, opt-in cell filtering for the plotting scripts.

Every criterion is off by default, so a script gets all cells unless it asks
for something.  This replaces the old pattern where filtering was hard-wired
(and, in plot_cache_activity.py, silently commented out) and where a missing
stim key raised a NameError instead of degrading gracefully.

Typical use, near the top of a plotting script:

    CELL_FILTERS = dict(
        use_stim_filter = False,   # keep only cells in the projection nucleus
        fr_thresh       = 0.05,    # Hz; None disables
        cell_type       = 'all',   # 'all' | 'excitatory' | 'inhibitory'
    )
    ...
    keep_cells, filter_report = filter_cells(
        data_dict, bird, session_id, n_cells_raw, **CELL_FILTERS)

`keep_cells` is a boolean mask over the ROWS of aligned_spikes.npy, so the
same mask must be applied to every other per-cell array (avg_firing_rate,
cache_modulated, cell_pos, ...) before any of them are indexed together.
Use `apply_cell_filter` for that.
'''

import numpy as np


VALID_CELL_TYPES = ('all', 'excitatory', 'inhibitory')


def _stim_filter(data_dict, bird, session_id, n_cells):
    '''
    Cells on or beside stim-responsive channels (i.e. in the projection
    nucleus).  Returns (mask, note); mask is None when the session has no
    usable stim data.
    '''
    session_data = data_dict[bird][session_id]

    # everything idx_cells_by_stim needs, checked up front so a missing key
    # produces a readable message instead of a KeyError deep in the call
    required = ['stim_resp_idx_ch', 'nucleus_dvs', 'cell_pos']
    missing = [k for k in required if k not in session_data]
    has_shank = ('shank_A_idx' in session_data) or ('shank_idx' in session_data)
    if not has_shank:
        missing.append('shank_A_idx/shank_idx')
    if missing:
        return None, (f'no stim filtering for {bird}_{session_id}: '
                      f'missing {", ".join(missing)}')

    # imported lazily: sessions without stim data should not need the stim
    # module (or its scipy.signal imports) on the path at all
    from format_chronic_stim import idx_cells_by_stim

    mask = np.asarray(idx_cells_by_stim(data_dict, bird, session_id)).astype(bool)
    if mask.shape[0] != n_cells:
        return None, (f'no stim filtering for {bird}_{session_id}: stim mask '
                      f'has {mask.shape[0]} cells but aligned_spikes has {n_cells}')
    return mask, None


def filter_cells(data_dict, bird, session_id, n_cells,
                 use_stim_filter=False,
                 fr_thresh=None,
                 cell_type='all',
                 verbose=True):
    '''
    Build a boolean mask over cells from a set of opt-in criteria.

    Params
    ------
    data_dict : dict
        the session data dictionary
    bird, session_id : str
        session to filter
    n_cells : int
        number of rows in aligned_spikes.npy for this session; every criterion
        is checked against this so a stale/mismatched array is caught early
    use_stim_filter : bool
        keep only cells on or beside stim-responsive channels.  Sessions with
        no stim data are reported and left unfiltered rather than skipped, so
        this can stay False for the LHy recordings without editing anything.
    fr_thresh : float or None
        keep only cells with session-average firing rate above this (Hz).
        None disables the criterion.
    cell_type : {'all', 'excitatory', 'inhibitory'}
        keep only cells of this waveform class.
    verbose : bool
        print a one-line summary per criterion

    Returns
    -------
    keep : bool array, shape (n_cells,)
    report : dict
        'n_in', 'n_out', per-criterion counts, and 'notes' (list of strings
        describing any criterion that could not be applied)
    '''
    if cell_type not in VALID_CELL_TYPES:
        raise ValueError(f'cell_type must be one of {VALID_CELL_TYPES}, '
                         f'got {cell_type!r}')

    session_data = data_dict[bird][session_id]
    keep = np.ones(n_cells, dtype=bool)
    report = {'n_in': int(n_cells), 'notes': [], 'applied': {}}

    # ── firing rate ──────────────────────────────────────────────────────
    if fr_thresh is not None:
        if 'waveform_props' in session_data:
            # waveform_props is (3, n_cells): [asymmetry, width, log10 rate]
            log_fr = np.asarray(session_data['waveform_props'])[2]
            if log_fr.shape[0] == n_cells:
                fr_mask = (10.0 ** log_fr) > fr_thresh
                keep &= fr_mask
                report['applied']['fr_thresh'] = int(np.sum(~fr_mask))
            else:
                report['notes'].append(
                    f'no firing-rate filtering: waveform_props has '
                    f'{log_fr.shape[0]} cells but aligned_spikes has {n_cells}')
        else:
            report['notes'].append('no firing-rate filtering: no waveform_props')

    # ── excitatory / inhibitory ──────────────────────────────────────────
    if cell_type != 'all':
        key = f'{cell_type}_idx'
        if key in session_data:
            type_mask = np.asarray(session_data[key]).astype(bool)
            if type_mask.shape[0] == n_cells:
                keep &= type_mask
                report['applied'][cell_type] = int(np.sum(~type_mask))
            else:
                report['notes'].append(
                    f'no {cell_type} filtering: {key} has '
                    f'{type_mask.shape[0]} cells but aligned_spikes has {n_cells}')
        else:
            report['notes'].append(f'no {cell_type} filtering: no {key}')

    # ── stim responsiveness ──────────────────────────────────────────────
    if use_stim_filter:
        stim_mask, note = _stim_filter(data_dict, bird, session_id, n_cells)
        if stim_mask is None:
            report['notes'].append(note)
        else:
            keep &= stim_mask
            report['applied']['stim'] = int(np.sum(~stim_mask))

    report['n_out'] = int(np.sum(keep))

    if verbose:
        for note in report['notes']:
            print(f'  warning! {note}')
        if report['applied']:
            dropped = ', '.join(f'{k}: -{v}' for k, v in report['applied'].items())
            print(f'  cell filters ({dropped}) -> '
                  f'{report["n_out"]}/{report["n_in"]} cells kept')
        else:
            print(f'  no cell filters applied -> all {report["n_in"]} cells kept')

    return keep, report


def apply_cell_filter(keep, **arrays):
    '''
    Index a set of per-cell arrays with the same mask.

    Guards against the failure mode in the old plot_feeder_responses.py, where
    spike_fr was filtered but avg_firing_rate was not, so every baseline line
    after the first dropped cell belonged to the wrong neuron.

    Returns a dict with the same keys, each array indexed along axis 0.
    Any value that is None is passed through untouched.
    '''
    keep = np.asarray(keep).astype(bool)
    out = {}
    for name, arr in arrays.items():
        if arr is None:
            out[name] = None
            continue
        arr = np.asarray(arr)
        if arr.shape[0] != keep.shape[0]:
            raise ValueError(
                f"can't apply cell filter to {name!r}: it has {arr.shape[0]} "
                f'rows but the mask covers {keep.shape[0]} cells')
        out[name] = arr[keep]
    return out
