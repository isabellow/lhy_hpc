'''
Write the session list for runSessionWaveformsV2.m.

The data dict is a Python pickle MATLAB can't read, so this resolves each
session's paths the same way collect_waveform_data does and writes them to a
csv: bird, session_id, dat_file, ks_dir, fs. Then, in MATLAB:

    runSessionWaveformsV2('Z:/Isabel/data/lhy_implants/waveform_v2_sessions.csv')

Sessions are skipped (and listed) if they have no ephys, no waveformStruct.mat,
or no raw data file that can be found.
'''
import ast
import csv
import os
import re

import numpy as np

ROOT_DIR = "Z:/Isabel/data/lhy_implants/"
DATA_FILE = f"{ROOT_DIR}good_session_data.npy"
OUT_CSV = f"{ROOT_DIR}waveform_v2_sessions.csv"
FS_DEFAULT = 30000


def read_ks_params(ks_path):
    ''' Kilosort's params.py as a dict (dat_path, n_channels_dat, sample_rate, ...) '''
    params = {}
    fn = f"{ks_path}params.py"
    if not os.path.isfile(fn):
        return params
    with open(fn) as fh:
        for line in fh:
            m = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", line)
            if not m:
                continue
            try:
                params[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                params[m.group(1)] = m.group(2)
    return params


def find_dat_file(ephys_dir, params):
    '''
    amplifier.dat in the ephys folder (what getSessionWaveforms read), else the
    file Kilosort was pointed at in params.py
    '''
    candidate = f"{ephys_dir}amplifier.dat"
    if os.path.isfile(candidate):
        return candidate
    dat = params.get('dat_path')
    if isinstance(dat, (list, tuple)):
        dat = dat[0] if len(dat) == 1 else None    # multiple files: can't use one memmap
    if isinstance(dat, str) and os.path.isfile(dat):
        return dat.replace('\\', '/')
    return None


def session_rows(data_dict, root_dir):
    rows, skipped = [], []
    for bird, bird_data in data_dict.items():
        if not isinstance(bird_data, dict) or 'all_sessions' not in bird_data:
            continue
        for session_id in bird_data['all_sessions']:
            session_data = bird_data[session_id]
            name = f"{bird}_{session_id}"
            if 'ephys' not in session_data.get('preprocessed_data', []):
                continue

            # same path rules as collect_waveform_data
            session_dir = f"{root_dir}{bird}/{bird}_{session_id}/"
            ephys_id = session_data['ephys_id']
            ks_path = f"{session_dir}{bird}_{ephys_id}/{session_data['ks_folder']}/"
            if 'lhy' in root_dir:
                ephys_dir = f"{session_dir}{bird}_{ephys_id}/"
            else:
                ephys_dir = f"{session_dir}{bird}_{ephys_id}/raw_ephys_output/"

            if not os.path.isfile(f"{ks_path}waveformStruct.mat"):
                skipped.append((name, 'no waveformStruct.mat'))
                continue
            params = read_ks_params(ks_path)
            dat_file = find_dat_file(ephys_dir, params)
            if dat_file is None:
                skipped.append((name, f'no amplifier.dat in {ephys_dir} or usable dat_path in params.py'))
                continue
            fs = params.get('sample_rate', FS_DEFAULT)
            rows.append(dict(bird=bird, session_id=str(session_id), dat_file=dat_file,
                             ks_dir=ks_path, fs=f"{float(fs):g}"))
    return rows, skipped


def main(data_file=DATA_FILE, root_dir=ROOT_DIR, out_csv=OUT_CSV):
    data_dict = np.load(data_file, allow_pickle=True).item()
    rows, skipped = session_rows(data_dict, root_dir)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['bird', 'session_id', 'dat_file', 'ks_dir', 'fs'])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sessions written to {out_csv}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    return rows, skipped


if __name__ == '__main__':
    main()
