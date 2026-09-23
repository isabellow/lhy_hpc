"""
Broken-channel check for a Cambridge Neurotech / Intan recording.

Reads the electrode impedances stored in the Intan header, thresholds them,
and prints both the 0-based indices for the Kilosort4 "Excluded channels"
field and the probe-map names/positions so you can check them against your
notes. Optionally writes the binary for bombcell.
"""

import struct
import numpy as np
import pandas as pd
import spikeinterface.extractors as se

# settings
intan_folder = "C:/Users/Isabel/Documents/data_temp/ROS107_260921/ROS107_260921_120030/"
map_path = "Z:/Isabel/ephys/channel_maps/H15_128Chan_IntanMap.xlsx"
thresh_Mohm = 3.0
save_binary = False


def _qstr(f):
    """Intan QString: uint32 byte length (0xFFFFFFFF = null) + UTF-16LE."""
    (n,) = struct.unpack("<I", f.read(4))
    return "" if n == 0xFFFFFFFF else f.read(n).decode("utf-16-le", "replace")


def read_impedances(rhd_path):
    """Impedances (MOhm) of the enabled amplifier channels, in recording order."""
    with open(rhd_path, "rb") as f:
        if struct.unpack("<I", f.read(4))[0] != 0xC6912702:
            raise ValueError(f"{rhd_path} is not an Intan RHD file")
        version = struct.unpack("<hh", f.read(4))
        f.read(40)                                   # sample rate .. filter settings
        for _ in range(3):
            _qstr(f)                                 # notes
        if version >= (1, 1):
            f.read(2)
        if version >= (1, 3):
            f.read(2)
        if version >= (2, 0):
            _qstr(f)                                 # reference channel

        z = []
        (n_groups,) = struct.unpack("<h", f.read(2))
        for _ in range(n_groups):
            _qstr(f), _qstr(f)                       # group name, prefix
            enabled, n_ch, _ = struct.unpack("<3h", f.read(6))
            if not (enabled and n_ch):
                continue                             # no channel records written
            for _ in range(n_ch):
                _qstr(f), _qstr(f)                   # native, custom name
                fields = struct.unpack("<10h", f.read(20))
                z_ohm, _phase = struct.unpack("<2f", f.read(8))
                if fields[2] == 0 and fields[3]:     # signal_type, channel_enabled
                    z.append(z_ohm / 1e6)
    return np.array(z)


# get impedances from Intan header
z = read_impedances(f"{intan_folder}info.rhd")
assert not np.allclose(z, 0), "no impedance measurement saved in this header"

# Kilosort indices and channel-map names
m = pd.read_excel(map_path).sort_values("Intan Channel")   # -> recording order
tbl = pd.DataFrame({
    "ks_index": np.arange(len(z)),                         # 0-based row of the binary
    "impedance_Mohm": z,
    "ch_name": (m["Shank Letter"].astype(str).str.strip() + "-"
                + m["Shank Row"].astype(int).map("{:02d}".format) + "-"
                + m["Shank Column"].astype(str).str.strip()).to_numpy(),
    "probe_position": m["Shank position"].to_numpy(),      # 1-128 by physical position
})

# threshold by impedance
bad = tbl[tbl.impedance_Mohm > thresh_Mohm].sort_values("probe_position")
print(f"median {np.median(z):.2f} MOhm, threshold {thresh_Mohm} MOhm, "
      f"{len(bad)} of {len(tbl)} excluded\n")
print(bad.to_string(index=False, columns=["probe_position", "ch_name",
                                          "impedance_Mohm", "ks_index"]))
print("\nKilosort4 'Excluded channels':",
      ", ".join(str(i) for i in sorted(bad.ks_index)))

# gain and sampling rate
recording = se.read_intan(f"{intan_folder}info.rhd", stream_id="0")
gains = recording.get_channel_gains()
assert np.allclose(gains, gains[0]), "expected uniform gain across channels"
print(f"fs: {recording.get_sampling_frequency()}, gain_to_uV: {gains[0]}")

# optionally save a binary file for bombcell
if save_binary:
    sort_dir = f"{intan_folder}bombcell/"
    print(f"Binary save path: {sort_dir}traces_cached_seg0.raw")
    recording.save(format="binary", folder=sort_dir, dtype="int16",
                   n_jobs=8, chunk_duration="1s")
