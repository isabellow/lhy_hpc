import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import scipy.io
import os
import h5py

''' for loading newer matlab data - from ChatGPT '''
def loadmat_v73(filename):
    def unpack(obj, f):
        if isinstance(obj, h5py.Group):
            return {key: unpack(obj[key], f) for key in obj.keys()}

        data = obj[()]

        # Resolve MATLAB object references, common for cells/struct arrays
        if data.dtype == h5py.ref_dtype:
            return np.array([unpack(f[ref], f) for ref in data.flat], dtype=object).reshape(data.shape)

        # Decode character arrays
        if data.dtype.kind == "u" and obj.attrs.get("MATLAB_class") == b"char":
            return "".join(chr(c) for c in data.flat)

        # Convert bytes to strings
        if data.dtype.kind in {"S", "U"}:
            return data.astype(str)

        # Transpose numeric arrays to match MATLAB orientation
        if isinstance(data, np.ndarray):
            data = np.squeeze(data).T
        
        return data

    with h5py.File(filename, "r") as f:
        return {key: unpack(f[key], f) for key in f.keys()}

""" scripts for loading Matlab structs - from Mark Plitt """

def loadmat_sbx(filename):
    """
    this function should be called instead of direct spio.loadmat

    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    print(filename)
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data_)


def _check_keys(dict):
    """
    checks if entries in dictionary rare mat-objects. If yes todict is called
    to change them to nested dictionaries
    """

    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_ca_mat(fname):
    """load results from cnmf"""

    ca_dat = {}
    try:
        with h5py.File(fname, 'r') as f:
            for k, v in f.items():
                try:
                    ca_dat[k] = np.array(v)
                except:
                    print(k + "not made into numpy array")
                    ca_dat[k] = v
    except:
        ca_dat = scipy.io.loadmat(fname)
        for key in ca_dat.keys():
            if isinstance(ca_dat[key], np.ndarray):
                ca_dat[key] = ca_dat[key].T
    return ca_dat