import numpy as np
import sys
sys.path.append("../utils/")
from load_matlab_data import loadmat_sbx
import scipy.io
import os

def load_mat_data(path_to_data):
    """
    loads anatomy matlab data struct
    as a numpy array

    Params
    ------
    path_to_data : string
        path to the matlab file

    Returns
    -------
    data : dict
        dict containing cell coords and region labels
    """
    # load data
    d = loadmat_sbx(path_to_data)
    if 'cells' in d:
        data = d['cells']
    else:
        print('did not recognize data format!')
    return data

def save_mat_data(data_folder, bird_id, save_data):
    """
    saves matlab anatomy files and to numpy arrays

    Params
    ------
    data_folder : string
        path to folder with anatomy files
    save_data : string
        path to folder to save the data
    """
    files = os.listdir(data_folder)
    for file in files:
        if bird_id in file: # this is an anatomy file       
            d = load_mat_data(f'{data_folder}{file}')
            np.save(f'{save_data}{file[:-4]}.npy', d)