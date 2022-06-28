import numpy as np
import sys
sys.path.append("../utils/")
from load_matlab_data import loadmat_sbx
import scipy.io
import os


def load_ma_data(path_to_data):
    """
    loads Marissa's anatomy matlab data struct
    as a numpy array

    Params
    ------
    path_to_data : string
        path to the matlab file

    Returns
    -------
    data : dict
        dict containing behavioral and spiking data
        data['sp'] gives dict of spiking data

    """
    # load data
    d = loadmat_sbx(path_to_data)
    if 'aligned' in d:
        data = d['aligned']
    else:
        print('did not recognize data format!')

    return data


def save_ma_data(data_folder, save_data):
    """
    finds Marissa's anatomy files and saves to numpy arrays

    Params
    ------
    data_folder : string
        path to folder with many anatomy files
    save_data : string
        path to folder to save the data
    """
    files = os.listdir(data_folder)
    for file in files:
        if 'MC' in file: # this is an anatomy file
            file_info = file.split('_')
            if len(file_info) > 2: # has hemisphere label
                bird_id = f'{file_info[0]}_{file_info[1]}'
            else:
                bird_id = file_info[0]           
            d = load_ma_data(f'{data_folder}{file}')
            np.save(f'{save_data}{bird_id}.npy', d)


def load_ma_npy(save_data, \
                filter_brain_region=True, region_id=3,\
                filter_ML=True):
    """
    loads Marissa's data from npy files saved above
    
    Params
    ------
    save_data : string
        path to folder with npy files
    filter_brain_region : bool
        filter data by brain region
        default is True
    region_id : int
        which brain region to keep
        default is LHY (3)
    filter_ML : bool
        whether to remove contralateral cells
        default is True

    Return
    ------
    birds : list of strings
        bird IDs
    data : dict
        holds the numpy array for each bird
    n_cells : ndarray
        number of cells in the designated brain region
        for each bird
    """
    data = {}
    birds = []

    files = os.listdir(save_data)
    for file in files:
        file_info = file.split('.')
        bird_id = file_info[0]
                
        # load numpy array and store in dict
        birds.append(bird_id)
        data[bird_id] = np.load(f'{save_data}{file}')

    # count the number of birds pre-filtering
    n_birds = len(birds)
    print(f'{n_birds} total birds\n')

    # count the number of cells in the LHY
    n_cells = np.zeros(n_birds)
    birds_new = []
    for i, b in enumerate(birds):
        d = data[b]

        if filter_brain_region & filter_ML:
            lhy_idx = (d[:, 3] == region_id) & (d[:, 0] > 0)
        else:
            lhy_idx = (d[:, 3] == region_id)

        n_cells[i] = np.sum(lhy_idx)
        print(f'{b}: {n_cells[i]} LHY cells')
        
        if filter_brain_region:
            if n_cells[i]:
                birds_new.append(b)
        else:
            birds_new.append(b) 
    
    birds = birds_new
    n_birds = len(birds)
    if filter_brain_region:
        n_cells = n_cells[n_cells > 0]
        print(f'\n{n_birds} birds have LHY labeling')

    return birds, data, n_cells