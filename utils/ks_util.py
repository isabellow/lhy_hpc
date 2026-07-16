'''
From RH may not need or may need to modify
'''


import os
import platform
import datetime
from pathlib import Path
import shutil

def get_file_modified_time(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    modified_time = path.stat().st_mtime
    timestamp = datetime.datetime.fromtimestamp(modified_time).strftime("%m%d%y_%H%M%S")
    return timestamp

def get_file_creation_time( filepath ):
    if platform.system() == 'Windows':
        # Windows gives true creation time
        creation_time = os.path.getctime(filepath)
    else:
        # On Unix (Linux/macOS), getctime returns the last metadata change time
        # True creation time is not available in most cases
        stat = os.stat(filepath)
        creation_time = stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime
    return datetime.datetime.fromtimestamp(creation_time)

def get_bird( basename ):
    parts = basename.split( '_' )
    return parts[0]

def basename_from_oe_path( oe_path ):
    str_split = oe_path.parts
    ip = [ i == 'acquisition' for i in str_split ];
    basename = str_split[ np.argwhere( ip )[0][0] + 1 ]
    return basename

def dat_path_oe( oe_path, node='101' ):
    this_node = str( int(node)-1 );
    dat_path = Path.joinpath(oe_path, 'recording1/continuous/OneBox-' + this_node + '.ProbeA/continuous.dat' )
    samples_path =  Path.joinpath(oe_path, 'recording1/continuous/OneBox-' + this_node + '.ProbeA/sample_numbers.npy' )
    timestamps_path = Path.joinpath(oe_path, 'recording1/continuous/OneBox-' + this_node + '.ProbeA/timestamps.npy')
    return [ dat_path, samples_path, timestamps_path ]

def target_dat_path( basepath, basename ):
    dat_path = basepath / Path( basename+'.dat' )
    samples_path = basepath / Path( basename+'_sample_numbers.npy' )
    timestamps_path = basepath / Path( basename+'_timestamps.npy' )
    return [ dat_path, samples_path, timestamps_path ]

def adc_path_oe( oe_path, node='101' ):
    this_node = str( int(node)-1 );
    dat_path = Path.joinpath( oe_path, 'recording1/continuous/OneBox-' + this_node + '.OneBox-ADC/continuous.dat' )
    samples_path = Path.joinpath( oe_path, 'recording1/continuous/OneBox-' + this_node + '.OneBox-ADC/sample_numbers.npy' )
    timestamps_path = Path.joinpath( oe_path, 'recording1/continuous/OneBox-' + this_node + '.OneBox-ADC/timestamps.npy' )
    return [ dat_path, samples_path, timestamps_path ]

def target_adc_path( basepath ):
    dat_path = basepath / 'analogin.dat'
    samples_path = basepath / 'analogin_sample_numbers.npy'
    timestamps_path = basepath / 'analogin_timestamps.npy'
    return [ dat_path, samples_path, timestamps_path ]


def get_experiment_paths( oe_path, node='101' ):
    recn = 'Record Node' + ' ' + node
    temp_path = oe_path / recn
    subdirs = [p for p in temp_path.iterdir() if p.is_dir()]
    return subdirs

def get_date():
    timestamp = datetime.datetime.now().strftime("%m%d%y_%H%M%S")
    return timestamp