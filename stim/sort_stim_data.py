import numpy as np
import os 
import sys

from format_chronic_stim import load_all_stim_times
sys.path.append("..//neural")
from format_waveform_data import map_contacts_to_intan
sys.path.append("..//utils")
import load_matlab_data

import spikeinterface as si # core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.exporters as sexp
from probeinterface import get_probe

import dartsort

import probeinterface.plotting as prb_plotting
import dartsort.vis as dartvis
import matplotlib.pyplot as plt

'''
Given a recording with antidromic stimulation:
learn waveform templates on spontaneous spiking data and 
match those templates to stimulation-evoked responses.

Uses SpikeInterface to format and preprocess the recording file and separate
the stimulation events and antidromic hash from the rest of the recording.

Then, uses DARTsort to first sort and learn templates on the spontaneous
spiking data, then match those templates to stimulation periods.
'''
show_plots = True

''' Set file paths '''
root_dir = "C:/Users/Isabel/Documents/data_temp/"

# session params
bird_id = "SLV132"
session_id = "250310"
ephys_id = "SLV132_250310_122522"

# path to .rhd intan file
intan_folder = f"{root_dir}{bird_id}_{session_id}/{ephys_id}/"

# path to various info files
map_file_path = 'Z:/Isabel/ephys/SILICON PROBE MAP H10_spikesort.xlsx'
stim_data_path = f"{intan_folder}raw_ephys_output/"

# output folders for dartsort and phy
output_folder = f"{intan_folder}dartsort/"
if os.path.isdir(output_folder):
    print('output folder exists')
else:
    os.mkdir(output_folder)

''' Load the recording file '''
recording = se.read_intan(f"{intan_folder}info.rhd", stream_id='0')

''' Add the probe map and check channel layout '''
# load the probe layout
probe = get_probe(manufacturer="cambridgeneurotech", probe_name="ASSY-236-H10")

# map to intan
contact_sort, ch_names = map_contacts_to_intan(probe, map_file_path)
probe = probe.get_slice(contact_sort)

# visualize to check
if show_plots:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    prb_plotting.plot_probe(probe, ax=ax)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], 400)
    for pos, ch_name in zip(probe.contact_positions, ch_names):
        ax.text(pos[0], pos[1], ch_name)

# set the channel indices in the probe object
n_channels = contact_sort.shape[0]
probe.set_device_channel_indices(np.arange(n_channels))

# add the probe map and channel names
recording = recording.set_probe(probe)
recording.set_property("channel_name", ch_names)
recording.set_property("group", shank_idx)

# split the recording by shank
recordings_by_shank = recording.split_by("group")

''' Preprocess similarly to kilosort4 '''
recordings_by_shank[0] = ks4_preprocess(recordings_by_shank[0])
recordings_by_shank[1] = ks4_preprocess(recordings_by_shank[1])

''' Separate stim/hash from spontaneous spiking '''
# load stim times
stim_t_samples = load_all_stim_times(stim_data_path)

# define the stim + hash window
pre_stim_samp = int(5e-3 * fs) # seconds * sampling rate
post_stim_samp = int(25e-3 * fs)

# separate the recordings
recordings_A, maps_A = split_by_stim(recordings_by_shank[0], stim_t=stim_t_samples,
                                     pre_stim=pre_stim_samp, post_stim=post_stim_samp)
recordings_B, maps_B = split_by_stim(recordings_by_shank[1], stim_t=stim_t_samples,
                                     pre_stim=pre_stim_samp, post_stim=post_stim_samp)

''' Save as binary files for DARTsort '''
recording_spont_bin_A = recordings_A['spont'].save(format="binary", 
                                                    folder=f"{output_folder}spont_binary_A")
recording_stim_bin_A = recordings_A['stim'].save(format="binary",
                                                    folder=f"{output_folder}stim_binary_A")
recording_spont_bin_B = recordings_B['spont'].save(format="binary",
                                                    folder=f"{output_folder}spont_binary_B")
recording_stim_bin_B  = recordings_B['stim'].save(format="binary",
                                                    folder=f"{output_folder}stim_binary_B")

''' Sort the data without stim/hash '''
sorting_spont_A = dartsort.dartsort(recording_spont_bin_A, f"{output_folder}A_shank/")
sorting_spont_B = dartsort.dartsort(recording_spont_bin_B, f"{output_folder}B_shank/")

''' Estimate the templates '''
template_data_A = dartsort.estimate_template_library(recording=recording_spont_bin_A, sorting=sorting_spont_A)
template_data_B = dartsort.estimate_template_library(recording=recording_spont_bin_B, sorting=sorting_spont_B)

''' Visualize the sorting results - TODO '''

''' Find the templates in the stim period '''
# match the spontaneous templates to the stim periods
matching_stim_A = dartsort.match(recording=recording_stim_bin_A, template_data=template_data_A)
matching_stim_B = dartsort.match(recording=recording_stim_bin_B, template_data=template_data_B)

# evaluate the matching results - TODO
''' 
check spike times/raw traces
matching_scores
matching_probs
amplitudes
residual_norms
'''
# scores = matching_stim.get_property("match_score")