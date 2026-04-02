'''
Given a session ID (bird_date) run the data through the analysis pipeline
'''
''' NEURAL PROPERTIES '''
# Check the waveform properties, cumulative firing rates, and stim response
# print: n/N cells are excitatory (%%)
# show plots: waveform params, cumulative FR
# show/save plots: median stim resp by channel

# user text entry: continue analyzing session? y/n
# if y...


''' COLLECT SESSION DATA '''
'''
params to collect:
- waveform data: width, asymmetry, log firing rates
- broken channels
- feeder open/close times
'''


''' BEHAVIOR '''
# caching/retrieving
# checks/visits
# feeders

# print: N total caches, retrievals, checks; N visits to each feeder (open/closed)

# show plots:
# cache, retrieval, eating bout durations
# occupancy map by perch

# show/save plots: 
# N seeds in arena, caches/retrievals over time, checks/visits over time

# user text entry: continue analyzing session? y/n
# if y...


''' COLLISION DETECTION '''



''' NEURAL ACTIVITY X BEHAVIOR'''
# all cells:
# ethogram with firing rates (Stringer sorting? zoom/scrollable?)
# barcode analysis
# video with firing rates (SC)

# user input: plot all cells? y/n

    # if n...
    # user input: select cells to plot/save?

    # if y...
    # user input: cell IDs?
    # [convert to index]

# per cell (all or selected):
# cache/retrieval/eating aligned activity
# occupied/empty checks and visits
# place coding (by perch, classic map)
# feeder-aligned raster and psth
# video with spikes for selected cells (DA)