import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

import os 
import sys
sys.path.append("..//behavior/")
from format_behavior_data import dist_binned_mean_sem

'''
Analysis as in Chettih, Mackevicius et al, 2024 Fig. 5

Compare barcode-barcode correlations across different delta caches
Compare barcode-retrieval correlations across different delta retrievals

Compute the barcode by subtracting the smooth spatial component
'''