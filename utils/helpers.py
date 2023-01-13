import numpy as np

def nan_interp(y):
    def find(x):
        return x.nonzero()[0]
    nans = np.isnan(y)
    y[nans] = np.interp(find(nans),find(~nans),y[~nans])
    return y