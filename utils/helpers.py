import numpy as np
from scipy.signal import butter, bessel, filtfilt
from scipy.ndimage import uniform_filter1d

def nan_interp(y):
    def find(x):
        return x.nonzero()[0]
    nans = np.isnan(y)
    y[nans] = np.interp(find(nans),find(~nans),y[~nans])
    return y


''' Filtering '''
def bandpass(x, lowcut, highcut, fs, order=5, axis=-1, kind='butter'):
    """
    Modified slightly from AHW.

    Parameters
    ----------
    x : ndarray
        1d time series data
    lowcut : float
        Defines lower frequency cutoff (e.g. in Hz)
    highcut : float
        Defines upper frequency cutoff (e.g. in Hz)
    fs : float
        Sampling frequency (e.g. in Hz)
    order : int
        Filter order parameter
    kind : str
        Specifies the kind of filter
        butter for butterworth; bessel for bessel
    axis : int
        Axis along which to bandpass filter data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if kind == "butter":
        b, a = butter(order, [low, high], btype="band")
    elif kind == "bessel":
        b, a = bessel(order, [low, high], btype="bandpass")
    else:
        raise ValueError("Filter kind not recognized.")
    return filtfilt(b, a, x, axis=axis)

def highpass(x, highcut, fs, order=5, axis=-1, kind='butter'):
    """
    Modified slightly from AHW.

    Parameters
    ----------
    x : ndarray
        1d time series data
    highcut : float
        Defines upper frequency cutoff (e.g. in Hz)
    fs : float
        Sampling frequency (e.g. in Hz)
    order : int
        Filter order parameter
    kind : str
        Specifies the kind of filter
        butter for butterworth; bessel for bessel
    axis : int
        Axis along which to bandpass filter data
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    if kind == "butter":
        b, a = butter(order, high, btype="high")
    elif kind == "bessel":
        b, a = bessel(order, high, btype="highpass")
    else:
        raise ValueError("Filter kind not recognized.")
    return filtfilt(b, a, x, axis=axis)


def moving_avg(x, fs=50, window=30, mode="reflect"):
    '''
    Get a moving average over the data stream x
    collected at fs Hz with a window-length time window.

    Params
    ------
    x : ndarray, shape (n_frames,)
        time series data, should be a consistent framerate and dtype float
    fs : int
        frames per second, default is 50 Hz
    window : float
        time window for moving average in minutes, default is 30 mins
    mode : string
        passed to scipy.ndimage.uniform_filter1d to define edge behavior
        default "reflect" mirrors the signal at the edges so that each
        avg value contains 30 mins worth of data
    '''
    x = np.asarray(x, dtype=float)
    window_frames = int(round(fs * window * 60))
    return uniform_filter1d(x, size=window_frames, mode=mode, origin=0)