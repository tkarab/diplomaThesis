import os.path
import numpy as np
import constants
import plot_functions
import scipy
import time

"""
    For segmenting sEMG signal using sliding window of given shape and size
    We assume that sEMG signal is given in form of a numpy array with dimensions (time_length x channels) i.e. (500x12)
    It returns a 1-D numeric array with all the starting indices of the segments.
    
    i.e. if element slice_start_indices[i]=124 and window size = 15, the segment should be:
        emg[slice_start_indices[i] : slice_start_indices[i] + window_size][:]
    ->  emg[124:139][:]
    
    PARAMETERS
    x : np.ndarray -> the emg signal
    window_size : int -> the window size
    window_step : int -> the window step
    
    RETURNS
    slice_start_indices : np.ndarray -> array of length N where N is the number of segments, which contains
                                        the starting indices of all segments
"""
def get_segmentation_indices(x:np.ndarray, window_size:int, window_step:int):
    slice_start_indices = np.arange(0,len(x)-window_size+1,window_step)
    return slice_start_indices

"""
    For subsampling an sEMG signal, from an initial frequency to a new one.
    We suppose that x is given as (time_samples x channels)
    
    PARAMETERS
    x   ->  the sEMG signal
    init_freq   ->  the initial frequency
    new_freq    ->  the new frequency of the signal
    
    RETURNS
    the subsampled signal
"""
def subsample(x: np.ndarray, init_freq:float, new_freq:float):
    sub_factor = int(init_freq/new_freq)
    indices = np.arange(0,len(x),sub_factor)
    return np.take(x,indices=indices,axis=0)

"""
    DESCRIPTION
    Applies a low-pass butterworth filter (usually 1Hz) to the emg    
    
    PARAMETERS
    x : semg signal
    Fc : Cutoff frequency
    Fs : Sampling frequency
    N : Filter order
    
    RETURNS
    filtered and rectified signal (by rectified we mean its absolute value)
"""
def applyLPFilter(x,Fc=1,Fs=100,N=1):
    f = 2*Fc/Fs
    x = np.abs(x)
    b, a = scipy.signal.butter(N=N, Wn=f, btype='low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output

def applyLPFilter2(emg,Fc=1,Fs=100,N=1):
    f_sos = scipy.signal.butter(N=1, Wn=2 * Fc / Fs, btype='low', output='sos')
    return scipy.signal.sosfilt(f_sos, emg,axis=0)


def rmsRect(emg:np.ndarray, fs = 2000, win_size_ms=200):
    emg_rect = np.zeros(emg.shape)
    W = int(win_size_ms*fs/1000)
    npad = np.floor(W / 2).astype(int)
    win = int(W)
    emg_pad = np.pad(emg, ((npad, npad), (0, 0)), 'symmetric')
    for i in range(len(emg_rect)):
        emg_rect[i, :] = np.sqrt(np.mean(emg_pad[i:i + win, :]**2, axis=0))
    return emg_rect

