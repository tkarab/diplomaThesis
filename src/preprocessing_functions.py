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
    EXAMPLE
    for a sliding window with step size 6, the starting indices of each segment should look like this: [0 6 12 18....]
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
    x  : semg signal
    Fc : Cutoff frequency
    Fs : Sampling frequency
    N  : Filter order
    
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


def rmsRect(x:np.ndarray, fs = 2000, win_size_ms=200):
    emg_rect = np.zeros(x.shape)
    W = int(win_size_ms*fs/1000)

    # npad: window_length/2 (used later for padding)
    npad = np.floor(W / 2).astype(int)
    win = int(W)

    # Symmetric padding with half the length of the window from each side
    # Thus ensuring the sliding window won't affect the total signal length
    # i.e. for x = [0,1,2,3,4,5,6,7] and W/2 == 2 symmetric padding should be
    #          [1,0,0,1,2,3,4,5,6,7,7,6]
    emg_pad = np.pad(x, ((npad, npad), (0, 0)), 'symmetric')

    # emg[i] is replaced by the rms value of all the samples contained by the sliding window
    # centered in position i
    for i in range(len(emg_rect)):
        emg_rect[i, :] = np.sqrt(np.mean(emg_pad[i:i + win, :]**2, axis=0))
    return emg_rect

"""
DESCRIPTION
    Faster version for performing RMS Rectification on emg data
    Instead of calculating sum of squares over a given time window it calculates the squares and their 
    cumulative sum beforehand and saves it in the emg_pad_csum array. That way emg_pad_csum[i] consists
    of the sum of all squares up to the i-th element in the original emg recording. Therefore, to calculate
    the sum of squares over a window (which covers the indices from i to j) all you need to do is calculate
    the difference between emg_pad_csum[j] and emg_pad_csum[i-1].

PARAMETERS
    x: emg recording to be rectified
    fs: sampling rate (2000 for DB2)
    win_size_ms: window size in milliseconds
"""
def rmsRect2(x:np.ndarray, fs = 2000, win_size_ms=200):
    emg_rect = np.zeros(x.shape)
    W = int(win_size_ms*fs/1000)

    # npad: window_length/2 (used later for padding)
    npad = np.floor(W / 2).astype(int)
    win = int(W)

    # Symmetric padding with half the length of the window from each side
    # Thus ensuring the sliding window won't affect the total signal length
    # i.e. for x = [0,1,2,3,4,5,6,7] and W/2 == 3 symmetric padding should be
    #        [2,1,0,0,1,2,3,4,5,6,7,7,6,5]
    emg_pad = np.pad(x, ((npad, npad), (0, 0)), 'symmetric')

    # Square values of all cells
    emg_pad_squared = emg_pad**2
    # Cumulative sum along the time axis (where emg_pad_csum[i] = sum(emg_pad_squared[:i] for each channel)
    emg_pad_csum = np.cumsum(emg_pad_squared,axis=0)

    # emg[i] is replaced by the rms value of all the samples contained by the sliding window
    # centered in position i
    emg_pad_csum = np.pad(emg_pad_csum,((1,0),(0,0)) ,mode='constant', constant_values=0)
    for i in range(len(emg_rect)):
        emg_rect[i, :] = np.sqrt((emg_pad_csum[i+win,:] - emg_pad_csum[i])/win)
    return emg_rect

# Keeps only a certain amount of samples from each emg, the middle 'num_samples_to_keep' ones
def discard_early_and_late_gest_stages(x, num_samples_to_keep):
    # Half the length of samples to keep
    W = num_samples_to_keep//2
    L = len(x)
    return x[max(L//2 - W, 0):min(L//2 + W, L)]


