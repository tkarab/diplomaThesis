import os.path
import numpy as np
import constants
import helper_functions
import scipy


"""
    For segmenting sEMG signal using sliding window of given shape and size
    We assume that sEMG signal is given in form of a numpy array with dimensions (time_length x channels) i.e. (500x12)
    It returns a 1-D numeric array with all the starting indices of the segments.
    
    i.e. if element slice_start_indices[i]=124 and window size = 15, the segment should be:
        emg[slice_start_indices[i] : slice_start_indices[i] + window_size][:]
    
    PARAMETERS
    x : np.ndarray -> the emg signal
    window_size : int -> the window size
    window_step : int -> the window step
    
    RETURNS
    slice_start_indices : np.ndarray -> array of length N where N is the number of segments, which contains
                                        sta starting indices of all segments
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

def rmsRect(emg:np.ndarray, fs = 2000, win_size_ms=200):
    emg_rect = np.zeros(emg.shape)
    W = int(win_size_ms*fs/1000)
    npad = np.floor(W / 2).astype(int)
    win = int(W)
    emg_pad = np.pad(emg, ((npad, npad), (0, 0)), 'symmetric')
    for i in range(len(emg_rect)):
        emg_rect[i, :] = np.sqrt(np.mean(emg_pad[i:i + win, :]**2, axis=0))
    return emg_rect

#Main

# old_freq = 2000
# new_freq = 100
#
# path = constants.DATA_PATH
# filepath = os.path.join(path, 'emg.npz')
# emg = np.transpose(np.load(filepath)['arr_0'])
# emg_rect = rmsRect(emg, fs=old_freq, win_size_ms=200)
# emg_rect_sub = subsample(emg_rect, init_freq=old_freq, new_freq=new_freq)
# emg_rect_sub_filt = applyLPFilter(emg_rect_sub, Fc=1, Fs=new_freq, N=1)
#
# emg_seg = get_segmentation_indices(emg_rect_sub_filt, window_size=15, window_step=6)
#
# helper_functions.plot_sEMG(emg, fs=old_freq, figure_name='raw')
# helper_functions.plot_sEMG(emg_rect, fs=old_freq, figure_name='rectified')
# helper_functions.plot_sEMG(emg_rect_sub, fs=new_freq, figure_name='subsampled')
# helper_functions.plot_sEMG(emg_rect_sub_filt, fs=new_freq, figure_name='filtered')
#
# helper_functions.plotSpectrum(emg[:,0], fs=old_freq, figure_name='raw spec')
# helper_functions.plotSpectrum(emg_rect[:,0], fs=old_freq, figure_name='rect spec')
# helper_functions.plotSpectrum(emg_rect_sub[:,0], fs=new_freq, figure_name='subsampled spec')
# helper_functions.plotSpectrum(emg_rect_sub_filt[:,0], fs=new_freq, figure_name='filtered spec')
#
#
#
# print()