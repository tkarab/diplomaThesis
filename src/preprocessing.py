import json

import numpy as np
import time
import os
import scipy
import plot_functions as pl
import data_augmentation as aug
from helper_functions import *
from constants import *

"""
    For segmenting sEMG signal using sliding window of given shape and size
    We assume that sEMG signal is given in form of a numpy array with dimensions (time_length x channels) i.e. (500x12)
    It returns a 1-D numeric array with all the starting indices of the segments.

    i.e. if element slice_start_indices[i]=124 and window size = 15, the segment should be:
        emg[slice_start_indices[i] : slice_start_indices[i] + window_size][:]
    ->  emg[124:139][:]

    PARAMETERS
    x : np.ndarray -> the emg signal
    window_size_ms : int -> the window size (in milliseconds)
    window_step_ms : int -> the window step (- \\ -)
    fs : the sampling rate in Hz

    RETURNS
    slice_start_indices : np.ndarray -> array of length N where N is the number of segments, which contains
                                        the starting indices of all segments
    EXAMPLE
    for a sliding window with step size 6, the starting indices of each segment should look like this: [0 6 12 18....]
"""
def get_segmentation_indices(x: np.ndarray, window_size_ms: int, window_step_ms: int, fs):
    window_size = int((window_size_ms * fs) / 1000)
    window_step = int((window_step_ms * fs) / 1000)
    slice_start_indices = np.arange(0, len(x) - window_size + 1, window_step)
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
def subsample(x: np.ndarray, init_freq: float, new_freq: float):
    sub_factor = int(init_freq / new_freq)
    indices = np.arange(0, len(x), sub_factor)
    return np.take(x, indices=indices, axis=0)

def get_filter_coeffs(fc=1, fs=100, N=1):
    f = 2 * fc / fs
    b, a = scipy.signal.butter(N=N, Wn=f, btype='low')

    return b,a

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
def applyLPFilter(x, b ,a):
    x = np.abs(x)
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

    return output


def applyLPFilter2(emg, fc=1, fs=100, N=1):
    f_sos = scipy.signal.butter(N=N, Wn=2 * fc / fs, btype='low', output='sos')
    return scipy.signal.sosfilt(f_sos, emg, axis=0)

def minmax_norm(x):
    max_x = np.max(x)
    min_x = np.min(x)
    return (x-min_x) / (max_x-min_x)

"""
DESCRIPTION
    Performs μ-Law on a given emg signal x for a given value of μ
PARAMETERS
    - χ: The emg signal
    - μ: The value used n the formula
    - scaling_type: "all" for scaling all channels with the absolute maximum of the while signal
                    "each" for scaling with the maximum of each channel
"""
def muLaw_transform(x, mu = 2048, scaling_type = 'all'):

    if scaling_type == "each":
        x = x/np.max(np.abs(x),axis=0)
    else:
        x= x/np.max(np.abs(x))

    return np.sign(x)*(np.log(1+mu*np.abs(x))/np.log(1+mu))

# Keeps only a certain amount of samples from each emg, the middle seconds_to_keep ones
def discard_early_and_late_gest_stages(x, seconds_to_keep, fs):
    num_samples_to_keep = int(seconds_to_keep*fs)
    # Half the length of samples to keep
    W = num_samples_to_keep // 2
    L = len(x)
    return x[max(L // 2 - W, 0):min(L // 2 + W, L)]


def apply_preprocessing(data_path, config_dict:dict, db:int):
    print("Performing preprocessing...\n")

    if db == 2:
        gestures = 49
        reps = 6
    elif db == 5:
        gestures = 52
        reps = 6
    elif db == 1:
        gestures = 52
        reps = 10
    total_samples_per_gesture = gestures * reps
    final_sample_subfix = f'g{gestures:02d}r{reps:02d}'

    data = np.load(data_path)
    data_proc = {key:None for key in data}
    data_seg = {key:None for key in data}

    config_operations = config_dict['ops'].copy()
    config_params = config_dict['params'].copy()
    op_no_seg = [op for op in preprocess_operations if not op == "SEGMENT"]

    if config_operations["LOWPASS"] == True :
        b, a = get_filter_coeffs(**config_params["LOWPASS"])
        config_params["LOWPASS"] = {"b" : b, "a" : a}

    operations_params = [(preprocess_funcs[op], config_params[op]) for op in preprocess_operations if
                         config_operations[op] == True and op != "SEGMENT"]

    t1 = time.time()
    for key, emg in data.items():
        # for op in op_no_seg:
        if key == "s12g34r04":
            print()
        for func, params in operations_params:
            emg = func(emg, **params)

        data_proc[key] = np.expand_dims(emg, -1)

        if key[3:] == final_sample_subfix:
            print(f"{key[:3]}/{len(data.items()) // total_samples_per_gesture} : {time.time() - t1:.2f}s")
            t1 = time.time()


    if config_operations["SEGMENT"] == True:
        for key, emg in data_proc.items():
            data_seg[key] = get_segmentation_indices(emg, **config_params["SEGMENT"])
    print("\n...preprocessing has finished\n")

    return data_proc, data_seg

preprocess_operations = ["SUBSAMPLE", "DISCARD", "LOWPASS", "M-LAW", "MIN-MAX", "SEGMENT"]

preprocess_funcs = {
    "SUBSAMPLE" :   subsample,
    "DISCARD"   :   discard_early_and_late_gest_stages,
    "LOWPASS"   :   applyLPFilter,
    "M-LAW"     :   muLaw_transform,
    "MIN-MAX"   :   minmax_norm,
    "SEGMENT"   :   get_segmentation_indices
}

if __name__ == "__main__":
    new_freq = 100
    for rms in [50,100,150,250]:
        unsampled_data_path = os.path.join(RMS_DATA_PATH_DB2,get_rms_rect_filename(2,rms))
        data = np.load(unsampled_data_path)
        subsampled_data = {}

        t1 = time.time()
        for key,emg in data.items():
            emg_sub = subsample(emg,2000,new_freq=new_freq)
            subsampled_data[key] = emg_sub
        total_time = time.time() - t1
        print(f"total time for subsampling {len(data.items())} recordings: {total_time:.2f}s")

        new_path = os.path.join(RMS_DATA_PATH_DB2,get_rms_sub_filename(2,rms,new_freq))
        np.savez(new_path,**subsampled_data)

        print(f"rms {rms} done")
