import json

import numpy as np
import time
import constants
import os
import helper_functions
import scipy
import plot_functions as pl
import data_augmentation as aug

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
def get_segmentation_indices(x: np.ndarray, window_size: int, window_step: int):
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


# Keeps only a certain amount of samples from each emg, the middle seconds_to_keep ones
def discard_early_and_late_gest_stages(x, seconds_to_keep, fs):
    num_samples_to_keep = int(seconds_to_keep*fs)
    # Half the length of samples to keep
    W = num_samples_to_keep // 2
    L = len(x)
    return x[max(L // 2 - W, 0):min(L // 2 + W, L)]


def apply_preprocessing(data_path, config_dict:dict):

    data = np.load(data_path)
    data_proc = {key:None for key in data}
    data_seg = {key:None for key in data}

    config_operations = config_dict['ops']
    config_params = config_dict['params']
    op_no_seg = [op for op in preprocess_operations if not op == "SEGMENT"]

    if config_operations["LOWPASS"] == True :
        b, a = get_filter_coeffs(**config_params["LOWPASS"])
        config_params["LOWPASS"] = {"b" : b, "a" : a}

    t1 = time.time()
    for key,emg in data.items():
        for op in op_no_seg:
            if config_operations[op] == True:
                emg = preprocess_funcs[op](emg, **config_params[op])

        data_proc[key] = np.copy(emg)

        if config_operations["SEGMENT"] == True:
            data_seg[key] = get_segmentation_indices(emg,**config_params["SEGMENT"])

        if key[3:] == "g49r06" :
            print(f"'{key[:3]}' : {time.time()-t1:.2f}s")
            t1 = time.time()

    return data_proc, data_seg





preprocess_operations = ["SUBSAMPLE", "DISCARD", "LOWPASS", "MIN-MAX", "M-LAW", "SEGMENT"]

preprocess_funcs = {
    "DISCARD"   :   discard_early_and_late_gest_stages,
    "SUBSAMPLE" :   subsample,
    "LOWPASS"   :   applyLPFilter,
    "MIN-MAX"   :   None,
    "M-LAW"     :   None,
    "SEGMENT"   :   get_segmentation_indices
}


if __name__ == "__main__":
    config = helper_functions.get_config_from_json_file(mode="preproc", filename='db2_lpf')
    data_dir_path = os.path.join(constants.PROCESSED_DATA_PATH_DB2, r'db2_rms_200\db2_rms_200.npz')
    data_proc, segments = apply_preprocessing(data_dir_path, config)

    aug_config = helper_functions.get_config_from_json_file('aug','db2_awgn')
    aug.apply_augmentation(data_proc, aug_config)

