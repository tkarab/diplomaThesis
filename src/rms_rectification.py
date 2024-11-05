import numpy as np
import time
import os
import sys

from plot_functions import *
from preprocessing import *
from helper_functions import *
from constants import *


def calculate_total_size(data_rms:dict):
    total_size = 0
    # Total size in bytes
    for key,emg in data_rms.items():
        total_size += emg.nbytes
    # size in giga bytes
    total_size/=(2**30)
    print(f"Total size: {total_size:.1f}Gb")

"""
DESCRIPTION
    For saving the dict of existing rms-rectified signals in the given path.
    Takes into account that the dict containing the rectified data, has all the possible
    keys for the specific database but not all data have been rectified, and so some values 
    for specific keys are None.
    If no rectification has been done and no folder exists, it creates one based on the given name
"""
def save_rectified_gestures(data_rms: dict, full_path: str, filename:str):
    # Keeps only the non None values
    data_rms = {key: data_rms[key] for key in data_rms if data_rms[key] is not None}

    full_file_path = os.path.join(full_path,filename)
    np.savez(full_file_path, **data_rms)
    print(f"Rectified data saved at: '{full_file_path}'")
    calculate_total_size(data_rms)
    return

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
def rmsRect(x:np.ndarray, fs = 2000, win_size_ms=200):
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

    emg_rect = np.sqrt((emg_pad_csum[win:-1]-emg_pad_csum[:-win-1])/win)
    return emg_rect


"""
DESCRIPTION

PARAMETERS
    db_dir_path:    full path of the directory where the data of the database in question exist 
                    ie 'C:\\Users\\ΤΑΣΟΣ\\Desktop\\Σχολή\\Διπλωματική\\Δεδομένα\\processed\\db2'
"""
def apply_rms_rect(db: int, db_dir_path: str, fs: int, win_size_ms: int):
    rms_filename = get_rms_rect_filename(db, win_size_ms)  # ie 'db2_rms_100.npz'
    full_rms_file_path = os.path.join(db_dir_path, rms_filename)
    # rms_filename = rms_dir_name + '.npz'

    separated_data_filename = os.path.join(SEPARATED_DATA_PATH, f'db{db}.npz')
    data_sep_raw = np.load(separated_data_filename)

    already_rectified = 0

    # Case where the folder exists (and thus the rectification has either been completed or at least partially done
    if rms_filename in os.listdir(db_dir_path):
        # Checking whether all gestures have been rectified
        data_rms = np.load(full_rms_file_path)

        # If all the keys exist in the file, then rms rectification with that specific window size has already been doneand there is no need to redo
        if (set(data_rms.files) == set(data_sep_raw.files)):
            print("RMS Rectification with that window size already exists")
            return
        already_rectified = len(data_rms.files)
        remaining_keys = sorted(list(set(data_sep_raw.files) - set(data_rms.files)))
        # Copying all values of already rectified gestures to the corresponding keys in data_rms dict
        data_rms = {key: data_rms[key] for key in data_rms.files}
        # Initializing values for all keys of non-rectified gestures
        for key in remaining_keys:
            data_rms[key] = None

    else:
        remaining_keys = data_sep_raw.files
        # Initializing values for all keys to None
        data_rms = {key: None for key in remaining_keys}

    total_keys = len(remaining_keys) + already_rectified
    t_start = time.time()
    t1 = time.time()
    for i, key in enumerate(remaining_keys):
        emg = data_sep_raw[key]
        emg_rms = rmsRect(emg, win_size_ms=win_size_ms, fs=fs)

        data_rms[key] = np.copy(emg_rms)
        if (key[3:] == 'g49r06'):
            time_for_subject = time.time() - t1
            print(f"subject '{key[:3]}' ({already_rectified + i + 1}/{total_keys}) - {time_for_subject:.2f}s")
            t1 = time.time()

    print("total_time:", time.time() - t_start)
    save_rectified_gestures(data_rms, full_path=db_dir_path, filename=rms_filename)

    return

"""
DESCRIPTION 
    Applies rms rectification (and optionally subsampling) to all recordings of the subject provided, 
    and saves it in the respective folder in the Preprocessed/db6 directory.

"""
def apply_rms_rect_db6(subject, win_size_ms, subsample_enabled = True):
    days = [1,2,3,4,5]
    times = [1,2]
    print(f"Subject: {subject}")

    subject_sep_data_path = os.path.join(SEPARATED_DATA_PATH, 'db6', f'db6_s{subject}')

    db6_rms_data_path = os.path.join(PROCESSED_DATA_PATH_DB6,f'rms{win_size_ms}')
    if not os.path.exists(db6_rms_data_path):
        os.mkdir(db6_rms_data_path)

    subject_rms_data_path = os.path.join(db6_rms_data_path, f'db6_s{subject}')
    if not os.path.exists(subject_rms_data_path):
        os.mkdir(subject_rms_data_path)

    for d in days:
        for t in times:
            print(f"\nD:{d}, T:{t}")
            dt_path = os.path.join(subject_sep_data_path, f's{subject}_d{d}_t{t}.npz')
            rms_path = os.path.join(subject_rms_data_path,f's{subject}_d{d}_t{t}.npz')

            dt_sep_dict = np.load(dt_path)
            rms_dict = {}
            keys = dt_sep_dict.keys()

            t = time.time()
            for key in keys:
                emg = dt_sep_dict[key]
                emg_rect = rmsRect(x=emg, win_size_ms=win_size_ms, fs=2000)
                if subsample_enabled == True:
                    emg_sub = subsample(x=emg_rect, init_freq=2000, new_freq=100)
                    rms_dict[key] = np.copy(emg_sub)
                else:
                    rms_dict[key] = np.copy(emg_rect)

            np.savez(rms_path, **rms_dict)
            print(f"{time.time() - t:.2f}")
            t = time.time()






"""    -- MAIN --    """
win_size_ms = 200
db = 6

if __name__ == "__main__":

    if db == 6:
        for sub in [1,2,3,4,5,6,7,8,9,10]:
            apply_rms_rect_db6(subject=sub, win_size_ms=win_size_ms, subsample_enabled=True)
    else:
        if db == 1:
            path = PROCESSED_DATA_PATH_DB1
            fs = 100
        elif db == 2:
            path = RMS_DATA_PATH_DB2
            fs= 2000
        elif db == 5:
            path = PROCESSED_DATA_PATH_DB5
            fs = 200

        apply_rms_rect(db=db, db_dir_path=path, fs=fs, win_size_ms=win_size_ms)
