import constants
import numpy as np
import time
import os
import sys
import helper_functions as help

def calculate_total_size(data_rms:dict):
    total_size = 0
    # Total size in bytes
    for key,emg in data_rms:
        total_size += emg.nbytes
    # size in giga bytes
    total_size/=(2**30)
    print(f"Total size: {total_size:.2f}Gb")

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

    if not os.path.exists(full_path):
        os.mkdir(full_path)

    full_file_path = os.path.join(full_path,filename)
    np.savez(full_file_path, **data_rms)
    print(f"Rectified data saved at: '{full_file_path}'")

    return

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

    emg_rect = np.sqrt((emg_pad_csum[win:-1]-emg_pad_csum[:-win-1])/win)
    return emg_rect


"""
DESCRIPTION

PARAMETERS
    db_dir_path:    full path of the directory where the data of the database in question exist 
                    ie 'C:\\Users\\ΤΑΣΟΣ\\Desktop\\Σχολή\\Διπλωματική\\Δεδομένα\\processed\\db2'
"""
def apply_rms_rect(db: int, db_dir_path: str, fs: int, win_size_ms: int):
    rms_dir_name = help.get_rmsRect_dirname(db, win_size_ms)  # ie 'db2_rms_100'
    full_rms_dir_path = os.path.join(db_dir_path, rms_dir_name)
    rms_filename = rms_dir_name + '.npz'

    separated_data_filename = os.path.join(constants.SEPARATED_DATA_PATH, f'db{db}.npz')
    data_sep_raw = np.load(separated_data_filename)

    already_rectified = 0

    # Case where the folder exists (and thus the rectification has either been completed or at least partially done
    if rms_dir_name in os.listdir(db_dir_path):
        # Checking whether all gestures have been rectified
        data_rms = np.load(os.path.join(full_rms_dir_path, rms_filename))

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
        emg_rms = rmsRect2(emg, win_size_ms=win_size_ms, fs=fs)

        data_rms[key] = np.copy(emg_rms)
        if (key[3:] == 'g49r06'):
            time_for_subject = time.time() - t1
            print(f"subject '{key[:3]}' ({already_rectified + i + 1}/{total_keys}) - {time_for_subject:.2f}s")
            t1 = time.time()

    print("total_time:", time.time() - t_start)
    save_rectified_gestures(data_rms, full_path=full_rms_dir_path, filename=rms_filename)
    calculate_total_size(data_rms)

    return


"""    -- MAIN --    """

if __name__ == "__main__":
    try:
        db = int(sys.argv[1])
        fs = int(sys.argv[2])
        win_size_ms = int(sys.argv[3])
    except IndexError:
        # Default values
        db = 2
        fs = 2000
        win_size_ms = 200

    if db == 1:
        path = constants.PROCESSED_DATA_PATH_DB1
    elif db == 2:
        path = constants.PROCESSED_DATA_PATH_DB2
    elif db == 5:
        path = constants.PROCESSED_DATA_PATH_DB5
    else:
        exit(0)

    apply_rms_rect(db=db, db_dir_path=path, fs=fs, win_size_ms=win_size_ms)
