import constants
import numpy as np
import time
import os
import sys
import preprocessing_functions as pr
import helper_functions as help

"""
DESCRIPTION
    For saving the dict of existing rms-rectified signals in the given path.
    Takes into account that the dict containing the rectified data, has all the possible
    keys for the specific database but not all data have been rectified, and so some values 
    for specific keys are None
"""


def save_rectified_gestures(data_rms: dict, full_path: str):
    # Keeps only the non None values
    data_rms = {key: data_rms[key] for key in data_rms if data_rms[key] is not None}

    np.savez(full_path, **data_rms)
    return


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
        os.mkdir(full_rms_dir_path)
        remaining_keys = data_sep_raw.files
        # Initializing values for all keys to None
        data_rms = {key: None for key in remaining_keys}

    # TODO - Remove
    time_list = []
    total_keys = len(remaining_keys) + already_rectified

    t_start = time.time()
    t1 = time.time()
    for i, key in enumerate(remaining_keys):
        emg = data_sep_raw[key]
        emg_rms = pr.rmsRect2(emg, win_size_ms=win_size_ms, fs=fs)
        # emg_rms2 = pr.rmsRect(emg, win_size_ms=win_size_ms, fs=fs)
        # print(np.array_equal(emg_rms, emg_rms2), "rms difference:",np.sqrt(np.mean((emg_rms2-emg_rms)**2)))

        data_rms[key] = np.copy(emg_rms)
        if (key[3:] == 'g49r06'):
            time_for_subject = time.time() - t1
            print(f"key '{key}' ({already_rectified + i + 1}/{total_keys}) - {time_for_subject:.2f}s")
            time_list.append(time_for_subject)
            t1 = time.time()

    print("total_time:", time.time() - t_start)
    save_rectified_gestures(data_rms, full_path=os.path.join(full_rms_dir_path, rms_filename))

    return


"""    -- MAIN --    """

if __name__ == "__main__":
    db = int(sys.argv[1])
    fs = int(sys.argv[2])
    win_size_ms = int(sys.argv[3])

    if db == 1:
        path = constants.PROCESSED_DATA_PATH_DB1
    elif db == 2:
        path = constants.PROCESSED_DATA_PATH_DB2
    elif db == 5:
        path = constants.PROCESSED_DATA_PATH_DB5
    else:
        exit(0)

    apply_rms_rect(db=db, db_dir_path=path, fs=fs, win_size_ms=win_size_ms)
