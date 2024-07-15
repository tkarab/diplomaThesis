import numpy as np
import os
import sys
import random
# import tensorflow as tf
# import custom_models
import preprocessing_functions as pr
import constants
import time
import helper_functions
import plot_functions

def getKeys(*entries:tuple):
    return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

DATABASE = 2

preprocess_operations = ["SUBSAMPLE", "DISCARD", "LOWPASS", "MIN-MAX", "M-LAW"]

preprocess_config = {
    "SUBSAMPLE" :   True,
    "DISCARD"   :   True,
    "LOWPASS"   :   True,
    "MIN-MAX"   :   False,
    "M-LAW"     :   False,
    "SEGMENT"   :   True
}
preprocess_funcs = {
    "DISCARD"   :   pr.discard_early_and_late_gest_stages,
    "SUBSAMPLE" :   pr.subsample,
    "LOWPASS"   :   pr.applyLPFilter,
    "MIN-MAX"   :   None,
    "M-LAW"     :   None,
    "SEGMENT"   :   pr.get_segmentation_indices
}
preprocess_params = {
    "DISCARD"   :   {"num_samples_to_keep" : 3.5*100},
    "SUBSAMPLE" :   {"init_freq" : 2000, "new_freq" : 100},
    "LOWPASS"   :   {"Fc" : 1, "Fs" : 100, "N" : 1},
    "MIN-MAX"   :   {},
    "M-LAW"     :   {},
    "SEGMENT"   :   {"window_size" : 15, "window_step" : 6}
}

separated_data_filename = os.path.join(constants.SEPARATED_DATA_PATH,f'db{DATABASE}.npz')

operations_to_perform = [op for op in preprocess_operations if preprocess_config[op] == True]

data_sep_raw = np.load(separated_data_filename)
data_proc = {}
data_seg = {}

keys = data_sep_raw.files

total_time_per_operation = {op : 0 for op in operations_to_perform}
total_time_per_operation["SEGMENT"] = 0

"""
t_start = time.time()
for i,key in enumerate(keys):
    emg = data_sep_raw[key]

    t_i = time.time()
    for operation in operations_to_perform:
        func = preprocess_funcs[operation]
        params = preprocess_params[operation]
        t1 = time.time()
        emg = func(x=emg,**params)
        total_time_per_operation[operation] += time.time() - t1
    data_proc[key] = np.copy(emg)
    print(f"key {i+1}/{len(keys)}: {time.time()-t_i}")

    if preprocess_config["SEGMENT"] == True:
        segments = preprocess_funcs["SEGMENT"](x=emg,**preprocess_params["SEGMENT"])
        t1 = time.time()
        data_seg[key] = np.copy(segments)
        total_time_per_operation["SEGMENT"] += time.time()-t1


total_time = time.time() - t_start
print()
"""





def apply_preprocessing(preprocess_operations : dict, preprocess_config : dict, preprocess_funcs : dict, preprocess_params : dict):
    return

"""
DESCRIPTION
    For saving the dict of existing rms-rectified signals in the given path.
    Takes into account that the dict containing the rectified data, has all the possible
    keys for the specific database but not all data have been rectified, and so some values 
    for specific keys are None
"""
def save_rectified_gestures(data_rms : dict, full_path : str):
    # Keeps only the non None values
    data_rms = {key:data_rms[key] for key in data_rms if data_rms[key] is not None}

    np.savez(full_path, **data_rms)
    return

"""
DESCRIPTION
    
PARAMETERS
    db_dir_path:    full path of the directory where the data of the database in question exist 
                    ie 'C:\\Users\\ΤΑΣΟΣ\\Desktop\\Σχολή\\Διπλωματική\\Δεδομένα\\processed\\db2'
"""
def apply_rms_rect(db:int, db_dir_path:str,  fs:int, win_size_ms:int):
    rms_dir_name = helper_functions.get_rmsRect_dirname(db, win_size_ms)    # ie 'db2_rms_100'
    full_rms_dir_path = os.path.join(db_dir_path, rms_dir_name)
    rms_filename = rms_dir_name + '.npz'

    separated_data_filename = os.path.join(constants.SEPARATED_DATA_PATH, f'db{db}.npz')
    data_sep_raw = np.load(separated_data_filename)

    already_rectified = 0

    # Case where the folder exists (and thus the rectification has either been completed or at least partially done
    if rms_dir_name in os.listdir(db_dir_path):
        # Checking whether all gestures have been rectified
        data_rms = np.load(os.path.join(full_rms_dir_path,rms_filename))

        # If all the keys exist in the file, then rms rectification with that specific window size has already been doneand there is no need to redo
        if(set(data_rms.files) == set(data_sep_raw.files)):
            print("RMS Rectification with that window size already exists")
            return
        already_rectified = len(data_rms.files)
        remaining_keys = sorted(list(set(data_sep_raw.files)-set(data_rms.files)))
        # Copying all values of already rectified gestures to the corresponding keys in data_rms dict
        data_rms = {key:data_rms[key] for key in data_rms.files}
        # Initializing values for all keys of non-rectified gestures
        for key in remaining_keys:
            data_rms[key] = None

    else:
        os.mkdir(full_rms_dir_path)
        remaining_keys = data_sep_raw.files
        # Initializing values for all keys to None
        data_rms = {key:None for key in remaining_keys}

    # TODO - Remove
    time_list = []
    total_keys = len(remaining_keys) + already_rectified

    t_start = time.time()
    t1 = time.time()
    for i,key in enumerate(remaining_keys):
        emg = data_sep_raw[key]
        emg_rms = pr.rmsRect2(emg, win_size_ms=win_size_ms, fs=fs)
        # emg_rms2 = pr.rmsRect(emg, win_size_ms=win_size_ms, fs=fs)
        # print(np.array_equal(emg_rms, emg_rms2), "rms difference:",np.sqrt(np.mean((emg_rms2-emg_rms)**2)))

        data_rms[key] = np.copy(emg_rms)
        if(key[3:] == 'g49r06'):
            time_for_subject = time.time()-t1
            print(f"key '{key}' ({already_rectified+i+1}/{total_keys}) - {time_for_subject:.2f}s")
            time_list.append(time_for_subject)
            t1 = time.time()

    print("total_time:",time.time()-t_start)
    save_rectified_gestures(data_rms, full_path=os.path.join(full_rms_dir_path,rms_filename))

    return


"""    -- MAIN --    """

apply_rms_rect(db=DATABASE, db_dir_path=constants.PROCESSED_DATA_PATH_DB2, fs=2000, win_size_ms=160)
# f1 = 2000
# f2 = 100
#
# rms_w_size = [50,100,200]
# separated_data_filename = os.path.join(constants.SEPARATED_DATA_PATH, f'db{DATABASE}.npz')
# data_s = np.load(separated_data_filename)
# key = random.choice(data_s.files)
# emg = data_s[key]
#
# for w in rms_w_size:
#     emg_rms = np.copy(pr.rmsRect(emg,fs=f1, win_size_ms=w))
#     emg_sub = np.copy(pr.subsample(emg_rms, init_freq=f1, new_freq=f2))
#     plot_functions.plot_sEMG(emg_sub,fs=f2, figure_name=f"{w}ms {f2}Hz", title=f"{w}ms")
#     plot_functions.plotSpectrum(emg_rms,fs=f1,title=f"{w}ms {f1}Hz", figure_name=f"{w}ms {f1}Hz")
#     plot_functions.plotSpectrum(emg_sub, fs=f2, title=f"{w}ms {f2}Hz", figure_name=f"{w}ms {f2}Hz")
#
#
