import numpy as np
import os
import sys
import random
# import tensorflow as tf
# import custom_models
import constants
import time
import helper_functions
import plot_functions

def getKeys(*entries:tuple):
    return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

DATABASE = 2



preprocess_config = {
    "SUBSAMPLE" :   True,
    "DISCARD"   :   True,
    "LOWPASS"   :   True,
    "MIN-MAX"   :   False,
    "M-LAW"     :   False,
    "SEGMENT"   :   True
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


