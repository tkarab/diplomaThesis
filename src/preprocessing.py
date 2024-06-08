import numpy as np
from tqdm.auto import tqdm
import time
import constants
import os
import helper_functions
from proprocessing_functions import *

"""
 *DESCRIPTION*
    This code is aimed to pick the separated data from their location and apply the preprocessing
    steps found in 'preprocessing_functions.py', depending on the needs of each Database.
    The way it's written it initially checks whether there already is a file containing preprocessed 
    data of that database within the directory, and if so, it checks whether some keys haven't been
    processed yet. That way the preprocessing process (which is time consuming) could be broken down into
    sessions.

"""


filepath = os.path.join(constants.SEPARATED_DATA_PATH, 'db2.npz')
processed_data_path = os.path.join(constants.PROCESSED_DATA_PATH_DB2, 'db2_processed.npz')
segments_path = os.path.join(constants.PROCESSED_DATA_PATH_DB2, 'db2_segments.npz')

data_sep = np.load(filepath)
try:
    data_proc = np.load(processed_data_path)
    data_seg = np.load(segments_path)
    keylist = list(set(data_sep.files) - set(data_proc.files))
    old_segments = dict(data_seg)
    old_processed_data = dict(data_proc)

except FileNotFoundError:
    keylist = data_sep.files
    old_segments = {}
    old_processed_data = {}


L = len(keylist)
processed_data = {}
segments = {}

old_freq = 2000
new_freq = 100

for i,key in enumerate(keylist):
    t1 = time.time()
    emg = np.transpose(data_sep[key])
    emg_rect = rmsRect(emg, fs=old_freq, win_size_ms=200)
    emg_rect_sub = subsample(emg_rect, init_freq=old_freq, new_freq=new_freq)
    emg_rect_sub_filt = applyLPFilter(emg_rect_sub, Fc=1, Fs=new_freq, N=1)
    emg_seg = get_segmentation_indices(emg_rect_sub_filt, window_size=15, window_step=6)
    total_time = time.time() - t1
    print(f"key: '{key}' ({i+1}/{L})")
    print(f"time: {total_time:.2f}")

    processed_data[key] = emg_rect_sub_filt
    segments[key] = emg_seg

processed_data = {**processed_data, **old_processed_data}
segments = {**segments, **old_segments}

print()
np.savez(processed_data_path, **processed_data)
np.savez(segments_path, **segments)
