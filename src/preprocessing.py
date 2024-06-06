import numpy as np
from tqdm.auto import tqdm
import time
import constants
import os
import helper_functions
from proprocessing_functions import *


filepath = os.path.join(constants.SEPARATED_DATA_PATH, 'db2.npz')
processedPath = os.path.join(constants.PROCESSED_DATA_PATH_DB2,'db2_processed.npz')

data_sep = np.load(filepath)
data_proc = np.load(processedPath)
# subjects,gestures,reps = helper_functions.get_unique_sgr(data_sep.files)

keylist = list(set(data_sep.files) - set(data_proc.files))
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

print()
np.savez(os.path.join(processedPath,"db2_processed2.npz"), **processed_data)
np.savez(os.path.join(processedPath,"segments2.npz"), **segments)
