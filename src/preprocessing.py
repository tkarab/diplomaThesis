import numpy as np
from tqdm.auto import tqdm
import time
import constants
import os
import helper_functions
from preprocessing_functions import *

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
"""
    Applies preprocessing steps to the data, either all of them or the remaining ones in case only a portion
    of them have been processed
"""
try:
    data_proc = np.load(processed_data_path)
    data_seg = np.load(segments_path)
    # keys of the remaining gestures that are yet to be processed
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

    # OUTPUT SHAPE: (time,channels)
    # Step 1: Transpose
    # Data are in the form channels x Time (i.e. 12x500)
    # They need to be transposed (Time x channels)
    emg = np.transpose(data_sep[key])

    # OUTPUT SHAPE: (time,channels)
    # Step 2: RMS Rectification
    emg_rect = rmsRect(emg, fs=old_freq, win_size_ms=200)

    # OUTPUT SHAPE: (reduced_time,channels)
    # Step 3: Subsampling
    # No need for anti-aliasing filter due to RMS rectification
    emg_rect_sub = subsample(emg_rect, init_freq=old_freq, new_freq=new_freq)

    # OUTPUT SHAPE: (reduced_time,channels)
    # Step 4: Low pass filter
    emg_rect_sub_filt = applyLPFilter(emg_rect_sub, Fc=1, Fs=new_freq, N=1)

    # Step 5: Signal segmentation
    # The following function creates a N sized array of integers, each corresponding to the start of a segment of the signal.
    # The number of segments is determined by the signal size and the given window size and step size
    # So in reality the signal doesn't get segmented (to avoid duplicate information and around window_size/step_size
    # as much space (i.e. 15/6 = 2.5).
    # Instead the starting indices of the would-be segments are created and stored and then a segment is taken each time
    # by indexing the signal as follows: emg[random_segment_starting_index:random_segment_starting_index+window_size]
    emg_seg = get_segmentation_indices(emg_rect_sub_filt, window_size=15, window_step=6)

    # Printing total time of the preprocessing operation after each loop
    total_time = time.time() - t1
    print(f"key: '{key}' ({i+1}/{L})")
    print(f"time: {total_time:.2f}")

    # SAVED SHAPE: (reduced_time,channels,1)
    # Expands the last dimension of the signal so that it goes from (time,channels) to (time,channels,1)
    # ie. (740,12) -> (740,12,1)
    processed_data[key] = np.expand_dims(emg_rect_sub_filt,-1)
    segments[key] = emg_seg

# Saving the newly created data
processed_data = {**processed_data, **old_processed_data}
segments = {**segments, **old_segments}

# print()
np.savez(processed_data_path, **processed_data)
np.savez(segments_path, **segments)
