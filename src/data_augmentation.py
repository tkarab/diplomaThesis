import numpy as np
import os
import constants
import random
import plot_functions
import time

processed_data_path = os.path.join(constants.PROCESSED_DATA_PATH_DB2, 'db2_processed.npz')

"""
DESCRIPTION
    Implements data augmentation methods for the preprocessed signal
    Augmented data are save in a folder called 'aug' within the directory of each .npz file containing the processe data.
    If this directory doesn't exist (meaning no augmentation has taken place) then the program will create one 
    and save the data in there.

CODE NAMES FOR EACH AUGMENTATION PROCESS
    For each process there are the following codes:
    
    AWGN - Additive Gaussian White Noise
    
    
"""

processed_data_path = constants.PROCESSED_DATA_PATH_DB2
processed_data_filename = os.path.join(processed_data_path,'db2_processed.npz')

aug_config = {
    "AWGN" : True
}

SNR = 25

# Functions
def addGaussianNoise(emg:np.ndarray, snr_db:int = 25):
    # reduce dimension (time,channels,1) -> (time,channels)
    # i.e. (740,12,1) -> (740,12)
    emg = np.squeeze(emg)

    # provides the mean square value (power) along each channels
    # Should return a (1,channels) sized output where each element is the mean square value of each channel
    emg_mean_square = np.mean(emg**2, axis=0,keepdims=True)

    # SNR from db to linear value
    snr = 10 ** (snr_db/10)

    # Gaussian noise power (variance)
    noise_power = emg_mean_square/snr
    # stdev should be a (1,channels) array where each value is the stdev of the awgn for the respective channel
    stdev = np.sqrt(noise_power)

    # Gaussian Noise creation with 0 mean and the stdev we computed earlier
    awgn = np.random.normal(size=emg.shape, scale=stdev, loc=0.0)
    emg_n = emg + awgn

    return np.expand_dims(emg_n,-1)


# Main

# If the 'aug' folder is not in the directory of the processed data then it is created
if 'aug' not in os.listdir(processed_data_path):
    os.mkdir(os.path.join(processed_data_path,'aug'))

data_proc = np.load(processed_data_filename)
keys = data_proc.files
augmented_data = {}

t_start = time.time()
for i,key in enumerate(keys):
    emg = data_proc[key]

    """
    Applying the function necessary for each operation
    """
    # AWGN
    if aug_config['AWGN'] == True:
        emg_aug = addGaussianNoise(emg, 25)
        augmented_data[key] = emg_aug

    print(f"key {i+1}/{len(keys)}")







