import numpy as np
import os
import constants
import helper_functions as h
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

def apply_augmentation(data, config_dict:dict):
    print("Performing Data Augmentation...\n")
    data_aug = {key: None for key in data}
    ops = config_dict['ops']
    params = config_dict['params']

    t1 = time.time()
    for key,emg in data.items():

        for op in [op for op in augmentation_operations if ops[op] == True]:
            emg = augmentation_funcs[op](emg, **params[op])
        data_aug[key] = np.copy(emg)

        if key[3:] == 'g49r06' :
            print(f"{key[:3]}/{len(data.items())//294} : {time.time()-t1:.2f}s")
            t1 = time.time()

    print("\n...augmentation has finished")
    return data_aug


augmentation_operations = ["AWGN", "FLIP"]
augmentation_funcs = {
    "AWGN" : addGaussianNoise,
    "FLIP" : None
}

# Main
if __name__ == "__main__" :
    config_dict = h.get_config_from_json_file('aug', 'db2_awgn')
    apply_augmentation(np.load(os.path.join(constants.PROCESSED_DATA_PATH_DB2,'db2_processed.npz')), config_dict)






