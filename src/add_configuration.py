import json
import constants
import os
import sys
from helper_functions import get_config_full_filename

# PREPROCESSING
preprocess_config = {
    "SUBSAMPLE" :   True,
    "DISCARD"   :   True,
    "LOWPASS"   :   True,
    "MIN-MAX"   :   False,
    "M-LAW"     :   False,
    "SEGMENT"   :   True
}

preprocess_params = {
    "SUBSAMPLE" :   {"init_freq" : 2000, "new_freq" : 100},
    "DISCARD"   :   {"seconds_to_keep" : 3.5, "fs" : 100},
    "LOWPASS"   :   {"fc" : 1, "fs" : 100, "N" : 1},
    "MIN-MAX"   :   None,
    "M-LAW"     :   None,
    "SEGMENT"   :   {"window_size" : 15, "window_step" : 6}
}

# AUGMENTATION
augmentation_config = {
    "AWGN" : True,
    "FLIP" : False
}

augmentation_params = {
    "AWGN" : {"snr_db" : 25},
    "FLIP" : None
}

"""

PARAETERS
    mode : "preprocess", "augment" or "train"
    filename : name of the file. Should be something along the lines of: 'config_preproc_db2_nofilt.json'
               'config_preproc_' is added automatically by choosing preprocess mode.
               Next should be the name of the database and after that something characteristic of the 
               specific configuration
               
"""
def save_config(mode:str, filename:str):
    if mode == "preproc" :
        config_dict = {"ops" : preprocess_config, "params" : preprocess_params}
        full_filename = get_config_full_filename(mode='preproc', name=filename)
        path = os.path.join(constants.DATA_CONFIG_PATH_PREPROC, full_filename)

    elif mode == "aug" :
        config_dict = {"ops": augmentation_config, "params": augmentation_params}
        full_filename = get_config_full_filename(mode='aug', name=filename)
        path = os.path.join(constants.DATA_CONFIG_PATH_AUG,full_filename)

    elif mode == "train" :
        return

    with open(path, 'w') as file:
        json.dump(config_dict, file, indent=4)

# mode : either "preproc" or "aug"
if __name__ == "__main__":
    mode = "preproc"
    save_config(mode, "db2_lpf")