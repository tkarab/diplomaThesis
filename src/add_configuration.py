import json
import constants
import os
import sys

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
    mode : "preprocess" or "augment"
    filename : name of the file. Should be something along the lines of: 'config_preproc_db2_nofilt.json'
               'config_preproc_' is added automatically by choosing preprocess mode.
               Next should be the name of the database and after that something characteristic of the 
               specific configuration
               
               Prefixes are : 'config_preproc_' for preprocessing and 'config_aug_' for augment
"""
def save_config(mode:str, filename:str):
    if mode == "preprocess" :
        config_dict = {"config" : preprocess_config, "params" : preprocess_params}
        full_filename = 'config_preproc_' + filename + '.json'
        path = os.path.join(constants.DATA_CONFIG_PATH_PREPROC, full_filename)

    elif mode == "augment" :
        config_dict = {"config": augmentation_config, "params": augmentation_params}
        full_filename = 'config_aug_' + filename + '.json'
        path = os.path.join(constants.DATA_CONFIG_PATH_AUG,full_filename)

    with open(path, 'w') as file:
        json.dump(config_dict, file, indent=4)

# mode : either "preprocess" or "augment"
if __name__ == "__main__":
    mode = "augment"
    save_config(mode, "db2_awgn_snr25")