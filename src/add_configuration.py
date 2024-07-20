import json
import constants
import os
import sys
from helper_functions import get_config_full_filename

# PREPROCESSING
preprocess_config = {
    "SUBSAMPLE" :   {"enable" : True,
                     "params" : {"init_freq" : 2000, "new_freq" : 100}},

    "DISCARD"   :   {"enable" : True ,
                     "params" : {"seconds_to_keep" : 3.5, "fs" : 100}},

    "LOWPASS"   :   {"enable" : False,
                     # "params" : None},
                     "params" : {"fc": 1, "fs": 100, "N": 1}},

    "MIN-MAX"   :   {"enable" : False,
                     "params" : None},

    "M-LAW"     :   {"enable" : False,
                     "params" : None},

    "SEGMENT"   :   {"enable" : True ,
                     "params" : {"window_size_ms" : 150, "window_step_ms" : 60, "fs" : 100}}
}

preprocess_ops = {key : preprocess_config[key]["enable"] for key in preprocess_config.keys()}
preprocess_params = {key : preprocess_config[key]["params"] for key in preprocess_config.keys()}



# AUGMENTATION
augmentation_config = {
    "AWGN" : {"enable" : True ,
              "params" : {"snr_db" : 25}},

    "FLIP" : {"enable" : False ,
              "params" : None}
}

augmentation_ops = {key : augmentation_config[key]["enable"] for key in augmentation_config.keys()}
augmentation_params = {key : augmentation_config[key]["params"] for key in augmentation_config.keys()}


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
        config_dict = {"ops" : preprocess_ops, "params" : preprocess_params}
        full_filename = get_config_full_filename(mode='preproc', name=filename)
        path = os.path.join(constants.DATA_CONFIG_PATH_PREPROC, full_filename)

    elif mode == "aug" :
        config_dict = {"ops": augmentation_ops, "params": augmentation_params}
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