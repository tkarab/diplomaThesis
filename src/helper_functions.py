import os.path
import re
import numpy as np
import json
from constants import *

"""
DESCRIPTION
    Gets a list of keys (ie ['s1g1r3','s12g23r2','s12g40r1'] and returns the unique
    s,g,r values of all keys.
    In this example th unique subjects are 1,12, the unique gestures are 1,23,40
    and the unique reps are 1,2,3.
    Therefore the returning values are the following arrays:
    array([ 1, 12]), array([ 1, 23, 40]), array([1, 2, 3])
"""
def get_unique_sgr(keylist):
    pattern = r'\d+\.?\d*'
    sgr_list = np.array([[int(i) for i in re.findall(pattern, key)] for key in keylist])
    s = np.unique(sgr_list[:,0])
    g = np.unique(sgr_list[:,1])
    r = np.unique(sgr_list[:,2])

    return s,g,r

"""
Prints the keys of the support and query sets of a specific task in a formatted way
i.e. |  s1g1r3  s2g5r4  s12g7r3 |
     | s14g1r5  s5g5r1  s9g7r1  |

"""
def printKeys(keys):
    print()
    for j in range(len(keys[0])):
        print("| |", end='')
        for i in range(len(keys)):
            # Key takes up 8 cells of space
            print("{:<8}".format(keys[i][j]),end="| |")
        print()


"""
DESCRIPTION
    For changing the formatting of the s,g,r values into the key String so that each int takes
    up 2 digits regardless of value
    ie : 's1g12r4' -> 's01g12r04'
    That way keys can also be easily sorted

PARAMETERS
    Key in old format ('s1g12r4')
    
OUTPUT
    Newly formatted key ('s01g12r04')
"""
def reformat_key(key):
    pattern = r'\d+\.?\d*'
    s,g,r = [int(i) for i in re.findall(pattern, key)]
    new_key = f"s{s:02d}g{g:02d}r{r:02d}"
    return new_key

"""
DESCRIPTION
    For getting a key based on given s,g,r values using the new format (2 digits regardless of value)
"""
def getKey(s,g,r):
    return f"s{s:02d}g{g:02d}r{r:02d}"

"""
DESCRIPTION
    Returns the name of the directory containing data of a certain database (db2 mainly) which are
    rms rectified (created offline) with a certain window size (in ms) 
    i.e. 'db2_rms_100'
"""
def get_rms_rect_filename(db, win_size_ms):
    return f"db{db}_rms_{win_size_ms}.npz"


"""
DESCRIPTION
    For getting the full name of a configuration .json file (either for preprocessing, augmentation or training).
    depending on the mode the full filename is "config_{mode}_{filename}.json where filename usually includes info
    regarding the database if it is referring to a specific one

PARAMETERS
    mode : "preproc", "aug" or "train"
"""
def get_config_full_filename(mode, name):
    return f"config_{mode}_{name}.json"

"""
DESCRIPTION
    Loads and returns configuration data for either preprocessing, augmentation or training.
    Data is in the form of a jason file and the returning value is a dict
    
PARAMETERS
    mode : "preproc", "aug" or "train"
    filename : name of the file without the 'config_preproc' prefix of the '.json' postfix 
"""
def get_config_from_json_file(mode, filename):
    full_filename = get_config_full_filename(mode, filename)
    if mode == "preproc":
        dir_path = DATA_CONFIG_PATH_PREPROC
    elif mode == "aug" :
        dir_path = DATA_CONFIG_PATH_AUG
    elif mode == "train" :
        return None
    else:
        return None

    with open(os.path.join(dir_path,full_filename)) as file:
        config = json.load(file)
        return config


"""
DESCRIPTION
    Returns the full filename of the .csv file where all the tasks for a given experiment exist
    It is something along the lines of ex{experiment}_{N}way_{k}shot.csv
    For the full path the FileInfoProvider in TaskGenerator takes care of it

PARAMETERS
    - mode : 'train', 'test' or 'val'
"""
def get_tasks_filename(ex,N,k, mode):
    return f'ex{ex}_{N}way_{k}shot_{mode}.csv'