import re
import numpy as np

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