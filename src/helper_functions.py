import re
import numpy as np

def get_unique_sgr(keylist):
    pattern = r'\d+\.?\d*'
    sgr_list = np.array([[int(i) for i in re.findall(pattern, key)] for key in keylist])
    s = np.unique(sgr_list[:,0])
    g = np.unique(sgr_list[:,1])
    r = np.unique(sgr_list[:,2])

    return s,g,r
