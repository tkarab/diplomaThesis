import numpy as np
import pandas as pd
import os
import sys
import random
import tensorflow as tf
import fsl_functions
# import custom_models
import constants
import time
import helper_functions
from plot_functions import *


data_path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\db2_processed.npz'

# data = np.load(data_path)
# total_keys = data.files
N = 4

# keys = random.sample(total_keys,N)

# print("Keys chosen:",keys)
# signals = {key :data[key] for key in keys}
# channels = signals[random.choice(keys)].shape[1]
# max_length = np.max([signal.shape[0] for signal in signals.values()])


channels = 3
max_length = 10
columns = [f's{i+1}c{j+1}' for i in range(N) for j in range(channels)]
df = pd.DataFrame(columns=columns)

for i in range(N):
    start = 10 + max_length*i
    a = np.tile(np.arange(start,start+max_length),[channels,1]).T
    keys = columns[i*channels:(i+1)*channels]
    df[keys] = a

keys = [list(keyset) for keyset in np.reshape(np.array(columns),[N,channels])]
print()









