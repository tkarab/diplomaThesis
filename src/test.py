import numpy as np
import pandas as pd
import os
import sys
import random
import tensorflow as tf
import fsl_functions
import custom_models
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

cnn1 = custom_models.AtzoriNetDB2_embedding_only()
cnn2 = custom_models.AtzoriNetDB2_embedding_only_extra_layers_added(extra_layers=3)
print(cnn1.summary())
print(cnn2.summary())
print()










