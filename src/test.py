import numpy as np
# import pandas as pd
import os
import sys
import random
import fsl_functions
from custom_models import *
from model_assembly import *
import constants
import time
from helper_functions import *
from plot_functions import *
from preprocessing import *
from matplotlib import pyplot as plt
import tensorflow as tf


data_path = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\processed\db2\db2_processed.npz'

batch_size = 5
inp_shape = (15,12,1)
x1 = tf.random.uniform(minval=0, maxval=5, shape=(batch_size,)+inp_shape)
x2 = tf.random.uniform(minval=0, maxval=5, shape=(batch_size,)+inp_shape)

cnn1 = AtzoriNetDB2_embedding_only_extra_layers_added()

siam_net = assemble_siamNet(cnn1, f=fsl_functions.l1_dist, input_shape=inp_shape)

out = siam_net([x1,x2])

print()