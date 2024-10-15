import keras.optimizers
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

batches = 100
batch_size = 5
inp_shape = (15,12,1)
x1 = tf.random.uniform(minval=0, maxval=5, shape=(batches,batch_size,)+inp_shape)
x2 = tf.random.uniform(minval=0, maxval=5, shape=(batches,batch_size,)+inp_shape)
labels = np.random.randint(0,2,[batches,batch_size])

path_to_cnn = 'C:\\Users\\ΤΑΣΟΣ\\Desktop\\Σχολή\\Διπλωματική\\Δεδομένα\\Results\\Experiment 1\\5_way_5_shot\\model_siamese_network_6\\model_siamese_network_6.h5'
cnn1 = keras.models.load_model(path_to_cnn)

path_to_dense = 'C:\\Users\\ΤΑΣΟΣ\\Desktop\\Σχολή\\Διπλωματική\\Δεδομένα\\Results\\Experiment 1\\5_way_5_shot\\model_siamese_network_6\\model_siamese_network_6dense_layers.h5'
dense_layers = keras.models.load_model(path_to_dense)

siam_net = SiameseNetwork(cnn_backbone=cnn1, f=l1_dist, inp_shape=inp_shape, dense_layers=dense_layers)

siam_net.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])

out = siam_net([x1[0],x2[0]])

inp = [[x1[i],x2[i]] for i in range(batches)]

siam_net.fit(x=inp, y=labels, epochs=2)

print()