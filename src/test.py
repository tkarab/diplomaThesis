# import keras.optimizers
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



root = r'C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Results\Experiment 1\5_way_5_shot'
subdir = "test_discard"

# name = f"SiamNet 3 extra layer l2dist discNone lpf no muLaw batch 128 res"
# input_file = os.path.join(root, subdir, name + '.txt')
# output_file = os.path.join(root, subdir, name + "ults.txt")
# parse_training_results_to_txt(input_file, output_file)
# os.remove(input_file)

# disc_num_display = ['1.5', '2.5', '3.5', '4.5', 'None']
#
# train_accuracies = []
# val_accuracies = []
#
# for disc_num in disc_num_display:
#     name = f"SiamNet 3 extra layer l2dist disc{disc_num} lpf no muLaw batch 128 res"
#     # input_file = os.path.join(root, subdir, name + '.txt')
#     output_file =os.path.join(root, subdir, name + "ults.txt")
#     #parse_training_results_to_txt(input_file, output_file)
#     # os.remove(input_file)
#     results = extract_scores_from_txt(output_file)
#     train_accuracies.append(np.mean(results['train_accuracy']))
#     val_accuracies.append(np.mean(results['val_accuracy']))
#
# disc_num_display[-1] = 'All'
# # plot_accuracies_hist(train_accuracies=train_accuracies, val_accuracies=val_accuracies, xlabels=disc_num_display)
#
#
# files = [
#     "SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 1dense64 lpf no muLaw batch 128 res",
#     "SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 1dense128 lpf no muLaw batch 128 res",
#     "SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 2dense64_32 lpf no muLaw batch 128 res",
#     "SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 2dense128_64 lpf no muLaw batch 128 res"
# ]
#
# subdir = 'test_dense'
#
# for name in files:
#     input_file = os.path.join(root, subdir, name + '.txt')
#     output_file =os.path.join(root, subdir, name + "ults.txt")
#     # plot_train_results(output_file, metric='val_accuracy', label=rms,title="RMS Window size")
#     # parse_training_results_to_txt(input_file, output_file)
#     # os.remove(input_file)
#
# print()
#
# root = r"C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Results\Experiment 1\5_way_5_shot\test_dense"
#
# files = [
#     ("SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 0 dense lpf no muLaw batch 128 results.txt", "no extra layers"),
#     ("SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 1dense64 lpf no muLaw batch 128 results.txt","64"),
#     ("SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 1dense128 lpf no muLaw batch 128 results.txt","128"),
#     ("SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 2dense64_32 lpf no muLaw batch 128 results.txt","62,32"),
#     ("SiamNet 3 extra layer l2dist disc1.5 rms100 emb64 2dense128_64 lpf no muLaw batch 128 results.txt","128,64")
# ]
#
# for filename, neuron_number in files:
#     filename = os.path.join(root, filename)
#     plot_train_results(input_file=filename, label=neuron_number, title="dense layers added", metric="val_loss")
#
# print()

inp = (15,12,1)
cnn_backbone =  AtzoriNetDB2_embedding_only_extra_layers_added(input_shape=inp)
siamnetTest = SiameseNetwork(cnn_backbone=cnn_backbone, inp_shape=inp, f=l2_dist, dense_layers=get_dense_layers([]))
x = tf.random.uniform(shape=(1,)+inp, minval=0, maxval=1)
y = tf.random.uniform(shape=(1,)+inp, minval=0, maxval=1)

siamnetTest([x,y])

