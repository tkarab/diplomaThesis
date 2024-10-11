import json
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from task_generator import TaskGenerator
import numpy as np

from model_assembly import *
from helper_functions import *
from constants import *
from custom_models import *
from custom_callbacks import *
from fsl_functions import *
from flags import *

validation_steps = 1000
training_steps = 1000
starting_epoch = 0
batch_size = 32
epochs = 0
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate)
# loss_function = 'categorical_crossentropy'
loss_function = 'binary_crossentropy'
# metrics = ['categorical_accuracy']
metrics = ['binary_accuracy']

#input shape tuple
inp_shape_5d = (None,) + inp_shape

# DB and rms
db = 2
rms = 100

# experiment, way, shot
ex = '1'
N = 5
k = 5

# Results
best_val_loss = float('inf')
best_val_accuracy = 0.0


cnn_backbone = AtzoriNetDB2_embedding_only_extra_layers_added(input_shape=inp_shape, add_dropout=True, add_regularizer=True)

resultsPath = os.path.join(RESULTS_DIRECTORIES_DICT[ex], get_results_dir_fullpath(ex, N, k))

print("Creating new model...\n")
# model = assemble_protonet_reshape_with_batch(cnn_backbone, inp_shape, way=N, shot=k)
model = SiameseNetwork(cnn_backbone=cnn_backbone, f=l1_dist, inp_shape=inp_shape, dense_layers=get_dense_layers(neurons_per_layer=[]))
model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
model_foldername = get_checkpoint_foldername(resultsPath, model.name)
print("Name:",model.name,'\n')

#Results
resultsPath = os.path.join(resultsPath, model_foldername)
os.mkdir(resultsPath)
checkpoint_latest_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='latest'))
checkpoint_best_acc_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='best_acc'))
checkpoint_best_loss_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='best_loss'))

print(f"...model saved at '{resultsPath}'")


preproc_config = get_config_from_json_file('preproc', "db2_no_discard_lpf_muLaw_min_max")
aug_enabled = True
aug_config = get_config_from_json_file('aug', 'db2_awgn_snr25')
data_intake = 'generate'
# network_type = "protoNet"
network_type = "siamNet"


train_loader = TaskGenerator(network_type=network_type, experiment=ex, way=N, shot=k, mode='train', data_intake=data_intake, database=db, preprocessing_config=preproc_config, aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms, batch_size=batch_size, batches=training_steps)

# Getting 1 output from train loader to test dimensions etc
[x,y], label = train_loader[0]



