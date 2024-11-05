import json
import os
import time

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from task_generator_ex4 import TaskGeneratorEx4IntraSubject
import numpy as np

from model_assembly import *
from plot_functions import *
from helper_functions import *
from constants import *
from custom_models import *
from custom_callbacks import *
from fsl_functions import *
from flags import *


def run_callbacks():
    logs = {"val_accuracy": val_accuracy, "val_loss": val_loss}
    full_logs = dict(
        **{"train_accuracy": history.history['binary_accuracy'][0], 'train_loss': history.history['loss'][0]},
        **logs)
    early_stopping_mode_on = False

    if SAVE_MODEL == True:
        saveModelCallback.on_epoch_end(epoch=epoch_num, logs=full_logs)

        # if CHECKPOINT_LATEST_ENABLED:    #TODO
        #     pass
        # if CHECKPOINT_BEST_LOSS_ENABLED: #TODO
        #     pass
        # if CHECKPOINT_BEST_ACC_ENABLED:  #TODO
        #     pass
        # if SAVE_TRAIN_STATS_ENABLED:     #TODO
        #     pass
            # trainingInfoCallback.on_epoch_end(epoch=epoch_num, logs=full_logs)

    if PLOT_RESULTS_ENABLED:
        pass

    if LR_SCHEDULER_ENABLED:
        min_lr_reached = lr_adjustment_callback.on_epoch_end(epoch_num, logs)

        if min_lr_reached and EARLY_STOPPING_ENABLED:
            early_stopping_mode_on = True

    if EARLY_STOPPING_ENABLED and early_stopping_mode_on:
        pass


    return


def contrastive_loss(y_true, y_pred):
    # For the correct class (where y_true == 1), we want the similarity score to be close to 1
    positive_loss = y_true * K.square(1 - y_pred)

    # For the incorrect classes (where y_true == 0), we want the similarity score to be close to 0
    negative_loss = (1 - y_true) * K.square(y_pred)

    # Combine the positive and negative losses and average over all classes
    loss = K.mean(positive_loss + negative_loss)

    return loss

def evaluate_model(model:SiameseNetwork, data_loader, N):
    model_val = assemble_siamNet_for_few_shot_infernce(model=model, inp_shape=inp_shape, N=N)
    loss1 = 'categorical_crossentropy'
    loss2 = contrastive_loss
    model_val.compile(optimizer, loss=loss2, metrics=['categorical_accuracy'])
    val_loss, val_accuracy = model_val.evaluate(data_loader)

    return val_loss, val_accuracy


def test_siamNet(model:SiameseNetwork, test_loader ,k=10):
    accuracies = []
    print("\n")
    print(" ~~~ Testing ~~~ ")
    for i in range(k):
        print(f"\n{i+1}/{k}")
        val_loss, val_accuracy = evaluate_model(model, test_loader, N)
        accuracies.append(100*val_accuracy)

    mean = np.mean([accuracies])
    stdev = np.std(accuracies)
    best = np.max(accuracies)

    print(f"Test results: {mean:.2f} ± {stdev:.2f}% (best: {best:.2f}%)")

    return mean, stdev, best

training_steps = 100
validation_steps = 100
starting_epoch = 0
batch_size = 128
epochs = 5
win_size = 15
channels = 14
inp_shape = (win_size,channels,1)
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate)
# loss_function = 'categorical_crossentropy'
loss_function = 'binary_crossentropy'
# metrics = ['categorical_accuracy']
metrics = ['binary_accuracy']


# DB and rms

rms = 100

# experiment, way, shot
N = 5
k = 5

# LR Scheduler Parameters
reduction_factor = 0.5
patience = 3
cooldown_patience = 2
min_lr = 1e-4
min_delta = 0.001

cnn_backbone = AtzoriNetDB2_embedding_only_extra_layers_added(input_shape=inp_shape, add_dropout=False, add_regularizer=False)
dense_layers = get_dense_layers(neurons_per_layer=[])
model = SiameseNetwork(cnn_backbone=cnn_backbone, f=l2_dist, inp_shape=inp_shape, dense_layers=dense_layers)
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])

preproc_config = get_config_from_json_file('preproc', "db2_discard_1.5_lpf_minmax_no_muLaw")
aug_enabled = True
aug_config = get_config_from_json_file('aug', 'db2_awgn_snr25')
# network_type = "protoNet"
network_type = "siamNet"

subject = 1
d_t = (2,2)

train_loader = TaskGeneratorEx4IntraSubject(
    subject=subject,
    val_day_time=d_t,
    way=N,
    shot=k,
    mode='train',
    task_type= 'intra_session',
    network_type='siamNet',
    preprocessing_config=preproc_config,
    aug_enabled=True,
    aug_config=aug_config,
    batch_size=batch_size,
    batches=training_steps,
    rms_win_size=rms
)

val_loader = TaskGeneratorEx4IntraSubject(
    subject=subject,
    val_day_time=d_t,
    way=N,
    shot=k,
    mode='val',
    task_type= 'intra_session',
    network_type='protoNet',
    preprocessing_config=preproc_config,
    aug_enabled=False,
    aug_config=None,
    batch_size=1,
    batches=validation_steps,
    rms_win_size=rms
)

# Getting 1 output from train loader to test dimensions etc
[x,y], label = train_loader[0]
[x2,y2], label2 = val_loader[0]

lr_adjustment_callback = ReduceLrSteadilyCustom(model=model, reduction_factor=reduction_factor,patience=patience,min_lr=min_lr)

for epoch_num in range(starting_epoch, starting_epoch+epochs):
    print(f"\nepoch {epoch_num + 1}/{starting_epoch + epochs}")
    print("Training")
    history = model.fit(train_loader,epochs=1)
    val_loss, val_accuracy = evaluate_model(model, val_loader, N)
    run_callbacks()
