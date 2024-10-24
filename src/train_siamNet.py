import json
import os
import time
from typing_extensions import deprecated

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from task_generator import TaskGenerator
import numpy as np

from model_assembly import *
from helper_functions import *
from constants import *
from custom_models import *
from custom_callbacks import *
from fsl_functions import *
from flags import *

"""
PARAMETERS
    epoch_num, logs, history are all global params
"""
def run_callbacks():
    logs = {"val_accuracy": val_accuracy, "val_loss": val_loss}
    early_stopping_mode_on = False

    if SAVE_MODEL == True:
        if CHECKPOINT_LATEST_ENABLED:    #TODO
            pass
        if CHECKPOINT_BEST_LOSS_ENABLED: #TODO
            pass
        if CHECKPOINT_BEST_ACC_ENABLED:  #TODO
            pass
        if SAVE_TRAIN_STATS_ENABLED:     #TODO
            train_results = dict(
                **{"train_accuracy": history.history['categorical_accuracy'][0], 'train_loss': history.history['loss'][0]},
                **logs)
            # trainingInfoCallback.on_epoch_end(epoch=epoch_num, logs=train_results)
    if PLOT_RESULTS_ENABLED:
        resultsPlotCallback.on_epoch_end(logs=dict(
            **{"train_accuracy": history.history['binary_accuracy'][0], 'train_loss': history.history['loss'][0]},
            **logs), epoch=epoch_num)

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
    print("Validation")
    val_loss, val_accuracy = model_val.evaluate(data_loader)

    return val_loss, val_accuracy


def evaluate_model2(model:SiameseNetwork, data_loader, validation_steps, N):
    # model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['categorical_accuracy'])
    inp_support_set = layers.Input(shape=(None,) + inp_shape)
    feature_extractor = model.feature_extractor
    feature_extractor_timeDist = keras.Model(inputs = inp_support_set, outputs=TimeDistributed(feature_extractor)(inp_support_set))

    dense_layers = model.dense_layers
    f = model.f

    correct_predictions = 0

    accuracy = 0.0

    progress_bar = tqdm(total=validation_steps, desc="Validation", unit=" iterations")

    for i in range(validation_steps):
        [support_set, query_image], labels = data_loader[i]
        # This is done because the support and the query image are taken in batches of 1
        support_set = support_set[0]
        query_image = query_image[0]

        # This produces a single L-long vector (where L is the chosen embedding shape for the feature extractor ie 64)
        query_embedding = feature_extractor(query_image)
        # We copy the embedding N times (as many as the classes of the suport set)
        # Input shape : (1,L) -> Output shape (N,L)
        query_embedding_copied = tf.tile(query_embedding, [N,1])

        # This produces a N,k set of L-long vectors
        support_embeddings = feature_extractor_timeDist(support_set)
        # Computes the prototypes of the classes that will go through the distance function
        # Input shape: (N,k,L) -> Output shape : (N,L)
        class_prototypes = produce_prototype(support_embeddings)

        # The distance function applied to the two (N,L) sets to produce a single output of N vectors of length L
        out_dist = f([query_embedding_copied, class_prototypes])

        # Final prediction after passing the output of the distance function through the dense layers
        pred = dense_layers(out_dist)

        if np.argmax(pred) == np.argmax(labels):
            correct_predictions += 1

        accuracy = correct_predictions/(i+1)
        progress_bar.set_postfix(accuracy=f"{accuracy:.4f}")
        progress_bar.update(1)  # Update progress bar by 1

    # progress_bar.close()


    return


training_steps = 100
validation_steps = 1000
starting_epoch = 0
batch_size = 128
epochs = 10
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate)
# loss_function = 'categorical_crossentropy'
loss_function = 'binary_crossentropy'
# metrics = ['categorical_accuracy']
metrics = ['binary_accuracy']


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

# LR Scheduler Parameters
reduction_factor = 0.5
patience = 3
cooldown_patience = 2
min_lr = 1e-4
min_delta = 0.001


cnn_backbone = AtzoriNetDB2_embedding_only_extra_layers_added(input_shape=inp_shape, add_dropout=False, add_regularizer=False)

resultsPath = os.path.join(RESULTS_DIRECTORIES_DICT[ex], get_results_dir_fullpath(ex, N, k))

print("Creating new model...\n")
# model = assemble_protonet_reshape_with_batch(cnn_backbone, inp_shape, way=N, shot=k)
model = SiameseNetwork(cnn_backbone=cnn_backbone, f=l2_dist, inp_shape=inp_shape, dense_layers=get_dense_layers(neurons_per_layer=[]))
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])
model_foldername = get_checkpoint_foldername(resultsPath, model.name)

#Results
resultsPath = os.path.join(resultsPath, model_foldername)

if SAVE_MODEL == True:
    # os.mkdir(resultsPath)
    pass

checkpoint_latest_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='latest'))
checkpoint_best_acc_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='best_acc'))
checkpoint_best_loss_path = os.path.join(resultsPath, get_model_checkpoint_fullname(model_foldername, criterion='best_loss'))

# print(f"...model saved at '{resultsPath}'")


preproc_config = get_config_from_json_file('preproc', "db2_no_discard_lpf_muLaw_min_max")
aug_enabled = True
aug_config = get_config_from_json_file('aug', 'db2_awgn_snr25')
data_intake = 'generate'
# network_type = "protoNet"
network_type = "siamNet"

if ex == "1":
    train_loader = TaskGenerator(network_type=network_type, experiment=ex, way=N, shot=k, mode='train',
                                 data_intake=data_intake, database=db, preprocessing_config=preproc_config,
                                 aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms,
                                 batch_size=batch_size, batches=training_steps)
    val_loader = TaskGenerator(network_type='protoNet', experiment=ex, way=N, shot=k, mode='test',
                                data_intake='generate', database=db, preprocessing_config=preproc_config,
                                aug_enabled=False, aug_config=aug_config, rms_win_size=rms, batch_size=1,
                                batches=validation_steps)

elif ex.startswith("2"):
    train_loader = TaskGenerator(network_type=network_type, experiment=ex, way=N, shot=k, mode='train',
                                 data_intake=data_intake, database=db, preprocessing_config=preproc_config,
                                 aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms,
                                 batch_size=batch_size, batches=training_steps)
    val_loader = TaskGenerator(network_type='protoNet', experiment="2a", way=N, shot=k, mode='val',
                                data_intake='generate', database=db, preprocessing_config=preproc_config,
                                aug_enabled=False, aug_config=aug_config, rms_win_size=rms, batch_size=1,
                                batches=validation_steps)
    test_loader = TaskGenerator(network_type='protoNet', experiment="2a", way=N, shot=k, mode='test',
                                data_intake='generate', database=db, preprocessing_config=preproc_config,
                                aug_enabled=False, aug_config=aug_config, rms_win_size=rms, batch_size=1,
                                batches=validation_steps)

elif ex == "3":
    train_loader = TaskGenerator(network_type=network_type, experiment=ex, way=N, shot=k, mode='train',
                                 data_intake=data_intake, database=db, preprocessing_config=preproc_config,
                                 aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms,
                                 batch_size=batch_size, batches=training_steps)
    val_loader = TaskGenerator(network_type='protoNet', experiment="2a", way=N, shot=k, mode='val',
                               data_intake='generate', database=db, preprocessing_config=preproc_config,
                               aug_enabled=False, aug_config=aug_config, rms_win_size=rms, batch_size=1,
                               batches=validation_steps)
    test_loader = TaskGenerator(network_type='protoNet', experiment="2a", way=N, shot=k, mode='test',
                                data_intake='generate', database=db, preprocessing_config=preproc_config,
                                aug_enabled=False, aug_config=aug_config, rms_win_size=rms, batch_size=1,
                                batches=validation_steps)
elif ex == "4":
    #TODO: fix for experiment 4
    pass


# Getting 1 output from train loader to test dimensions etc
[x,y], label = train_loader[0]
[x2,y2], label2 = val_loader[0]


# Custom Callbacks
# lr_adjustment_callback = ReduceLrOnPlateauCustom(model=model, criterion="loss", reduction_factor=reduction_factor, patience=patience,cooldown_patience=cooldown_patience,min_lr=min_lr, min_delta=min_delta, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy)
lr_adjustment_callback = ReduceLrSteadilyCustom(model=model, reduction_factor=reduction_factor,patience=patience,min_lr=min_lr)
# checkpointCallBack_val_acc = ModelCheckpoint(checkpoint_best_acc_path, save_best_only=True, monitor='val_accuracy', mode='min')
# checkpointCallBack_val_acc.set_model(model)
filenames_comp = [
    r"C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Results\Experiment 1\5_way_5_shot\test_embedding_size\SiamNet 3 extra layer l2dist disc1.5 rms100 emb128 0dense lpf no muLaw batch 128 results.txt",
    r"C:\Users\ΤΑΣΟΣ\Desktop\Σχολή\Διπλωματική\Δεδομένα\Results\Experiment 1\5_way_5_shot\test_embedding_size\SiamNet 3 extra layer l2dist disc1.5 rms100 emb256 0dense lpf no muLaw batch 128 results.txt"
]
labels_comp = ['128','256']

resultsPlotCallback = PlotResultsCallback(filenames=filenames_comp, labels=labels_comp, title='embedding size', metric='val_accuracy')

for epoch_num in range(starting_epoch, starting_epoch+epochs):
    print(f"\nepoch {epoch_num + 1}/{starting_epoch + epochs}")
    print("Training")
    history = model.fit(train_loader,epochs=1)
    val_loss, val_accuracy = evaluate_model(model, val_loader, N)
    run_callbacks()

# Testing the model
if ex == "1":
    evaluate_model(model, val_loader,N)
else:
    evaluate_model(model, test_loader,N)

print()



