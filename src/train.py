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

def keep_result_lines_until_best(filepath, epoch_to_keep_until):
    with open(filepath, 'r') as f_read:
        lines = f_read.readlines()
        lines_to_keep = lines[:2]
        for line in lines[2:]:
            if int(line[0]) <= epoch_to_keep_until: #TODO - Fix so that the whole number of the line is read
                lines_to_keep.append(line)

    with open(filepath, 'w') as f_write:
        for line in lines_to_keep:
            f_write.write(line)

    return

def get_training_config_from_json_file(json_filename):
    # training parameters
    global validation_steps
    global training_steps
    global batch_size

    # optimizer
    global optimizer
    global learning_rate
    global loss
    global metrics
    global starting_epoch

    # processing
    global preproc_config
    global aug_enabled
    global aug_config

    # data generator
    global data_intake
    global db
    global rms

    with open(json_filename) as f:
        info = json.load(f)
        # Processing
        preproc_config  = info["PROCESSING"]["PREPROCESSING"]
        aug_enabled     = info["PROCESSING"]["AUG_ENABLED"]
        aug_config      = info["PROCESSING"]["AUGMENTATION"]

        # Training Info
        training_steps   = info["TRAINING_INFO"]["ITERATIONS_PER_EPOCH"]
        validation_steps = info["TRAINING_INFO"]["VALIDATION_STEPS"]
        batch_size       = info["TRAINING_INFO"]["BATCH_SIZE"]
        optimizer_name   = info["TRAINING_INFO"]["OPTIMIZER"]
        learning_rate    = info["TRAINING_INFO"]["BEST_EPOCH_LEARNING_RATE"]
        loss             = info["TRAINING_INFO"]["LOSS"]
        metrics          = info["TRAINING_INFO"]["METRICS"]
        starting_epoch   = info["TRAINING_INFO"]["BEST_EPOCH_KEPT"]

        if 'loss' in metrics:
            metrics.remove('loss')

        # Data generator
        data_intake = info["DATA_GENERATOR"]["DATA_INTAKE"]
        db          = info["DATA_GENERATOR"]["DB"]
        rms         = info["DATA_GENERATOR"]["RMS"]

        if optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate)

    return

validation_steps = 1000
training_steps = 10
starting_epoch = 0
batch_size = 32
epochs = 30
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
cnn_backbone = AtzoriNetDB2_embedding_only(input_shape=inp_shape, add_dropout=True, add_regularizer=True)
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

#input shape tuple
inp_shape_5d = (None,) + inp_shape

preproc_config = get_config_from_json_file('preproc', 'db2_lpf_minmax')
aug_enabled = True
aug_config = get_config_from_json_file('aug', 'db2_awgn_snr25')
data_intake = 'generate'

# DB and rms
db = 2
rms = 100

# experiment, way, shot
ex = '1'
N = 5
k = 5

# Results
best_loss = float('inf')
best_accuracy = 0.0


model_name = 'model_protoNet_1'
resultsPath = os.path.join(RESULTS_DIRECTORIES_DICT[ex], get_results_dir_fullpath(ex, N, k))

if not LOAD_EXISTING_MODEL:
    print("Creating new model...\n")
    model = assemble_protonet_reshape_with_batch(cnn_backbone, inp_shape, way=N, shot=k)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model_foldername = get_checkpoint_foldername(resultsPath, model.name)
    print("Name:",model.name,'\n')
    model_filename = model_foldername + '.h5'
    resultsPath = os.path.join(resultsPath, model_foldername)
    os.mkdir(resultsPath)
    model_fullpath = os.path.join(resultsPath, model_filename)
    print(f"...model saved at '{model_fullpath}'")

else:
    print("Loading existing model...\n")
    print(f"Name: {model_name}.h5\n")
    model_foldername = model_name
    model_filename = model_foldername + '.h5'
    resultsPath = os.path.join(resultsPath, model_foldername)
    model_fullpath = os.path.join(resultsPath, model_filename)
    get_training_config_from_json_file(os.path.join(resultsPath,model_foldername + "_training_info.json"))
    keep_result_lines_until_best(filepath=os.path.join(resultsPath,model_foldername + "_results.txt"),epoch_to_keep_until=starting_epoch)
    print(f"...model loaded. Resuming training from epoch {starting_epoch}")

    model = keras.models.load_model(model_fullpath)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

data_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', data_intake=data_intake, database=db, preprocessing_config=preproc_config, aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms, batch_size=batch_size, batches=training_steps, print_labels=True, print_labels_frequency=5)

# Getting 1 output from train loader to test dimensions etc
[x,y], label = data_loader[0]

# Callbacks
# Logger
iterationLoggingCallback = IterationLoggingCallback()

# Checkpoint
checkpointCallBack = ModelCheckpoint(model_fullpath, save_best_only=True, monitor='val_loss', mode='min')
checkpointCallBack.set_model(model)

# Training info
trainingInfoCallback = TrainingInfoCallback(resultsPath, model, batch_size, model_foldername, cnn_backbone.name, ex, training_steps, validation_steps, preproc_config, aug_enabled, aug_config, data_intake, rms, db)

# LR Adjustment
reduction_factor = 0.5
patience = 2
cooldown_patience = 2
min_lr = 1e-4
min_delta = 0.001
#lr_adjustment_callback = ReduceLrOnPlateauCustom(model=model, reduction_factor=reduction_factor, patience=patience,cooldown_patience=cooldown_patience,min_lr=min_lr, min_delta=min_delta, best_val_loss=best_loss)
lr_adjustment_callback = ReduceLrSteadilyCustom(model=model, reduction_factor=reduction_factor,patience=patience,min_lr=min_lr)


early_stopping_mode_on = EARLY_STOPPING_ENABLED and (not LR_SCHEDULER_ENABLED)

for epoch_num in range(starting_epoch, starting_epoch+epochs):
    # training
    data_loader.setMode('train')
    data_loader.set_iterations_per_epoch(training_steps)
    data_loader.set_batch_size(batch_size)
    data_loader.set_aug_enabled(aug_enabled)
    print(f"\nEpoch {epoch_num+1:2d}/{starting_epoch+epochs}")

    # train for 1 epoch
    history = model.fit(data_loader, epochs=1, shuffle=False, callbacks=[iterationLoggingCallback])

    # validation
    data_loader.setMode('test')
    data_loader.set_iterations_per_epoch(validation_steps)
    data_loader.set_batch_size(1)
    data_loader.set_aug_enabled(False)
    print("Validation")
    val_loss, val_accuracy = model.evaluate(data_loader)
    logs = {"val_accuracy": val_accuracy, "val_loss": val_loss}

    # Model checkpoint: save model if it has the best performance
    if(val_loss < best_loss):
        checkpointCallBack.on_epoch_end(epoch_num,logs)
        best_loss = val_loss
        trainingInfoCallback.best_epoch_kept = epoch_num+1
        trainingInfoCallback.best_epoch_lr = float(model.optimizer.learning_rate.numpy())

    if(val_accuracy > best_accuracy):
        best_accuracy = val_accuracy

    # Write results
    train_results = dict(**{"train_accuracy" : history.history['categorical_accuracy'][0], 'train_loss' : history.history['loss'][0]},**logs)
    trainingInfoCallback.on_epoch_end(epoch=epoch_num, logs=train_results)

    if LR_SCHEDULER_ENABLED:
        min_lr_reached = lr_adjustment_callback.on_epoch_end(epoch_num,logs)

        if min_lr_reached and EARLY_STOPPING_ENABLED:
            early_stopping_mode_on = True

    if early_stopping_mode_on:
        continue



print("END")


