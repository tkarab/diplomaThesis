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

def print_array(array, name:str):
    print(f'\t-- {name}\n')
    try:
        array = array.numpy()
    except AttributeError:
        pass
    shape = array.shape
    l = len(shape)
    if l==1:
        print(array,'\n\n')
        return

    elif l == 2:
        array = np.expand_dims(array,axis=1)
        shape = array.shape

    cols = shape[0]

    for j in range(shape[1]):
        for i in range(cols):
            print(array[i][j], end="\t\t")
        print()

    print('\n\n')
    return







validation_steps = 1000
training_steps = 10
batch_size = 32
epochs = 11
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
cnn_backbone = AtzoriNetDB2_embedding_only(input_shape=inp_shape, add_dropout=True, add_regularizer=True)


#input shape tuple
inp_shape_5d = (None,) + inp_shape

ex = '1'
N = 5
k = 5


model = assemble_protonet_reshape_with_batch(cnn_backbone, inp_shape, way=N, shot=k)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['categorical_accuracy'])


db = 2
rms = 100

preproc_config = get_config_from_json_file('preproc', 'db2_lpf_minmax')

aug_enabled = True
aug_config = get_config_from_json_file('aug', 'db2_awgn_snr25')
data_intake = 'generate'


data_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', data_intake=data_intake, database=db, preprocessing_config=preproc_config, aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms, batch_size=batch_size, batches=training_steps, print_labels=True, print_labels_frequency=5)

# Getting 1 output from train loader to test dimensions etc
[x,y], label = data_loader[0]

# Callbacks

iterationLoggingCallback = IterationLoggingCallback()
checkpointPath = os.path.join(RESULTS_DIRECTORIES_DICT[ex],get_results_dir_fullpath(ex,N,k))
# save_weights_only=False,
model_filename = get_checkpoint_filename(checkpointPath, model.name)
checkpointCallBack = ModelCheckpoint(os.path.join(checkpointPath,model_filename),  save_best_only=True, monitor='val_loss', mode='min')
checkpointCallBack.set_model(model)

trainingInfoCallback = TrainingInfoCallback(checkpointPath,model,batch_size,model_filename,cnn_backbone.name,ex,training_steps,preproc_config,aug_config)


best_loss = float('inf')
best_accuracy = 0.0

for epoch_num in range(epochs):
    # training
    data_loader.setMode('train')
    data_loader.set_iterations_per_epoch(training_steps)
    data_loader.set_batch_size(batch_size)
    data_loader.set_aug_enabled(aug_enabled)
    print(f"\nEpoch {epoch_num+1:2d}/{epochs}")

    # training for 1 epoch
    history = model.fit(data_loader, epochs=1, shuffle=False, callbacks=[iterationLoggingCallback])

    # validation
    data_loader.setMode('test')
    data_loader.set_iterations_per_epoch(validation_steps)
    data_loader.set_batch_size(1)
    data_loader.set_aug_enabled(False)

    print("Validation")
    val_loss, val_accuracy = model.evaluate(data_loader)
    logs = {"val_accuracy": val_accuracy, "val_loss": val_loss}

    # save model if it has the best performance
    if(val_loss < best_loss):
        checkpointCallBack.on_epoch_end(epoch_num,logs)
        best_loss = val_loss
        trainingInfoCallback.best_epoch_kept = epoch_num+1

    if(val_accuracy > best_accuracy):
        best_accuracy = val_accuracy

    train_results = dict(**{"train_accuracy" : history.history['categorical_accuracy'][0], 'train_loss' : history.history['loss'][0]},**logs)
    trainingInfoCallback.on_epoch_end(epoch=epoch_num, logs=train_results)


print("END")


