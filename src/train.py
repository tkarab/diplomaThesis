import json
import os
import time

import custom_models
import fsl_functions
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


"""
    DESCRIPTION
    Determines the name of the model based on the number of existing models of the same type.
    i.e. it could be a protoNet with different backbone each time. It creates a file with the correct enumeration
    based on the number of existing models of that type.

    For example if there are already models 'model_protoNet_1.h5' and 'model_protoNet_2.h5' the file will be named
    'model_protoNet_3.h5' etc. If there are none the number will be set to '1'

    To provide mode info for the model and the training process in general a .txt file wll be provided
"""


def get_checkpoint_filename(dir_path, model_name):
    name = f"model_{model_name}_1.h5"
    if name in os.listdir(dir_path):
        num = int((name.split('.')[0]).split('_')[-1])
        name = f"model_{model_name}_{num + 1}.h5"
    return name


class IterationLoggingCallback(keras.callbacks.Callback):
    # def on_batch_end(self, batch, logs=None):
    #     if (batch % 100) == 0:
    #         # print(f"Batch {batch + 1}: loss = {logs.get('loss'):.2f}\n")
    #         print()
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # print('win_size: ', win_size)

class TrainingInfoCallback(keras.callbacks.Callback):
    def __init__(self, file_path, model,  batch_size, model_filename, model_backbone_name, experiment, iterations_per_epoch, preprocessing_dict, aug_dict):
        super(TrainingInfoCallback, self).__init__()
        self.file_path = file_path
        self.model = model
        self.batch_size = batch_size
        self.filename = model_filename.split('.')[0] + "_training_info.json"
        self.model_backbone_name = model_backbone_name
        self.experiment = experiment
        self.iterations_per_epoch = iterations_per_epoch
        self.preprocessing_dict = preprocessing_dict
        self.aug_dict = aug_dict

    def on_epoch_end(self, epoch, logs=None):
        # Create a dictionary with training info
        training_info = {
            "MODEL" : {
                "NAME" : self.model.name,
                "BASE"  : self.model_backbone_name,
                "ARCHITECTURE_INFO" : json.loads(self.model.to_json())
            },
            "PROCESSING" : {
                "PREPROCESSING" : self.preprocessing_dict,
                "AUGMENTATION" : self.aug_dict
            },
            "TRAINING_INFO" : {
                "EXPERIMENT" : self.experiment,
                "BATCH_SIZE" : self.batch_size,
                "ITERATIONS_PER_EPOCH" : self.iterations_per_epoch,
                "EPOCH" : epoch,
                "OPTIMIZER" : self.model.optimizer,
                "LEARNING_RATE" : None
            },
            "RESULTS" : {

            }
        }


        # Save the dictionary as a JSON file
        with open(os.path.join(self.file_path,self.filename), 'w') as file:
            json.dump(training_info, file, indent=4)

        print(f"Training information saved to {self.file_path}")


validation_steps = 1000
training_steps = 40
batch_size = 32
epochs = 11
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
cnn_backbone = custom_models.AtzoriNetDB2_embedding_only(input_shape=inp_shape, add_dropout=True, add_regularizer=True)


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
checkpointPath = RESULTS_DIRECTORIES_DICT[ex]
# save_weights_only=False,
checkpointCallBack = ModelCheckpoint(os.path.join(checkpointPath,f"model_{model.name}.h5"),  save_best_only=True, monitor='val_loss', mode='min')
checkpointCallBack.set_model(model)


def save_history(filepath,history):
    history_dict = history.history
    with open(filepath, 'w') as f:
        for key, values in history_dict.items():
            f.write(f"{key}:\n")
            for epoch, value in enumerate(values, 1):
                f.write(f"  Epoch {epoch}: {value}\n")
            f.write("\n")



for epoch_num in range(epochs):
    print(f"\nEpoch {epoch_num+1:2d}/{epochs}")
    # training for 1 epoch
    history = model.fit(data_loader, epochs=4, shuffle=False, callbacks=[iterationLoggingCallback])

    # validation
    data_loader.setMode('test')
    data_loader.set_iterations_per_epoch(validation_steps)
    data_loader.set_batch_size(1)
    data_loader.set_aug_enabled(False)
    val_loss, val_accuracy = model.evaluate(data_loader)

    # passing val_accuracy and val_loss into logs variable for monitoring during training
    logs = {"val_accuracy" : val_accuracy, "val_loss" : val_loss}
    checkpointCallBack.on_epoch_end(epoch_num,logs)
    save_history(os.path.join(checkpointPath,'history.txt'),history)

    data_loader.setMode('train')
    data_loader.set_iterations_per_epoch(training_steps)
    data_loader.set_batch_size(batch_size)
    data_loader.set_aug_enabled(aug_enabled)


print("END")


