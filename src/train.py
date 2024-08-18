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



class IterationLoggingCallback(keras.callbacks.Callback):
    # def on_batch_end(self, batch, logs=None):
    #     if (batch % 100) == 0:
    #         # print(f"Batch {batch + 1}: loss = {logs.get('loss'):.2f}\n")
    #         print()
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # print('win_size: ', win_size)

class TrainingInfoCallback(keras.callbacks.Callback):
    def __init__(self, file_path, model,  batch_size, model_filename, model_backbone_name, experiment, iterations_per_epoch, preprocessing_dict, aug_dict, best_epoch_kept = 0):
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
        self.best_epoch_kept = best_epoch_kept

    def round_results(self,logs):
        return {key: round(value, 2) for key, value in logs.items()}

    # logs has the following form: {'train_loss':1.2, 'train_accuracy':0.6, 'val_loss':1.4, 'val_accuracy':0.5}
    def on_epoch_end(self, epoch, logs=None):
        if self.filename in os.listdir(self.file_path):
            with open(os.path.join(self.file_path,self.filename), 'r') as f:
                training_info = json.load(f)
            training_info["RESULTS"][f"epoch {epoch+1}"] = self.round_results(logs)
            training_info["TRAINING_INFO"]["TOTAL_EPOCHS"] += 1
            training_info["TRAINING_INFO"]["BEST_EPOCH_KEPT"] = self.best_epoch_kept


        else:
            # Create a dictionary with training info
            training_info = {
                "MODEL" : {
                    "NAME" : self.model.name,
                    "BASE" : self.model_backbone_name
                },
                "PROCESSING" : {
                    "PREPROCESSING" : self.preprocessing_dict,
                    "AUGMENTATION"  : self.aug_dict
                },
                "TRAINING_INFO" : {
                    "EXPERIMENT" : self.experiment,
                    "BATCH_SIZE" : self.batch_size,
                    "ITERATIONS_PER_EPOCH" : self.iterations_per_epoch,
                    "OPTIMIZER" : self.model.optimizer._name,
                    "LEARNING_RATE" : float(self.model.optimizer.learning_rate.numpy()),
                    "TOTAL_EPOCHS" : 1,
                    "BEST_EPOCH_KEPT" : self.best_epoch_kept
                },
                "RESULTS" : {
                    "epoch 1" : self.round_results(logs)
                }
            }


        # Save the dictionary as a JSON file
        with open(os.path.join(self.file_path,self.filename), 'w') as file:
            json.dump(training_info, file, indent=4)



validation_steps = 1000
training_steps = 500
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


