import os
import time

import custom_models
import fsl_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from task_generator import TaskGenerator
import numpy as np
import model_assembly
import helper_functions

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

class ModeSwitchingCallback(keras.callbacks.Callback):
    def __init__(self, taskGenerator:TaskGenerator, val_frequency:int = 1):
        super(ModeSwitchingCallback, self).__init__()
        self.taskGenerator = taskGenerator
        self.val_frequency = val_frequency

    def on_epoch_begin(self, epoch, logs=None):
        # Set the mode of the sequence object at the beginning of each epoch
        if self.taskGenerator.getMode() == 'test':
            self.taskGenerator.setMode('train')
            self.taskGenerator.set_batch_size(batch_size)
            self.taskGenerator.batches_per_epoch = training_steps

            return

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % (self.val_frequency) == 0.0 :
            if self.taskGenerator.getMode() == 'train':
                self.taskGenerator.setMode('test')
                self.taskGenerator.set_batch_size(1)
                # self.taskGenerator.batches_per_epoch = 20
                model2.evaluate(self.taskGenerator,steps=validation_steps)

        return

validation_steps = 1000
training_steps = 1000
batch_size = 64
epochs = 11
win_size = 15
channels = 12
inp_shape = (win_size,channels,1)
cnn_backbone = custom_models.AtzoriNetDB2_embedding_only(input_shape=inp_shape, add_dropout=True, add_regularizer=True)
                            #simplest_conv_net_1_layer(input_shape = inp_shape, feature_vector_size=12)


#input shape tuple
inp_shape_5d = (None,) + inp_shape

ex = '1'
N = 5
k = 5

#model
# model = keras.Model(inputs=[support_set_inp_shape_layer,query_set_inp_shape_layer], outputs=query_prediction_layer)
model = model_assembly.assemble_protonet_timeDist(cnn_backbone, inp_shape)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['categorical_accuracy'])#, run_eagerly=True)

model2 = model_assembly.assemble_protonet_reshape_with_batch(cnn_backbone, inp_shape, way=N, shot=k)
model2.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['categorical_accuracy'])


db = 2
rms = 100
preproc_config = helper_functions.get_config_from_json_file('preproc', 'db2_lpf_minmax')

aug_enabled = False
aug_config = helper_functions.get_config_from_json_file('aug', 'db2_awgn_snr25')
data_intake = 'generate'



train_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', data_intake=data_intake, database=db, preprocessing_config=preproc_config, aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms, batch_size=batch_size, batches=training_steps, print_labels=True, print_labels_frequency=5)

[x,y], label = train_loader[0]

iterationLoggingCallback = IterationLoggingCallback()
modeChangingCallback = ModeSwitchingCallback(taskGenerator=train_loader, val_frequency=1)
# model2.fit(train_loader, epochs=epochs,   shuffle=False, callbacks=[callback])
# train_loader.set_batch_size(32)
for i in range(epochs):
    print(f"\nEpoch {i+1:2d}/{epochs}")
    model2.fit(train_loader, epochs=1, shuffle=False, callbacks=[iterationLoggingCallback])

    train_loader.setMode('test')
    train_loader.set_iterations_per_epoch(validation_steps)
    train_loader.set_batch_size(1)
    train_loader.set_aug_enabled(False)
    model2.evaluate(train_loader)

    train_loader.setMode('train')
    train_loader.set_iterations_per_epoch(training_steps)
    train_loader.set_batch_size(batch_size)
    train_loader.set_aug_enabled(aug_enabled)


print("END")


