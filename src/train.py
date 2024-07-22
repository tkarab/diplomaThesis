import os
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

iterations_per_epoch = 1000
epochs = 25
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

aug_enabled = True
aug_config = helper_functions.get_config_from_json_file('aug', 'db2_awgn_snr25')

train_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train',database=db, preprocessing_config=preproc_config, aug_enabled=aug_enabled, aug_config=aug_config, rms_win_size=rms, batch_size=32, batches=iterations_per_epoch, print_labels=True, print_labels_frequency=5)

[x,y], label = train_loader[0]

print("END")
callback = IterationLoggingCallback()
model2.fit(train_loader, epochs=epochs,   shuffle=False, callbacks=[callback])


