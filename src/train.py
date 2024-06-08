import os
import custom_models
import fsl_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from task_generator import TaskGenerator
import numpy as np

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

def get_prototypes_distances_for_small_set(test_N=3, test_k=2, test_dim=5):
    test_support_set = np.array([[[1, 3, 2, 0], [1, 5, 2, 9]], [[1, 6, 8, 3], [1, 43, 2, 5]], [[5, 1, 7, 8], [2, 6, 5, 1]]])
    test_query = np.array([1, 4, 2, 5], dtype=np.float32)
    test_inp_layer = layers.Input(shape=test_support_set.shape[1:])
    test_query_inp = layers.Input(shape=(4))
    test_prototype_no_cnn = Lambda(function=fsl_functions.produce_prototype)(test_inp_layer)
    test_distances_no_cnn = Lambda(function=fsl_functions.euc_dist)([test_prototype_no_cnn, test_query_inp])
    model_prototypes_no_cnn = keras.Model(inputs=test_inp_layer, outputs=test_prototype_no_cnn)
    model_distances_no_cnn = keras.Model(inputs=[model_prototypes_no_cnn.output, test_query_inp],
                                         outputs=test_distances_no_cnn)

    out_prototypes_no_cnn = model_prototypes_no_cnn.predict(test_support_set)
    test_query = np.expand_dims(test_query, axis=0)
    out_distances = fsl_functions.euc_dist([out_prototypes_no_cnn, test_query])
    out_prediction = fsl_functions.softmax_classification([out_prototypes_no_cnn, test_query])

    print_array(test_support_set, 'support_set')
    print_array(test_query, 'query')
    print_array(out_prototypes_no_cnn, 'prototypes')
    print_array(out_distances, 'distances')
    print_array(out_prediction, 'prediction')

    return
  


dim = 9
N = 5
k = 3

inp_shape = (12,40,1)
cnn_backbone = custom_models.improvedAtzoriNet(inp_shape)#simplest_conv_net_1_layer(input_shape = inp_shape, feature_vector_size=12)


model_timeDist = layers.TimeDistributed(cnn_backbone)

#input shape tuple
inp_shape_5d = (None,) + inp_shape

#Layers
support_set_inp_shape_layer = layers.Input(inp_shape_5d)
query_set_inp_shape_layer = layers.Input(inp_shape)

support_set_embeddings_layer = model_timeDist(support_set_inp_shape_layer)

query_set_embedding_layer = cnn_backbone(query_set_inp_shape_layer)

prototypes_layer = Lambda(function=fsl_functions.produce_prototype)(support_set_embeddings_layer)


# query_distances_layer = Lambda(function=fsl_functions.euc_dist)([prototypes_layer, query_set_embedding_layer])

query_prediction_layer = Lambda(function=fsl_functions.softmax_classification)([prototypes_layer, query_set_embedding_layer])



#models


model_support_embeddings = keras.Model(inputs=support_set_inp_shape_layer, outputs=support_set_embeddings_layer)
model_query_embeddings = keras.Model(inputs=query_set_inp_shape_layer, outputs=query_set_embedding_layer)
model_prototypes = keras.Model(inputs = model_support_embeddings.output, outputs=prototypes_layer)


model = keras.Model(inputs=[support_set_inp_shape_layer,query_set_inp_shape_layer], outputs=query_prediction_layer)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['categorical_accuracy'])#, run_eagerly=True)

ex = '1'
N = 5
k = 3

train_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', batches=10000)

[x,y], label = train_loader[0]


check_support_embeddings_array = model_support_embeddings.predict(x)
check_query_embedding_array = model_query_embeddings.predict(y)
check_prototypes_array = model_prototypes.predict(check_support_embeddings_array)
check_prototypes_manual_bool_check = np.all(check_prototypes_array == tf.reduce_sum(check_support_embeddings_array, axis=1) / k)
check_distances = fsl_functions.euc_dist([check_prototypes_array, check_query_embedding_array])
check_predictions = fsl_functions.softmax_classification([check_prototypes_array, check_query_embedding_array])

get_prototypes_distances_for_small_set()

print("END")

#model.fit(train_loader, epochs=25,   shuffle=False)



