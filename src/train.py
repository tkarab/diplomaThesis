import os
import custom_models
import fsl_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from task_generator import TaskGenerator
import numpy as np


dim = 9
N = 5
k = 3

inp_shape = (12,40,1)
cnn_backbone = custom_models.improvedAtzoriNet(inp_shape)#simplest_conv_net_1_layer(input_shape = inp_shape, feature_vector_size=12)

# inp = layers.Input(shape=inp_shape)
# model_4d = keras.Model(inputs=inp, outputs=cnn_backbone(inp))
model_timeDist = layers.TimeDistributed(cnn_backbone)

#input shape tuple
inp_shape_5d = (None,) + inp_shape

#Layers
support_set_inp_shape_layer = layers.Input(inp_shape_5d)
query_set_inp_shape_layer = layers.Input(inp_shape)
query_set_keep_first_tensor_layer = Lambda(function=fsl_functions.keep_first_tensor)(query_set_inp_shape_layer)

support_set_embeddings_layer = model_timeDist(support_set_inp_shape_layer)
# query_set_embedding_layer = cnn_backbone(query_set_keep_first_tensor_layer)
query_set_embedding_layer = cnn_backbone(query_set_inp_shape_layer)

prototypes_layer = Lambda(function=fsl_functions.produce_prototype)(support_set_embeddings_layer)
# query_embedding_reshaped_layer = Lambda(function=fsl_functions.keep_first_tensor)(query_set_embedding_layer)
#
# query_distances_layer = Lambda(function=fsl_functions.euc_dist)([prototypes_layer, query_set_embedding_layer])

query_prediction_layer = Lambda(function=fsl_functions.softmax_classification)([prototypes_layer, query_set_embedding_layer])



#models
model_keep_first_tensor = keras.Model(inputs=query_set_inp_shape_layer, outputs=query_set_keep_first_tensor_layer)

model_support_query_embeddings = keras.Model(inputs=[support_set_inp_shape_layer, query_set_inp_shape_layer], outputs=[support_set_embeddings_layer, query_set_embedding_layer])
model_prototypes = keras.Model(inputs = model_support_query_embeddings.output[0], outputs=prototypes_layer)

test_N = 3
test_k = 2
test_dim = 5
test_inp = np.array([[[1,3,2,0],[1,5,2,9]],[[1,6,8,3],[1,43,2,5]],[[5,1,7,8],[2,6,5,1]]])
test_query = np.array([1,4,2,5], dtype=np.float32)
test_inp_layer = layers.Input(shape=test_inp.shape[1:])
test_query_inp = layers.Input(shape=(4))
test_prototype_no_cnn = Lambda(function=fsl_functions.produce_prototype)(test_inp_layer)
test_distances_no_cnn = Lambda(function=fsl_functions.euc_dist)([test_prototype_no_cnn, test_query_inp])
model_prototypes_no_cnn = keras.Model(inputs=test_inp_layer , outputs=test_prototype_no_cnn)
model_distances_no_cnn = keras.Model(inputs=[model_prototypes_no_cnn.output, test_query_inp], outputs=test_distances_no_cnn)

out_prototypes_no_cnn = model_prototypes_no_cnn.predict(test_inp)
# out_distances_no_cnn = model_distances_no_cnn.predict([out_prototypes_no_cnn, test_query])

model = keras.Model(inputs=[support_set_inp_shape_layer,query_set_inp_shape_layer], outputs=query_prediction_layer)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['categorical_accuracy'])#, run_eagerly=True)

ex = '1'
N = 5
k = 3


train_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', batches=10000)

[x,y], label = train_loader[0]

check_keep_first_tensor = model_keep_first_tensor.predict(y)
check_support_query_embeddings = model_support_query_embeddings.predict([x, y])
check_prototypes = model_prototypes.predict(check_support_query_embeddings[0])
check_prototypes_manual = np.all(check_prototypes == tf.reduce_sum(check_support_query_embeddings[0],axis=1)/k)

print("END")

#model.fit(train_loader, epochs=25,   shuffle=False)

# model1.save('saved_models/model1.h5')
