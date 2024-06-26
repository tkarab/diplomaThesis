import os
import custom_models
import fsl_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from task_generator import TaskGenerator


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
model = keras.Model(inputs=[support_set_inp_shape_layer,query_set_inp_shape_layer], outputs=query_prediction_layer)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['categorical_accuracy'])#, run_eagerly=True)

ex = '1'
N = 5
k = 3
train_loader = TaskGenerator(experiment=ex, way=N, shot=k, mode='train', batches=10000)

[x,y], label = train_loader[10]

model_check_query_embedding = keras.Model(inputs=query_set_inp_shape_layer, outputs=query_set_embedding_layer)
check_query_embedding = model_check_query_embedding.predict(y)

model.fit(train_loader, epochs=25,   shuffle=False)
print('END')
# model1.save('saved_models/model1.h5')
