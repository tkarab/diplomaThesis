import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fsl_functions import *

"""
This file includes functions for assembling different types of models
"""

"""
DESCRIPTION
    Creates a model by placing a time distributed layer in front of the backbone so the model can process and produce output for 2-d inputs (i.e 5x3 images of size 30x15)
PARAMETERS:
    - nn_backbone: a keras Sequential model corresponding to the neural network which serves as the backbone of the few-shot learning model. Depending on the approach this could be a CNN, a TCN, a transformer etc
    - input_shape: a tuple containing the shape of the input data of the model. i.e. (15,12,1) 
"""
def assemble_protonet_timeDist(nn_backbone, input_shape:tuple):
    layer_time_dist = layers.TimeDistributed(nn_backbone, name="Time_Distributed")

    # tuple of 2d input set shape containing one extra dimension
    input_shape_5d = (None,) + input_shape
    layer_support_set_input = layers.Input(input_shape_5d, name="Support_Set_Input")
    layer_query_set_input = layers.Input(input_shape, name="Query_Set_Input")

    layer_support_set_embeddings = layer_time_dist(layer_support_set_input)
    layer_prototypes = layers.Lambda(produce_prototype,name="Prototypes")(layer_support_set_embeddings)

    layer_query_set_embedding = nn_backbone(layer_query_set_input)

    layer_output_prediction = layers.Lambda(softmax_classification, name="Prediction")([layer_prototypes, layer_query_set_embedding])
    
    model = keras.Model(inputs=[layer_support_set_input, layer_query_set_input], outputs=layer_output_prediction, name="Atzori_DB2_Protonet")

    return model

"""
DESCRIPTION
    Instead of adding a time-distributed layer in front of the cnn backbone to process 2d sets of data
    it reshapes the support set into a 1d set. i.e. a (5,3) set of 15x12 images will be turned into a 5 x 3 = 15 
    images of size 15x12 and will all be processed simultaneously. 
    This will produces the 15 embeddings which will then be reshaped into the original 5x3 shape again
"""
def assemble_protonet_reshape(nn_backbone, input_shape:tuple, way:int, shot:int):
    input_shape_5d = (None,) + input_shape
    layer_support_set_input = layers.Input(input_shape_5d, name="Support_Set_Input")
    layer_query_set_input = layers.Input(input_shape, name="Query_Set_Input")
    
    # Concatenate the support set and query image into a 1d set of the standard input shape.
    # i.e. a (3,2) support set and 1 query image of size (15,12,1) will form a set of 2x3+1 = 7 images (15,12,1)
    layer_support_query_concat = tf.concat([tf.reshape(layer_support_set_input,[way*shot]+list(input_shape)), layer_query_set_input],axis=0)

    # We Produce Embeddings for all input data (support and query) by passing them all together through the network in feed forward fashion
    # Based on the previous example, given that a (15,12,1) image produces a (64,1) embedding, the whole
    # reshaped (7,15,12,1) set should produce a (7,64,1) set of embeddings
    layer_support_query_embeddings = nn_backbone(layer_support_query_concat)

    # Separate quary and support embeddings
    # Separate and reshape support set embeddings (6,64,1) -> (3,2,64,1)
    layer_support_set_embeddings = tf.reshape(layer_support_query_embeddings[:way*shot],[way,shot,layer_support_query_embeddings.shape[1]])
    layer_query_set_embedding = layer_support_query_embeddings[way*shot:]

    # Produce the prototypes of the support set
    # This should reduce the support set embeddings down to: (3,2,64,1) -> (3,64,1) by averaging the 2 embeddings of each of the 3 classes
    layer_prototypes = layers.Lambda(produce_prototype)(layer_support_set_embeddings)

    # Make final prediction
    layer_prediction = layers.Lambda(softmax_classification)([layer_prototypes, layer_query_set_embedding])

    # Assemble the final Model
    model = keras.Model(inputs = [layer_support_set_input, layer_query_set_input], outputs=layer_prediction)

    return model

def assemble_protonet_reshape_with_batch(nn_backbone, input_shape:tuple, way:int, shot:int):
    input_shape_4d = (1,) + input_shape
    input_shape_5d = (None,None,) + input_shape
    layer_support_set_input = layers.Input(input_shape_5d, name="Support_Set_Input")
    layer_query_set_input = layers.Input(input_shape_4d, name="Query_Set_Input")

    nn_backbone_timeDist = layers.TimeDistributed(nn_backbone)

    # Concatenate the support set and query image into a 1d set of the standard input shape.
    # i.e. a (3,2) support set and 1 query image of size (15,12,1) will form a set of 2x3+1 = 7 images (15,12,1)
    layer_support_query_concat = tf.concat([tf.reshape(layer_support_set_input,[-1,way*shot]+list(input_shape)), layer_query_set_input],axis=1)

    # We Produce Embeddings for all input data (support and query) by passing them all together through the network in feed forward fashion
    # Based on the previous example, given that a (15,12,1) image produces a (64,1) embedding, the whole
    # reshaped (7,15,12,1) set should produce a (7,64,1) set of embeddings
    layer_support_query_embeddings = nn_backbone_timeDist(layer_support_query_concat)

    # Separate query and support embeddings
    # Separate and reshape support set embeddings (6,64,1) -> (3,2,64,1)
    layer_support_set_embeddings = tf.reshape(layer_support_query_embeddings[:,:way * shot], [-1,way, shot, layer_support_query_embeddings.shape[-1]])
    layer_query_set_embedding = layer_support_query_embeddings[:,way*shot:]

    # Produce the prototypes of the support set
    # This should reduce the support set embeddings down to: (3,2,64,1) -> (3,64,1) by averaging the 2 embeddings of each of the 3 classes
    layer_prototypes = layers.Lambda(produce_prototype)(layer_support_set_embeddings)

    # Make final prediction
    layer_prediction = layers.Lambda(softmax_classification)([layer_prototypes, layer_query_set_embedding])

    # Assemble the final Model
    model = keras.Model(inputs = [layer_support_set_input, layer_query_set_input], outputs=layer_prediction, name="ProtoNet")

    return model