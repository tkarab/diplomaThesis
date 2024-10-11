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

"""
DESCRIPTION
    Returns a set of fully connected layers, each with the number of neurons defined by a list parameter.
    Last layer is a single neuron layer providing a scalar output
    Each layer has a relu activation function except for the last which has a sigmoid (for providing a normalized output between 0.0 and 1.0)
    
PARAMETERS
    - neurons_per_layer = [128,64] would mean 2 layers of 128 and 64 neurons etc before the single neuron output 
"""
def get_dense_layers(neurons_per_layer=[]):
    dense_layers = keras.Sequential()
    for i,neurons_number in enumerate(neurons_per_layer):
        dense_layers.add(layers.BatchNormalization())
        dense_layers.add(layers.Dense(units=neurons_number,activation='relu',name=f"dense_layer_{i+1}"))
    dense_layers.add(layers.Dense(units=1,activation='sigmoid', name=f"prediction_dense_layer"))

    return dense_layers


"""
PARAMETERS
    - f: the distance (or similarity) function
"""
def assemble_siamNet(cnn_backbone, f, input_shape:tuple):
    # input_shape_4d = (1,) + input_shape
    x1 = keras.Input(shape=input_shape, name="input_X1")
    x2 = keras.Input(shape=input_shape, name="input_X2")

    embedding1 = cnn_backbone(x1)
    embedding2 = cnn_backbone(x2)

    # where f is a distance/similarity function
    embedding_dist = f([embedding1,embedding2])

    # Dense Layer produces a weighted sum of the difference/product (which is scalar)
    # It is then passed through the sigmoid to produce a normalized similarity score between 0 and 1
    # The weights should be adjusted during training to produce large positive values for images of the same class
    # and large negative values for inputs of different class
    similarity_score = layers.Dense(units=1,activation="sigmoid")(embedding_dist)

    model = keras.Model(inputs=[x1,x2], outputs=similarity_score)

    return model

class SiameseNetwork(keras.Model):
    def __init__(self,cnn_backbone:keras.Model,f,inp_shape:tuple, dense_layers):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = cnn_backbone
        self.f = f
        self.inp_shape = inp_shape

        self.dense_layers = dense_layers

        return

    def call(self, input):
        x1,x2 = input

        embedding1 = self.feature_extractor(x1)
        embedding2 = self.feature_extractor(x2)

        embedding_dist = self.f([embedding1,embedding2])

        similarity_score = self.dense_layers(embedding_dist)

        return similarity_score