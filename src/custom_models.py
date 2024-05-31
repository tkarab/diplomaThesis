import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import argparse

def simple_conv_net(feature_vector_size=5, input_shape = (28,28,1)):
    convnet = keras.Sequential(name='mymodel')
    convnet.add(layers.Input(shape=input_shape))
    for i in range(3):
        convnet.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding='same', name='conv2d_'+str(i+1)))
        convnet.add(layers.BatchNormalization(name='batchNorm_'+str(i+1)))
        convnet.add(layers.Activation(activation='relu', name='activ_'+str(i+1)))
        convnet.add(layers.MaxPooling2D(name='maxPool_'+str(i+1)))

    convnet.add(layers.Conv2D(filters=feature_vector_size,kernel_size=(3,3),padding='same', name='conv2d_'+str(i+2)))
    convnet.add(layers.BatchNormalization( name='batchNorm_'+str(i+2)))
    convnet.add(layers.Activation(activation='relu', name='activ_'+str(i+2)))
    convnet.add(layers.MaxPooling2D(name='maxPool_'+str(i+2)))
    convnet.add(layers.Flatten(name='flatten'+str(i+2)))
    return convnet

def simplest_conv_net_1_layer(feature_vector_size=5, input_shape = (5,5,1)):
    convnet = keras.Sequential(name='mymodel')
    convnet.add(layers.Input(shape=input_shape))
    convnet.add(layers.Conv2D(filters=feature_vector_size, kernel_size=input_shape[0]-2, padding='valid', name='conv2d'))
    convnet.add(layers.BatchNormalization(name='batchNorm'))
    convnet.add(layers.Activation(activation='relu', name='activ'))
    convnet.add(layers.MaxPooling2D(name='maxPool'))
    convnet.add(layers.Flatten(name='flatten'))
    return convnet


def atzoriNetDB1(N = 53):
    model = keras.Sequential(
        layers=[
            #Input
            layers.Input(shape=(15,10,1)),

            #Layer 1
            layers.Conv2D(filters=32, kernel_size=(1,10), activation='relu', padding='same'),

            #Layer 2
            layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(3,3)),

            #Layer 3
            layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(3, 3)),

            #Layer 4
            layers.Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same'),

            #Layer 5
            layers.Conv2D(filters=N, kernel_size=(1,1), activation='softmax', padding='same')
        ]
    )

    return model

def atzoriNetDB2(N = 49):
    model = keras.Sequential(
        layers=[
            #Input
            layers.Input(shape=(15,12,1)),

            #Layer 1
            layers.Conv2D(filters=32, kernel_size=(1,12), activation='relu', padding='same', name='Conv1'),

            #Layer 2
            layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', name='Conv2'),
            layers.AveragePooling2D(pool_size=(3,3), name='pool2'),

            #Layer 3
            layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same', name='Conv3'),
            layers.AveragePooling2D(pool_size=(3, 3), name='pool3'),

            #Layer 4
            layers.Conv2D(filters=64, kernel_size=(9, 1), activation='relu', padding='same', name='Conv4'),

            #Layer 5
            layers.Conv2D(filters=N, kernel_size=(1,1), activation='softmax', padding='same', name='Conv5')
        ]
    )

    return model


def improvedAtzoriNet(input_shape=(15,10,1),name='ImprovedAtzoriNet'):
  model = keras.Sequential(
      name=name,
      layers=[
            keras.Input(shape=input_shape),
            layers.Conv2D(filters=32, kernel_size=(1,input_shape[1]), padding='same', activation='relu', name='conv1'),
            layers.Dropout(0.15),

            layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', name='conv2'),
            layers.Dropout(0.15),
            layers.MaxPooling2D(pool_size=(3,3)),

            layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu', name='conv3'),
            layers.Dropout(0.15),
            layers.MaxPooling2D(pool_size=(3,3)),

            layers.Conv2D(filters=64, kernel_size=(5,1), padding='same', activation='relu', name='conv4'),
            layers.Flatten(name='flatten'),
            layers.Dropout(0.15),
        ])
  return model


def test_more_dims_cnn():
    convnet = keras.Sequential(name='mymodel')
    convnet.add(layers.Input(shape=(28,28,1)))
    convnet.add(layers.Conv2D(filters=63, kernel_size=3, padding='valid', name='conv2d'))
    convnet.add(layers.BatchNormalization(name='batchNorm'))
    convnet.add(layers.Activation(activation='relu', name='activ'))
    convnet.add(layers.MaxPooling2D(name='maxPool'))
    convnet.add(layers.Flatten(name='flatten'))
    return convnet

"""
CNN Backbone used in the Prototypical Network Implementation by Godoy et al. in the Paper 
    'Electromyography Based Gesture Decoding
    Employing Few-Shot Learning, Transfer Learning,
    and Training From Scratch' 

Input:  a 8x240 sEMG recording
Output: a 8*15*64 = 7680 long vector (embedding)
 
Made up of 4 Convolutional Blocks, each consisting of a conv2D layer, a BatchNorm layer and a max-pooling layer (only in the time dimension) 
Lastly a flattening layer is added
"""
def test_godoy_net(input_shape = (8,240,1)):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(4):
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(1,2)))

    model.add(layers.Flatten())

    return model

def convert_to_tuple(input_string):
    try:
        # Convert the string to a tuple
        return tuple(map(int, input_string.strip("()").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: '{input_string}'")

def main():
    # parser = argparse.ArgumentParser(description="Example script with tuple argument")
    # parser.add_argument("--tuple_arg", type=convert_to_tuple, required=True, help="Tuple argument (e.g., '(1,2,3)')")
    #
    # args = parser.parse_args()

    print(improvedAtzoriNet().summary())


if __name__ == '__main__':

    main()