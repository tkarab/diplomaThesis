import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def improvedAtzoriNet(input_shape=(12,40,1),name='ImprovedAtzoriNet'):
  model = keras.Sequential(
      name=name,
      layers=[
            keras.Input(shape=input_shape),
            layers.Conv2D(filters=32, kernel_size=(1,input_shape[1]), padding='same', activation='relu', name='conv1'),
            layers.Dropout(0.15),
            layers.MaxPooling2D(pool_size=(1,3)),

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

def test_godoy_net(input_shape = (12,40,1)):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(4):
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(1,2)))

    model.add(layers.Flatten())

    return model

model = test_godoy_net()
print(model.summary())