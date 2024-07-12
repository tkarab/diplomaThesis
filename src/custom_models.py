import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers, regularizers

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

"""
Inspired by Atzori's implementation for Ninapro DB1 data.
Input: (15,10,1) sEMG recording
Output: (N,1) vector with prediction for each of N classes

No Dropout Layer Used
"""
def AtzoriNetDB1(N = 53, input_shape = (15,10,1)):
    model = keras.Sequential(
        name = 'AtzoriNetDB1',
        layers = [
            #Input
            layers.Input(shape=input_shape),

            #Layer 1: Padding (0,4) -> Conv2D [32 x (1,10)] -> ReLU
                # Input:  (15,10,1)
                # Output: (15,9,32)
            layers.ZeroPadding2D(padding=(0,4)),
            layers.Conv2D(filters=32, kernel_size=(1,10), padding='valid', activation='relu'),

            #Layer 2: Padding (1,1) -> Conv2D [32 x (3,3)] -> ReLU -> AvgPooling(3,3)
                # Input:  (15,9,32)
                # Output: (5,3,32)

            layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
            layers.AveragePooling2D(pool_size=(3,3)),

            #Layer 3: Padding (2,2) -> Conv2D [64 x (5,5)] -> ReLU -> AvgPooling(3,3)
                # Input:  (5,3,32)
                # Output: (1,1,64)
            # layers.ZeroPadding2D(padding=(2,2)),
            layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'),
            layers.AveragePooling2D(pool_size=(3,3)),

            #Layer 4: Padding (2,0) -> Conv2D [64 x (5,1)] -> ReLU
            # The Output is a vector of length 64 corresponding to the input image
                # Input:  (1,1,64)
                # Output: (1,1,64)
            # layers.ZeroPadding2D(padding=(2,0)),
            layers.Conv2D(filters=64, kernel_size=(5,1), padding='same', activation='relu'),

             #Layer 5: Conv2D [Nx(1,1)] -> Softmax -> Flatten
            layers.Conv2D(filters=N, kernel_size=(1,1), padding='valid', activation='softmax'),
            layers.Flatten()
        ]
    )

    return model


"""
Inspired by Atzori's implementation for Ninapro DB2 data.
Input: (30,12,1) sEMG recording
Output: (N,1) vector with prediction for each of N classes

No Dropout Layer Used
"""
def AtzoriNetDB2(N = 49, input_shape = (15,12,1)):

    model = keras.Sequential(
        name = 'AtzoriNetDB2',
        layers = [
            #Input
            layers.Input(shape=input_shape),

            #Layer 1: Padding (0,5) -> Conv2D [32 x (1,12)] -> ReLU
                # Input:  (15,12,1)
                # Output: (15,11,32)
            layers.ZeroPadding2D(padding=(0,5)),
            layers.Conv2D(filters=32, kernel_size=(1,12), padding='valid', activation='relu'),

            #Layer 2: Padding (1,1) -> Conv2D [32 x (3,3)] -> ReLU -> AvgPooling(3,3)
                # Input:  (15,11,32)
                # Output: (5,3,32)
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu'),
            layers.AveragePooling2D(pool_size=(3,3)),

            #Layer 3: Padding (2,2) -> Conv2D [64 x (5,5)] -> ReLU -> AvgPooling(3,3)
                # Input:  (5,3,32)
                # Output: (1,1,64)
            layers.ZeroPadding2D(padding=(2,2)),
            layers.Conv2D(filters=64, kernel_size=(5,5), padding='valid', activation='relu'),
            layers.AveragePooling2D(pool_size=(3,3)),

            #Layer 4: Padding (4,0) -> Conv2D [64 x (9,1)] -> ReLU
            # The Output is a vector of length 64 corresponding to the input image
                # Input:  (1,1,64)
                # Output: (1,1,64)
            layers.ZeroPadding2D(padding=(4,0)),
            layers.Conv2D(filters=64, kernel_size=(9,1), padding='valid', activation='relu'),

             #Layer 5: Conv2D [Nx(1,1)] -> Softmax -> Flatten
            layers.Conv2D(filters=N, kernel_size=(1,1), padding='valid', activation='softmax'),
            layers.Flatten()
        ]
    )

    return model


def AtzoriNetDB2_embedding_only(input_shape = (15,12,1), add_dropout = True, dropout_pct = 0.15, krnl_init_name = "glorot_normal", add_regularizer:bool = False, l2 = 0.0002):
    # Kernel Initializer
    if krnl_init_name == "glorot_normal":
        kernel_init = initializers.glorot_normal(seed=0)
    else:
        kernel_init = initializers.glorot_uniform(seed=0)

    if add_regularizer:
        kernel_reg = regularizers.l2(l2)
    else:
        kernel_reg = None

    # Input
    X_inp = layers.Input(shape=input_shape)

    # Layer 1: Padding (0,5) -> Conv2D [32 x (1,12)] -> ReLU
    # Input:  (15,12,1)
    # Output: (15,11,32)
    X = layers.ZeroPadding2D(padding=(0, 5))(X_inp)
    X = layers.Conv2D(filters=32, kernel_size=(1, 12), padding='valid', activation='relu', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(X)
    if add_dropout == True:
        X = layers.Dropout(dropout_pct)(X)

    # Layer 2: Padding (1,1) -> Conv2D [32 x (3,3)] -> ReLU -> AvgPooling(3,3)
    # Input:  (15,11,32)
    # Output: (5,3,32)
    X = layers.ZeroPadding2D(padding=(1, 1))(X)
    X = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(X)
    if add_dropout == True:
        X = layers.Dropout(dropout_pct)(X)
    X = layers.AveragePooling2D(pool_size=(3, 3))(X)

    # Layer 3: Padding (2,2) -> Conv2D [64 x (5,5)] -> ReLU -> AvgPooling(3,3)
    # Input:  (5,3,32)
    # Output: (1,1,64)
    X = layers.ZeroPadding2D(padding=(2, 2))(X)
    X = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid', activation='relu', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(X)
    if add_dropout == True:
        X = layers.Dropout(dropout_pct)(X)
    X = layers.AveragePooling2D(pool_size=(3, 3))(X)

    # Layer 4: Padding (4,0) -> Conv2D [64 x (9,1)] -> ReLU
    # The Output is a vector of length 64 corresponding to the input image
    # Input:  (1,1,64)
    # Output: (1,1,64)
    X = layers.ZeroPadding2D(padding=(4, 0))(X)
    X = layers.Conv2D(filters=64, kernel_size=(9, 1), padding='valid', activation='relu', kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(X)
    if add_dropout == True:
        X = layers.Dropout(dropout_pct)(X)

    Y = layers.Flatten()(X)

    model = keras.Model(name = 'AtzoriNetDB2',
                        inputs = X_inp,
                        outputs = Y)

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

# model = AtzoriNetDB2_embedding_only()
# print(model.summary())