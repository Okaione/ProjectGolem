import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

seed = 21

from keras.datasets import cifar10


class config:
    def __init__(self, 
                filter_x_size = 3, 
                filter_y_size = 3,  
                filter_channels = 32,
                dropout_fraction = 0.2,
                num_pooling_layers = 2,
                num_CNN_blocks = 3,
                neurons_per_dense_layer = 32,
                num_dense_layers = 1,
                dense_activation_type = 'softmax'):
        self._filter_x_size = filter_x_size
        self._filter_y_size = filter_y_size
        self._filter_channels = filter_channels
        self._dropout_fraction = dropout_fraction
        self._num_pooling_layers = num_pooling_layers
        self._num_CNN_blocks = num_CNN_blocks
        self._neurons_per_dense_layer = neurons_per_dense_layer
        self._num_dense_layers = num_dense_layers
        self._dense_activation_type = dense_activation_type
        

def add_CNN_block(model, cfg: config, inp_shape, multiply: int ):
    """Create a basic block of a relu CNN. Uses batch normalization.

    Args: Hyperparameters to use in the CNN:
        num_filter_channels (int): [number of filters/channels to use]
        filter_x_size (int): [x-dimension size of the filter to use]
        filter_y_size (int): [y dimension size of the filter to use]
        inp_shape ([type]): [input_shape]

    Returns:
        [instance of keras.Sequential()]: [Returns a basic block of a CNN.]
    """
    
    ################# The 4 main parts of a basic "block" used to build CNNs ###############
    # 1: make layer 1 a sequential model. CNN, relu
    
    model.add(keras.layers.Conv2D(multiply*cfg._num_filter_channels, (cfg._filter_x_size,cfg._filter_y_size), input_shape=inp_shape, activation='relu', padding='same'))

    # 2: specify max pooling
    model.add(keras.layers.MaxPooling2D(cfg._num_pooling_layers))

    # 3: specify dropout fraction in layer2 (default, 0.2 or 20%)
    model.add(keras.layers.Dropout(cfg._dropout_fraction))

    # 4: use batch normalisation.
    model.add(keras.layers.BatchNormalization())
    
    return model
    ########################################################################################
    
    
def add_dense_layer(model,cfg, i):

    model.add(keras.layers.Dense(i*cfg._neurons_per_dense_layer, activation='relu'))
    model.add(keras.layers.Dropout(cfg._dropout_fraction))
    model.add(keras.layers.BatchNormalization())
    return model