import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

seed = 21

from keras.datasets import cifar10


class config:
    """
    Configure model with __init__.
    Call train_model to train it and store the history and scores.
    Plot the results with plot_my_model
    
    Model is built using the other funcs as part of train_model.
    
        def __init__(self, 
                filter_x_size = 3, 
                filter_y_size = 3,  
                filter_channels = 32,
                dropout_fraction = 0.2,
                num_pooling_layers = 2,
                num_CNN_blocks = 3,
                neurons_per_dense_layer = 32,
                num_dense_layers = 1,
                dense_activation_type = 'softmax',
                num_epochs = 15,
                batch_size = 64):
    """
    
    def __init__(self, 
                filter_x_size: int, 
                filter_y_size: int,  
                filter_channels: int,
                dropout_fraction: float,
                num_pooling_layers: int,
                num_CNN_blocks: int,
                neurons_per_dense_layer: int,
                num_dense_layers: int,
                dense_activation_type: str,
                num_epochs: int,
                batch_size: int):
        self._filter_x_size = filter_x_size
        self._filter_y_size = filter_y_size
        self._filter_channels = filter_channels
        self._dropout_fraction = dropout_fraction
        self._num_pooling_layers = num_pooling_layers
        self._num_CNN_blocks = num_CNN_blocks
        self._neurons_per_dense_layer = neurons_per_dense_layer
        self._num_dense_layers = num_dense_layers
        self._dense_activation_type = dense_activation_type
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        

    def add_CNN_block(self, inp_shape, multiply: int ):
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
        
        self._model.add(keras.layers.Conv2D(multiply*self._filter_channels, (self._filter_x_size,self._filter_y_size), input_shape=inp_shape, activation='relu', padding='same'))

        # 2: specify max pooling
        self._model.add(keras.layers.MaxPooling2D(self._num_pooling_layers))

        # 3: specify dropout fraction in layer2 (default, 0.2 or 20%)
        self._model.add(keras.layers.Dropout(self._dropout_fraction))

        # 4: use batch normalisation.
        self._model.add(keras.layers.BatchNormalization())

        ########################################################################################
        
        
    def add_dense_layer(self, i):

        self._model.add(keras.layers.Dense(i*self._neurons_per_dense_layer, activation='relu'))
        self._model.add(keras.layers.Dropout(self._dropout_fraction))
        self._model.add(keras.layers.BatchNormalization())
    
    def train_model(self):
        # created variables
        # cfg = config(filter_x_size = 3,
        # filter_y_size = 3,
        # filter_channels = 32,
        # dropout_fraction = 0.2,
        # num_pooling_layers = 2,
        # num_CNN_blocks = 5,
        # neurons_per_dense_layer = 32,
        # num_dense_layers = 1,
        # dense_activation_type = 'softmax',
        # numEpochs = 25,
        # batchSize=64)


        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        #normalise inputs
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # assume 255 cause 8bit rgb image dataset
        x_train /=255.0
        x_test/=255.0

        # one hot encoding
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        class_num = y_test.shape[1]                                 # number of classes in the training dataset. will be variable


        # create the model. 
        # model = keras.Sequential()
        # model.add(keras.layers.layer1)
        # model.add(keras.layers.layer2)
        # model.add(keras.layers.layer3)
        # or do this:
        # model = keras.Sequential([keras.layers.layer1, keras.layers.layer2, keras.layers.layer3])

        # model = create_CNN_block(filter_channels, filter_x_size, filter_y_size, x_train.shape[1:], num_pooling_layers, dropout_fraction)

        # Create ._model
        self._model = keras.Sequential()
        
        # create a model with a number of CNN layers incrementally more abstract/larger/complex
        for x in range(self._num_CNN_blocks):
            i=x+1
            self.add_CNN_block(x_train.shape[1:],i)
            
        # flatten the data and add another dropout layer
        self._model.add(keras.layers.Flatten())
        self._model.add(keras.layers.Dropout(self._dropout_fraction))

        # add some dense layers to the model
        for y in range(self._num_dense_layers):
            j=y+1
            self.add_dense_layer(j)

        # tell the model how many categories to classify the data into
        # dense_activation_type = softmax = highest probability voting.
        self._model.add(keras.layers.Dense(class_num, activation=self._dense_activation_type))

        # optimize the model with some optimizer algorithm such as adaptive moment estimation (adam), rmsprop, nadam etc)
        # track the accuracy in training data against test data. divergence means its overfitted to the test data. 
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        # print a summary of the model.
        #print(model.summary())

        numpy.random.seed(seed)



        self._history = self._model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self._num_epochs, batch_size=self._batch_size)

        self._scores = self._model.evaluate(x_test, y_test, verbose=0)
        #f"Accuracy: {self._scores[1]*100}%"
        
        #return self._history, self._scores


    def plot_my_model(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        pd.DataFrame(self._history.history).plot()
        plt.show()