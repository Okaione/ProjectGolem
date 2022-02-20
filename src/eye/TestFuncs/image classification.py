import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

seed = 21

from keras.datasets import cifar10

# bad practice but it's my file, i do what i want
from image_classification_functions import *

# filter parameters and num_CNN_blocks affect runtime majorly

# created variables
cfg = config(filter_x_size = 3,
filter_y_size = 3,
filter_channels = 32,
dropout_fraction = 0.2,
num_pooling_layers = 2,
num_CNN_blocks = 5,
neurons_per_dense_layer = 32,
num_dense_layers = 1,
dense_activation_type = 'softmax')

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

# Fitting variables
numEpochs = 25
batchSize=64

# create the model. 
# model = keras.Sequential()
# model.add(keras.layers.layer1)
# model.add(keras.layers.layer2)
# model.add(keras.layers.layer3)
# or do this:
# model = keras.Sequential([keras.layers.layer1, keras.layers.layer2, keras.layers.layer3])

# model = create_CNN_block(filter_channels, filter_x_size, filter_y_size, x_train.shape[1:], num_pooling_layers, dropout_fraction)

model = keras.Sequential()

# create a model with a number of CNN layers incrementally more abstract/larger/complex
for x in range(num_CNN_blocks):
    i=x+1
    model = add_CNN_block(model, cfg, x_train.shape[1:], i)
    
# flatten the data and add another dropout layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(dropout_fraction))

# add some dense layers to the model
for y in range(num_dense_layers):
    j=y+1
    model = add_dense_layer(model,cfg, j)

# tell the model how many categories to classify the data into
# dense_activation_type = softmax = highest probability voting.
model.add(keras.layers.Dense(class_num, activation=dense_activation_type))

# optimize the model with some optimizer algorithm such as adaptive moment estimation (adam), rmsprop, nadam etc)
# track the accuracy in training data against test data. divergence means its overfitted to the test data. 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# print a summary of the model.
print(model.summary())

numpy.random.seed(seed)



history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=numEpochs, batch_size=batchSize)

scores = model.evaluate(x_test, y_test, verbose=0)
f"Accuracy: {scores[1]*100}%"

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()
plt.show()