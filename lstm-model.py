'''

LSTM Classification Model

Purpose: Define and test different kinds of Long-Short Term Memory Deep Learning Networks

Overall approach:
    1. Load training and test datasets
    2. Define LSTM model parameters
    3. Set up an LSTM model in Keras (Default Tensorflow backend)
    4. Calculate predicted classes for the train and test data sets
    5. Evaluate model using metrics such as precision, recall and F1-score and print a confusion matrix

Author: Rajesh S (@rexplorations)
Email: rexplorations@gmail.com

'''



import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from data_gen import gen_data_labels

'''
Reading in the training and test data.

1. All data read into the script are `*.npy` numpy array files
2. These are not text files but binary data files generated in the dataset_prep.py script
3. The data consists of:
    a. Multiple sequences of time series data
    b. Each sequence has 100 data points
    c. A target set which is a series of labels (indicating the kind of time series)

'''

trainX = np.load("data/trainX.npy")
trainY = np.load("data/trainY.npy")
testX = np.load("data/testX.npy")
testY = np.load("data/testY.npy")

print(trainX.shape, trainY.shape, testX.shape, testY.shape)

'''
Defining a Sequential() Keras model
'''
seq_size = trainX.shape[1]

'''
 Below we set the random seed to ensure repeatability. 
 Methods like fixed inital_state values for the LSTM have not been shown here. But they could also be used. 
'''
np.random.seed(123) 

'''
Defining key model parameters

'''

batchsize = 64
epochs = 16
n_units = 16
n_hidden = 16
lr = 1e-3
decay = 0.1 * lr
dropout_pc = 0.01
l1, l2 = 0.01, 0.01
n_classes = trainY.shape[1]

'''
1. The model is defined using Keras' Sequential model constructor. This takes a python list of layers as arguments. 
2. Note that the first LSTM() layer both receives an input_shape argument, and returns sequences
3. Since this is a multi-class sequence classification problem, the last layer is a Dense() layer with the `softmax` activation function
4. Activation for the LSTM() layers and the Dense() layers are different and could be treated as hyperparameters
'''


model = Sequential([
    LSTM(units = n_units, activation = "relu", kernel_regularizer = l1_l2(l1, l2), input_shape = (seq_size, 1), return_sequences = True),
    Dense(n_hidden, activation = "tanh"),
    LSTM(units = n_units, activation = "relu", kernel_regularizer = l1_l2(l1, l2), dropout = dropout_pc, return_sequences = False),
    Dense(n_hidden, activation = "tanh"),
    BatchNormalization(),
    Dropout(dropout_pc),
    Dense(n_hidden, activation = "tanh"),
    BatchNormalization(),
    Dropout(dropout_pc),
    Dense(n_classes, activation = "softmax")
    ])

'''
Below, the model is compiled and a summary is printed. The model can then be fit to the training data
'''

model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics= ['acc'])
print(model.summary())

history = model.fit(x = trainX, y = trainY, batch_size= batchsize, epochs= epochs, verbose= 2, validation_data=[testX, testY])

'''
Predictions from the classification model for the training and test datasets are prepared below.
'''

train_pred = model.predict_classes(trainX).reshape(len(trainX),1)
test_pred = model.predict_classes(testX).reshape(len(testX), 1)

print(np.unique(train_pred), np.unique(test_pred), train_pred.shape, test_pred.shape)


'''
Below we calculate the following model metrics:

1. Precision
2. Recall
3. F1 Score
4. Confusion matrix

'''

traintrue = np.argmax(trainY, axis = 1)
testtrue = np.argmax(testY, axis= 1)

precision1, recall1, f11 = precision_score(testtrue, test_pred, average='macro'), recall_score(testtrue, test_pred, average='macro'), f1_score(testtrue, test_pred, average='macro')
precision2, recall2, f12 = precision_score(testtrue, test_pred, average='micro'), recall_score(testtrue, test_pred, average='micro'), f1_score(testtrue, test_pred, average='micro')

print("Macro metrics:", precision1, recall1, f11)
print("Micro metrics:", precision2, recall2, f12)

print(confusion_matrix(traintrue, train_pred))
print(confusion_matrix(testtrue, test_pred))