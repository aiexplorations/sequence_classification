import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from data_gen import gen_data_labels

'''
Generating and segmenting data

'''

seqlength = 1000
samples = 10**4

data, labels = gen_data_labels( num_sequences = samples, seq_length = seqlength)
print(data.shape, labels.shape)
print (labels)

# Scaling the data

scaler = StandardScaler()
data_scaled = scaler.fit_transform(np.nan_to_num(data))

data_reshaped = np.reshape( data_scaled, (data_scaled.shape[0], data_scaled.shape[1], 1))
labels_ohe = pd.get_dummies(pd.Series(labels)).values
print(labels_ohe.shape)

# Splitting the data

trainX, testX, trainY, testY = train_test_split(data_reshaped, labels_ohe, test_size = 0.2)


'''
Defining a Sequential() Keras model
'''

np.random.seed(123)

batchsize = 16
epochs = 16
n_units = 16
n_hidden = 16
lr = 1e-3
decay = 0.05 * lr
dropout_pc = 0.1
l1, l2 = 0.1, 0.1
n_classes = len(np.unique(labels))

model = Sequential([
    Conv1D(filters = n_units, kernel_size = (5), input_shape = (seqlength, 1), padding = 'same', kernel_regularizer = l1_l2(l1, l2), activation = "relu"),
    MaxPooling1D(pool_size = 5),
    Conv1D(filters = n_hidden*2, kernel_size = (4), kernel_regularizer = l1_l2(l1, l2), padding = 'same', activation = "relu"),
    MaxPooling1D(pool_size = 4),
    Conv1D(filters = n_hidden*4, kernel_size = (3), kernel_regularizer = l1_l2(l1, l2), padding = 'valid', activation = "relu"),
    MaxPooling1D(pool_size = 3),
    Flatten(),
    Dense(n_hidden, activation = "tanh"),
    BatchNormalization(),
    Dropout(dropout_pc),
    Dense(n_hidden, activation = "tanh"),
    BatchNormalization(),
    Dropout(dropout_pc),
    Dense(n_classes, activation = "softmax")
    ])

model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics= ['acc'])
print(model.summary())

history = model.fit(x = trainX, y = trainY, batch_size= batchsize, epochs= epochs, verbose= 1, validation_data=[testX, testY])

train_pred = model.predict_classes(trainX).reshape(len(trainX),1)
test_pred = model.predict_classes(testX).reshape(len(testX), 1)
print(testY.shape, test_pred.shape)

testvals = np.argmax(test_pred, axis= 1)
test_true = np.argmax(testY, axis= 1)

trainvals = np.argmax(train_pred, axis = 1)
train_true = np.argmax(trainY, axis = 1)

precision1, recall1, f11 = precision_score(test_true, testvals, average='macro'), recall_score(test_true, testvals, average='macro'), f1_score(test_true, testvals, average='macro')
precision2, recall2, f12 = precision_score(test_true, testvals, average='micro'), recall_score(test_true, testvals, average='micro'), f1_score(test_true, testvals, average='micro')

print("Macro metrics:", precision1, recall1, f11)
print("Micro metrics:", precision2, recall2, f12)

print(confusion_matrix(train_true, trainvals))
print(confusion_matrix(test_true, testvals))