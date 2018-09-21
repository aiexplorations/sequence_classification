'''

Machine Learning Dataset Preparation

Purpose: Prepare a series of training and test numpy arrays (*.npy files) for sequence classification models.

Author: Rajesh S (@rexplorations)
Email: rexplorations@gmail.com


'''




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from data_gen import gen_data_labels


'''
Generating data using data_gen and preparing it for the LSTM model

'''

seqlength = 100
samples = 5*10**3

data, labels = gen_data_labels( num_sequences = samples, seq_length = seqlength)

# Scaling the data

scaler = StandardScaler()
data_scaled = scaler.fit_transform(np.nan_to_num(data))

n_samples, seq_size = data_scaled.shape[0], data_scaled.shape[1]

data_reshaped = np.reshape( data_scaled, (n_samples, seq_size, 1))
labels_ohe = pd.get_dummies(pd.Series(labels)).values
print(labels_ohe.shape)

trainX, testX, trainY, testY = train_test_split(data_reshaped, labels_ohe, test_size = 0.2)

np.save("data/trainX.npy", trainX)
np.save("data/trainY.npy", trainY)
np.save("data/testX.npy", testX)
np.save("data/testY.npy", testY)

