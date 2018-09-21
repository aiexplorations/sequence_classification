'''

Data Generation

Purpose: Generate sequences of univariate data with and without special values or change points
    1. Sequence type A: common cause variation only
    2. Sequence type B: common cause variation with a change point (special cause process change)
    3. Sequence type C: anomalous sequences in a time series (occasional spikes with mostly common cause variation traffic)

Author: Rajesh S (@rexplorations)
Email: rexplorations@gmail.com

'''




import numpy as np
import pandas as pd

def gen_data_labels(num_sequences = 10**3, seq_length = 100):

    '''
    :param arg1: number of sequences
    :param arg2: sequence length (uniform length)

    :returns: numpy arrays for data and labels
    
    **Data generation function**

    This is a script which generates training data for sequence classification problems.

    1. Sequence type A: common cause variation only
    2. Sequence type B: common cause variation with a change point (special cause process change)
    3. Sequence type C: anomalous sequences in a time series (occasional spikes with mostly common cause variation traffic)

    The data is generated from all these different kinds of sequences and the associated labels are generated
    '''
    master_loc = 10.0
    master_scale = 1



    def gen_type_a(num = int(num_sequences/3)-1):
        benign_seq = np.random.normal(loc=10.0, scale = 0.1, size = seq_length)
        return np.reshape(benign_seq, (len(benign_seq),1))

    def gen_type_b(num = int(num_sequences/3)-1):
        seq1 = np.random.normal(loc= 10.0, scale = 0.1, size = int(seq_length/2))
        seq2 = np.random.normal(loc= 10.0 + np.random.normal(20, 0.01, 1), scale = 1 + np.random.normal(0.5, 0.01, 1), size = int(seq_length/2))
        change_point_seq = np.concatenate([seq1, seq2])
        return np.reshape(change_point_seq, (len(change_point_seq), 1))
    
    def gen_type_c(num = int(num_sequences/3), anomalous_seq_length = int(seq_length/5)):
        raw_seq = np.random.normal(loc=10.0, scale = 0.1, size = int(seq_length))

        randlen = int(np.random.randint(low = 10, high = 20, size = 1))
        randpos = int(np.random.randint(low = 0, high = seq_length - randlen, size = 1))
        
        #print(randpos.shape, randlen.shape)
        anomalous_seq = np.random.normal(500, 0.1, randlen)
        final_anomalous_seq = raw_seq
        final_anomalous_seq[randpos: randpos+randlen] = anomalous_seq
        return np.reshape(final_anomalous_seq, (len(final_anomalous_seq), 1))

    type_a_count, type_b_count, type_c_count = int(num_sequences/3)-1, int(num_sequences/3)-1, int(num_sequences/3)
    type_a_data, type_b_data, type_c_data = np.empty((seq_length, 1)), np.empty((seq_length, 1)), np.empty((seq_length, 1))

    for count in range(type_a_count):
        type_a_data = np.hstack( [type_a_data, gen_type_a()] )
    
    for count in range(type_b_count):
        type_b_data = np.hstack( [type_b_data, gen_type_b()] )
    
    for count in range(type_c_count):
        type_c_data = np.hstack( [type_c_data, gen_type_c()] )
    
    data = np.concatenate([type_a_data, type_b_data, type_c_data], axis= 1).T
    labels = np.hstack( (np.repeat(0, type_a_count+1), np.repeat(1, type_b_count+1), np.repeat(2, type_c_count+1)) )
    
    return data, labels