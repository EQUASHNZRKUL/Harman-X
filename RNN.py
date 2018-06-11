from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# define data
# translate data into tf format
# make model/layers
# make loss tensor and define training op
# initialize layer
# run session/start execution
# training
# run 
# TensorBoard

# define inputs
# Placeholder for the inputs in a given iteration. 
words = tf.placeholder(tf.int32, [batch_size, num_steps])
# define embedding matrix (initialized randomly, but trained with data)
# embedding_matrix is a tensor of shape [vocabulary_size, embedding_size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

# # make model/layer
# lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size) #makes a LSTM Cell
# initial_state = state = tf.zeros([batch_size, lstm.state_size])

# make multi-model/layers
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)]) # makes <nuymber_of_layers> LSTMs layers

for i in range(num_steps):
    # value of state updated after processing each batch
    output, state = lstm(words[:, i], state)
    # output is normal output of layer, state is the new state, since RNNs need memory

    # initiate state tensor
    numpy_state = initial_state.eval()

    # define loss, training op
    total_loss = 0.0
    for current_batch_of_words in words_in_dataset:
        # run session/start execution
        numpy_state, current_loss = session.run([final_state, loss], 
            # Initialize the LSTM state from the previous iteration.
            feed_dict = {initial_state: numpy_state, words: current_batch_of_words})
        total_loss += current_loss

    # make loss tensor and define training op
    # initialize layer
    # run session/start execution
    # training
    # run
final_state = state