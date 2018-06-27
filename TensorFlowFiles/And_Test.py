from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

# define data in Python syntax
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
yr = [[0], [0], [0], [1]]

# translate data into Tensors
x = tf.constant(x, dtype = tf.float32)
yr = tf.constant(yr, dtype = tf.float32)

# make model/layers
linear_model = tf.layers.Dense(units = 1, activation = tf.nn.sigmoid)
yp = linear_model(x)

# make loss tensor & training op
loss = tf.losses.mean_squared_error(labels=yr, predictions=yp)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# initialize layer
init = tf.global_variables_initializer()

# run session/start execution
sess = tf.Session()
sess.run(init)

print(sess.run(yp))

# training
for i in range(1000):
    _, loss_ex = sess.run((train, loss))
    print(loss_ex)

print(sess.run(yp))

#TensorBoard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())