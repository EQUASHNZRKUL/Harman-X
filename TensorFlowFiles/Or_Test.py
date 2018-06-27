from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# define raw data
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [1]]

# convert to tensors
x = tf.constant(x, dtype = tf.float32)
y = tf.constant(y, dtype = tf.float32)

layer = tf.layers.Dense(units=1)

ypred = layer(x)
loss = tf.losses.mean_squared_error(labels=y, predictions = ypred)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
    _, lossEx = sess.run((train, loss))
    print(lossEx)

print(sess.run(ypred))