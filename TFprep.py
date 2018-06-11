from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#Graphs:
# A computation graph is a series of Tensor flow ops arranged into a graph
# Composed of two types of objs
#   - Ops: Nodes of graph. Describe calculations that consume or produce tensors
#   - Tensors: The edges in the graph. Rep values flowing through graph
#   - Most functions return tf.Tensors. 
# Operations:
#   - Most basic op: Constant. 
#     - Python func that builds the op takes tensor value as input, and resulting op takes no inputs
a = tf.constant(3.0, dtype= tf.float32)
b = tf.constant(4.0) #implicit cast to tf.float32
total = a + b
print(a)
print(b)
print(total)
#  total isn't 7.0, but an add operation instead. Interesting. 

#TensorBoard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

#Session
#  think of it as TF executable, and Graph is the code
sess = tf.Session()
print(sess.run({'ab':(a, b), 'total':total}))