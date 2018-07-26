import numpy as np 
import math
import tensorflow as tf 
import tensorflow.layers as l
import tensorflow.nn
from os import listdir

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope

class VGG:
  def __init__(self, dir):
    self.datadict = {}
    for filename in listdir(dir):
      if filename.find(".DS_Store") == -1: 
        dic = np.load(dir + "/" + filename)
        v = dic["arr_0"]
        key = filename[:-4]
        self.datadict[key] = []
        for elt in v:
          self.datadict[key].append(elt)

  def build(self, input):
    self.conv3_1 = self.conv_layer(input, 64, "conv3_1")
    self.mpool_1 = self.max_pool(self.conv3_1, "mpool_1")

    self.conv3_2 = self.conv_layer(self.mpool_1, "conv3_2")
    self.conv3_3 = self.conv_layer(self.conv3_2, 512, "conv3_3")
    self.mpool_2 = self.max_pool(self.conv3_3, 4096, "mpool_2")

    self.fc_1 = self.fc_layer(self.mpool_2, "fc_1")
    self.fc_2 = self.fc_layer(self.fc_1, "fc_2")
    self.fc_3 = self.fc_layer(self.fc_2, "fc_3")

    self.logits = tf.nn.softmax(self.fc_3, name="logits")

    self.data_dict = None

  def avg_pool(self, input, name):
    return tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)  

  def max_pool(self, input, name):
    return tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)

  def max_layer(self, input, name):
    return l.max_pooling2d(inputs=input, pool_size=[2,2], strides=2, name=name)

  def conv_layer(self, input, filters, name):
    # might get issue with this line later: 
    # ValueError: Negative dimension size caused by subtracting 3 from 1 for 
    # 'conv2d/Conv2D' (op: 'Conv2D') with input shapes: [14,1568,1,1], [3,3,1,64].
    return l.conv2d(input, filters, [3,3], activation=tf.nn.relu) 

  def conv_node(self, input, name):
    with tf.variable_scope(name):
      filt = self._get_conv_filter(name)
      conv = tf.nn.conv2d(input, filt, [1,1,1,1], padding='SAME')
      conv_bias = self._get_bias(name)
      bias = tf.nn.bias_add(conv, conv_bias)
      relu = tf.nn.relu(bias)
      return relu

  def fc_layer(self, input, name):
    with tf.variable_scope(name):
      shape = input.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim = dim * d
      x = tf.reshape(input, [-1, dim])
      weight = self._get_fc_weight(name)
      bias = self._get_bias(name)
      fc = tf.nn.bias_add(tf.matmul(x, weight), bias)
      return fc

  def load_file(self, npz):
    key = npz[:-4]
    try:
      val = self.datadict[key]
    except KeyError:
      val = []
    dic = np.load(npz)
    v = dic["arr_0"]
    for elt in v:
      self.datadict[key] = val.append(elt)

  def _get_conv_filter(self, name):
    return tf.constant(self.datadict[name][0], name="filter")

  def _get_bias(self, name):
    return tf.constant(self.data_dict[name][0], name="bias")

  def _get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name="weight")