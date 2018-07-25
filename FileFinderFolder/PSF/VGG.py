import numpy as np 
import math
import tensorflow as tf 
import layers as l
import tf.nn
import os

class VGG:
  def __init__(self, dir):
    self.datadict = {}
    for filename in os.listdir(dir):
      if filename.find(".DS_Store") != -1: 
        dic = np.load(filename)
        v = dic["arr_0"]
        key = filename[:-4]
        self.datadict[key] = []
        for elt in v:
          self.datadict[key].append(elt)

  def build(self, input):
    self.conv3_1 = self.conv_layer(input, "conv3_1")
    self.mpool_1 = self.max_pool(self.conv3_1, "mpool_1")

    self.conv3_2 = self.conv_layer(self.mpool_1, "conv3_2")
    self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
    self.mpool_2 = self.max_pool(self.conv3_3, "mpool_2")

    self.fc_1 = self.fc_layer(self.mpool_2, "fc_1")
    self.fc_2 = self.fc_layer(self.fc_1, "fc_2")
    self.fc_3 = self.fc_layer(self.fc_2, "fc_3")

    self.output = tf.nn.softmax(self.fc_3, name="output")

    self.data_dict = None

  def avg_pool(self, input, name):
    return tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)  

  def max_pool(self, input, name):
    return tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)

  def conv_layer(self, input, name):
    with tf.variable_scope(name):
      filter = self.get_conv_filter(name)
      conv = tf.nn.conv2d(input, filt, [1,1,1,1], padding='SAME')
      conv_bias = self.get_bias(name)
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
      weight = self.get_fc_weight(name)
      bias = self.get_bias(name)
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