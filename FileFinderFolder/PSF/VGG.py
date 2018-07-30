import numpy as np 
import math
import tensorflow as tf 
import tensorflow.layers as l
import tensorflow.nn
import os

from random import seed, choice, shuffle

# GLOBALS
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

class VGG:
  def __init__(self, dir=None):
    """ Instantiates a VGG object, which holds the data and any metadata of the 
    VGG Neural Network. 
    Requires:
    - [dir]: is a valid directory of an .npz file
    """
    if dir is None:
      self.datadict = {}
    else:
      self.datadict = {}
      name = _get_filename(dir)
      datadict = np.load(dir)
      self.mapping = datadict.keys()
      i = 0

      # need to translate keys into ints and store a mapping
      # this is necessary now because they will be randomized later
      for _, v in datadict.iteritems():
        self.datadict[name][i] = v
        i += 1

  def _get_filename(dir):
    """ [_get_filename: (str -> str)] is the name of the file found at [dir] 
    e.g.) '/Users/justinkae/.../train.npz' -> 'train'
    """
    k = dir.rfind('.')
    h = dir.rfind('/')
    return dir[h+1:k]

  def split(self, biasdict):
    """ [split] splits up self.datadict into separate dictionaries weighted
      according to [biasdict].
    Requires:
    - [self.datadict]: Has only one key. 
    - [biasdict]: keys are the titles of the categories (must contain 'train')
      & values are weights of the categories. e.g. {train:1, test:9} would mean
      1/10 prob each datapoint is categorized as 'train', and 9/10 for 'test'.
    """
    # Unrolls the biasedlist & wipes the biasdict
    seed(1)
    biasedlist = []
    for k, v in biasdict.iteritems():
      biasedlist = biasedlist + [k] * v
      biasdict[k] = {}

    # Reveals the [d] master dict & processes categorization 
    d = self.datadict[self.datadict.keys()[0]]
    for cmd, mfcc_array in d.iteritems():
      for mfcc in mfcc_array:
        classification = choice(biasedlist)
        try:
          biasdict[classification][cmd] = np.append(biasdict[classification][cmd], [mfcc], axis=0)
        except KeyError:
          biasdict[classification][cmd] = np.array([mfcc])
    
    self.datadict = biasdict

  def load_npz(self, dir):
    """ [load_npz] stores the contents of an .npz file as a separate dataset 
    into the object's datadict. 
    Requires:
    - [dir] is a valid directory of an .npz file, and the name of the file can't
      already exist in the datadict. 
    """
    d = np.load(dir)
    name = _get_filename(dir)
    self.datadict[name] = d


  def _variable_on_cpu(self, name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      dtype = tf.float32
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

  def _variable_with_weight_decay(self, name, shape, stddev=5e-2):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = self._variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    return var

  def build(self, input):
    """ [build] constructs the Neural Network structure with TensorFlow. 
    """
    # Layer 1:
    self.conv3_1 = self.conv_node(input, 3, 64, "conv3_1")
    self.mpool_1 = self.max_pool(self.conv3_1, "mpool_1")

    # Layer 2:
    self.conv3_2 = self.conv_node(self.mpool_1, 3, 512, "conv3_2")
    self.conv3_3 = self.conv_node(self.conv3_2, 3, 4096, "conv3_3")
    self.mpool_2 = self.max_pool(self.conv3_3, "mpool_2")

    # Final Layer:
    self.fc_1 = self.conv_node(self.mpool_2, 1, 4096, "fc_1")
    self.fc_2 = self.conv_node(self.fc_1, 1, 4096, "fc_2")
    self.fc_3 = self.conv_node(self.fc_2, 1, 1000, "fc_3")

    self.output = tf.nn.softmax(self.fc_3, name="output")

    # self.data_dict = None

  def avg_pool(self, input, name):
    return tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)  

  def max_pool(self, input, name):
    print input.name + ": " + str(input.shape)
    return tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'SAME', name=name)

  def max_layer(self, input, name):
    return l.max_pooling2d(inputs=input, pool_size=[2,2], strides=2, name=name)

  def conv_layer(self, input, filters, name):
    # TODO: might get issue with this line later: 
    # ValueError: Negative dimension size caused by subtracting 3 from 1 for 
    # 'conv2d/Conv2D' (op: 'Conv2D') with input shapes: [14,1568,1,1], [3,3,1,64].
    return l.conv2d(input, filters, [3,3], [1,1], activation=tf.nn.relu) 

  def conv_node(self, input, size, filters, name):
    with tf.variable_scope(name) as scope:
      print input.name + ": " + str(input.shape)
      in_dim = input.shape[3]
      kernel = self._variable_with_weight_decay('weights', shape=[size, size, in_dim, filters])
      # print in_dim
      conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding='SAME')
      biases = self._variable_on_cpu('biases', [filters], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv_1 = tf.nn.relu(pre_activation, name=scope.name)
      return conv_1
      
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

  # TODO: TRANSLATE THE DICTIONARY INTO INPUTS
  def dic_to_inputs(self, dic):
    """ Translates the data dictionary [dic] from the .npz file into a tf.Tensor
    Returns: [inputs] 4D tensor of [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]
             [labels] 1D tensor of [BATCH_SIZE] """
    data = []
    labels = []
    prep_assoc_list = []

    # Randomize order of inputs:
    for cmd, mfcc_list in dic.iteritems():
      for mfcc in mfcc_list:
        prep_assoc_list.append((cmd, mfcc))
    shuffle(prep_assoc_list)

    # Populate data & label lists
    for (cmd, val) in prep_assoc_list:
      data.append(val)
      labels.append(cmd)
    
    # Convert into tensors
    data = np.array(data)
    data = tf.constant(data, dtype=tf.float32, name='inputs')
    data = tf.expand_dims(data, 3)
    labels = np.array(labels)
    labels = tf.constant(labels, name='labels')

    return data, labels

  def loss(self, logits, labels):
    labels = tf.cast(labels, tf.int64)

    # Backpropagation sort of thing? TODO: read up on cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels = labels, logits = logits, name = 'cross_entropy_per_datapoint')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # total loss = cross_entropy plus all weight decay terms. 
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

  # TODO: Need both the training and eval data in the object at the same time
  def train_step(self, dataset, loss, step)
    # Collect variables that affect learning rate
    data_count = len(self.datadict['train'])
    class_count = len(self.mapping)
    num_batches_per_epoch = data_count / class_count 

    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    # Generate moving averages of all losses and associated summaries
    # Compute gradients
    # Apply gradients
    # Add histograms (optional)
    # Track moving averages of all trainable variables
    # return average variable operation 