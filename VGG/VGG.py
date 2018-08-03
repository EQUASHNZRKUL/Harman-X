import numpy as np 
import math
import tensorflow as tf 
import tensorflow.nn
import os
import re
import sys

from random import seed, choice, shuffle

# GLOBALS
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 900
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 600
TOWER_NAME = 'tower'

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz',
                           """Path to the CIFAR-10 data directory.""")

def _get_filename(dir):
  """ [_get_filename: (str -> str)] is the name of the file found at [dir] 
  e.g.) '/Users/justinkae/.../train.npz' -> 'train'
  """
  k = dir.rfind('.')
  h = dir.rfind('/')
  return dir[h+1:k]

def _dict_length(d):
  l = 0
  for _, v in d.items():
    l += len(v)
  return l

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  print ("--loss_averages_op: ")
  print (loss_averages_op)

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

class VGG:
  def __init__(self, dir=None):
    """ Instantiates a VGG object, which holds the data and any metadata of the 
    VGG Neural Network. 
    Requires:
    - [dir]: is a valid directory of an .npz file
    """
    self.datadict = {}
    self.mapping = {}
    self.batch_size = 0
    print(tf.Graph())
    if dir is not None:
      self.datadict = {}
      self.mapping = {}
      name = _get_filename(dir)
      datadict = np.load(dir)
      self.size = _dict_length(datadict)
      i = 0

      # need to translate keys into ints and store a mapping
      # this is necessary now because they will be randomized later
      for k, v in datadict.items():
        try:
          self.datadict[name][i] = v
          self.mapping[k] = i 
        except KeyError:
          name_dict = {}
          name_dict[i] = v
          self.datadict[name] = name_dict
        i += 1
    print "mapping: "
    print (self.mapping)
  
  # -=- GETTERS & SETTERS -=-

  def get_train_dict(self):
    """ Returns the training dataset from the data dictionary. None if datadict
    doesn't contain a 'train' key yet. """
    try:
      return self.datadict['train']
    except KeyError:
      return None

  def load_npz(self, dir):
    """ [load_npz] stores the contents of an .npz file as a separate dataset 
    into the object's datadict. 
    Requires:
    - [dir] is a valid directory of an .npz file, and the name of the file can't
      already exist in the datadict. 
    """
    d = np.load(dir)

    # Adjusting metadata
    self.size += _dict_length(d)
    self.mapping | set(d.keys())

    # Loading data into object
    name = _get_filename(dir)
    self.datadict[name] = d

  # -=- HELPERS -=-

  # def load_file(self, npz):
  #   key = npz[:-4]
  #   try:
  #     val = self.datadict[key]
  #   except KeyError:
  #     val = []
  #   dic = np.load(npz)
  #   v = dic["arr_0"]
  #   for elt in v:
  #     self.datadict[key] = val.append(elt)

  def split(self, biasdict, eq_len=False):
    """ [split] splits up self.datadict into separate dictionaries weighted
      according to [biasdict].
    Requires:
    - [self]: split() has not been called on this object yet. 
    - [self.datadict]: Has only one key. 
    - [biasdict]: keys are the titles of the categories (must contain 'train')
      & values are weights of the categories. e.g. {train:1, test:9} would mean
      1/10 prob each datapoint is categorized as 'train', and 9/10 for 'test'.
    - [eq_len]: specifies if its okay if the lengths of the dicts aren't equal.
    """
    # Unrolls the biasedlist & wipes the biasdict
    seed(1)
    biasedlist = []
    for k, v in biasdict.items():
      biasedlist = biasedlist + [k] * v
      biasdict[k] = {}

    # Reveals the [d] master dict & processes categorization 
    d = self.datadict[list(self.datadict.keys())[0]]
    for cmd, mfcc_array in d.items():
      for mfcc in mfcc_array:
        classification = choice(biasedlist)
        try:
          biasdict[classification][cmd] = np.append(biasdict[classification][cmd], [mfcc], axis=0)
        except KeyError:
          biasdict[classification][cmd] = np.array([mfcc])
    
    eq_len_bool = False
    length = len(biasdict[biasedlist[0]])
    for _, v in biasdict:
      eq_len_bool or (len(v) == length)
      self.datadict = biasdict

  def dic_to_inputs(self, dic):
    """ Translates the dataset [dic] from the .npz file into a tf.Tensor
    Requires:
    - [dic]: is a dataset within the datadict.
    Returns: [inputs] 4D tensor of [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]
             [labels] 1D tensor of [BATCH_SIZE] """
    data = []
    labels = []
    prep_assoc_list = []

    # Randomize order of inputs:
    for cmd, mfcc_list in dic.items():
      for mfcc in mfcc_list:
        prep_assoc_list.append((cmd, mfcc))
    shuffle(prep_assoc_list)

    # Populate data & label lists
    for (cmd, val) in prep_assoc_list:
      data.append(val)
      labels.append(cmd)
    
    # Convert into tensors
    raw_data = np.array(data)
    raw_data = tf.constant(raw_data, dtype=tf.float32, name='inputs')
    # print(raw_data)
    data = tf.expand_dims(raw_data, 3)
    # print(data)
    # data = tf.constant(data, dtype=tf.float32, name='inputs')
    labels = np.array(labels)
    labels = tf.constant(labels, name='labels')

    # Update batch_size field
    self.batch_size = labels.shape[0]

    return data, labels

# -=- BIG BOY METHODS -=-

  def build(self, input):
    """ [build] constructs the Neural Network structure with TensorFlow. 
    Requires:
    - [input]: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]
    Returns:
    - [logits]: A 1D tensor of [batch_size]
    """
    # Layer 1:
    self.conv3_1 = self._conv_node(input, 3, 4, "conv3_1")
    self.mpool_1 = self._max_pool(self.conv3_1, "mpool_1")

    # Layer 2:
    self.conv3_2 = self._conv_node(self.mpool_1, 3, 16, "conv3_2")
    self.conv3_3 = self._conv_node(self.conv3_2, 3, 32, "conv3_3")
    self.mpool_2 = self._max_pool(self.conv3_3, "mpool_2")
    print ("------=======------")
    print (self.mpool_2.shape)

    # Reshape Layer:
    with tf.variable_scope('fc_1') as scope:
      # self.resize = self._reshape_node(self.mpool_2, length, "resize")
      self.reshape = tf.reshape(self.mpool_2, [input.get_shape().as_list()[0], -1])
      dim = self.reshape.get_shape()[1].value
      weights = self._variable_with_weight_decay('weights', shape=[dim, 100], 
                stddev=0.04, wd=0.004)
      biases = self._variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
      self.fc_1 = tf.nn.relu(tf.matmul(self.reshape, weights) + biases, name=scope.name)
      _activation_summary(self.fc_1)

    # FC Layers:
    # self.fc_1 = self._local_layer(self.reshape, 100, "fc_1")

    self.output = self._output_layer(self.fc_1, name="softmax_linear")

    return self.output

  # BUILD: VARIABLE PRODUCERS
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

  def _variable_with_weight_decay(self, name, shape, stddev=5e-2, wd=None):
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
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var

  # BUILD: LAYER PRODUCERS
  def _avg_pool(self, input, name):
    return tf.nn.avg_pool(input, [1,2,2,1], [1,2,2,1], 'VALID', name=name)  

  def _max_pool(self, input, name):
    # print(input.name + ": " + str(input.shape))
    return tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'VALID', name=name)

  def _conv_node(self, input, size, filters, name):
    with tf.variable_scope(name) as scope:
      # print(input.name + ": " + str(input.shape))
      in_dim = input.shape[3]
      kernel = self._variable_with_weight_decay('weights', shape=[size, size, in_dim, filters])
      # print in_dim
      conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding='VALID')
      biases = self._variable_on_cpu('biases', [filters], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv_1 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(conv_1)
      return conv_1

  def _reshape_node(self, input, length, name):
    with tf.variable_scope(name) as scope:
      # print(input.name + ": " + str(input.shape))
      # Convert to a 2D array (layers of lists) so we can perform a single matr*
      self.reshape = tf.reshape(input, [length, -1])
      dim = self.reshape.get_shape()[1].value
      # print(dim)
  
  def _local_layer(self, input, length, name):
    with tf.variable_scope(name) as scope:
      # print(input.name + ": " + str(input.shape))
      dim = input.get_shape()[1].value
      weights = self._variable_with_weight_decay('weights', shape=[dim, length], 
                stddev=0.04)
      biases = self._variable_on_cpu('biases', [length], tf.constant_initializer(0.1))
      local = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
      _activation_summary(local)
      return local
  
  def _output_layer(self, input, name):
    with tf.variable_scope('output') as scope:
      dim = input.get_shape()[1].value
      num_classes = len(list(self.mapping.keys()))+1
      weights = self._variable_with_weight_decay('weights', shape=
                [dim, num_classes], stddev=(1.0/dim))
      biases = self._variable_on_cpu('biases', [num_classes], 
                tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(input, weights), biases, name=name)
      _activation_summary(softmax_linear)
      return softmax_linear

  def loss(self, logits, labels):
    """ Calculates the loss of the calculated [logit] values and true [labels]
    Returns: scalar representing the total loss. 
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    print ("--labels: ")
    print (labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   labels = labels, logits = logits, name = 'cross_entropy_per_datapoint')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # # print "cross_entropy_mean"
    # # print cross_entropy_mean
    # tf.add_to_collection('losses', cross_entropy_mean)

    # # total loss = cross_entropy plus all weight decay terms. 
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')

  def train(self, total_loss, global_step):
    """ Trains the model. 
    Creates an optimizer and applies to all trainable variables. Adds moving avg
    for trainable vars. 
    Requires: 
    - [total_loss]: Total loss calculated from [loss()].
    - [global_step]: Integer Variables counting elapsed iterations of training steps. 
    Returns:
    - [trian_op]: training operation tensor. 
    """
    # Collect variables that affect learning rate
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries
    loss_avgs_op = _add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_avgs_op]):
      opt = tf.train.GradientDescentOptimizer(lr)
      # print opt
      grads = opt.compute_gradients(total_loss)
      # print grads

    # Apply gradients
    apply_grad_op = opt.apply_gradients(grads, global_step=global_step)
    # print apply_grad_op 

    # Add histograms (optional)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + 'gradients', grad)

    # Track moving averages of all trainable variables
    variable_avgs = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_grad_op]):
      variable_avgs_op = variable_avgs.apply(tf.trainable_variables())

    return variable_avgs_op

    # return average variable operation 