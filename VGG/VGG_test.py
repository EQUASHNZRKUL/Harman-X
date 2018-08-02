import VGG
from VGG_eval import eval_step
import tensorflow as tf
import time
from datetime import datetime

# GLOBALS:

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval_logs/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '../MFCCData_folder/MFCCData_split/test.npz',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")   
with tf.Graph().as_default() as g:
  # Get images and labels for CIFAR-10.
  # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
  # vgg.split({'train':6, 'test':4})
  print "1) Instantiate: "
  vgg = VGG.VGG(FLAGS.eval_data)
  data, labels = vgg.dic_to_inputs(vgg.datadict['test'])
  print vgg.datadict
  print data
  print labels

  # Build Graph to compute logit predictions
  print "2) Build graph"
  logits = vgg.build(data)
  print logits

  # Calculate predictions
  print "3) Get top_k_op"
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  print top_k_op

  # Restore the moving average version of the learned vars for eval. 
  print "Build variable averages"
  variable_averages = tf.train.ExponentialMovingAverage(VGG.MOVING_AVERAGE_DECAY)
  print variable_averages
  variables_to_restore = variable_averages.variables_to_restore()
  print variables_to_restore
  saver = tf.train.Saver(variables_to_restore)
  print saver

  # Build summary op based on collection of Summaries.
  # summary_op = tf.summary.merge_all()
  # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

  while True:
    eval_step(saver, top_k_op)
    if FLAGS.run_once:
      break
    sleep(FLAGS.eval_interval_secs)
