import VGG
import tensorflow as tf
import numpy as np 
from time import sleep

# ../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval_logs/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")                
                    
def eval_step(saver, top_k_op):
  """ Runs eval once

  Requires:
  - ckpt_dir: directory to store the checkpoints
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start Queue Runners
    coord = tf.train.Coordinators()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS): # worried about QR
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

      # Instantiate the loop variables. 
      # TODO: translate FLAGS.checkpoint_dir, num_examples, batch_size
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0 # Counts #correct predictions
      total_sample_count = FLAGS.num_examples
      step = 0

      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        total_sample_count = num_iter * FLAGS.batch_size
        step += 1
      
      # Post-processing: Compute precision @ 1
      precision = true_count / total_sample_count
      print(("precision @ 1: %3f" % precision))

    except Exception as e: #pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

"""Eval CIFAR-10 for a number of steps."""
with tf.Graph().as_default() as g:
  # Get images and labels for CIFAR-10.
  # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
  # vgg.split({'train':6, 'test':4})
  vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz")
  print "78"
  data, labels = vgg.dic_to_inputs(vgg.datadict['test'])
  print "80"

  # Build Graph to compute logit predictions
  logits = vgg.build(data)

  # Calculate predictions
  top_k_op = tf.nn.in_top_k(logits, labels, 1)

  # Restore the moving average version of the learned vars for eval. 
  variable_averages = tf.train.ExponentialMovingAverage(VGG.MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  # Build summary op based on collection of Summaries.
  # summary_op = tf.summary.merge_all()
  # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

  while True:
    eval_step(saver, top_k_op)
    if FLAGS.run_once:
      break
    sleep(FLAGS.eval_interval_secs)

