import VGG
import tensorflow as tf
import time
import math
from datetime import datetime

# Basic model parameters
FLAGS = tf.app.flags.FLAGS

# Training Flags
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('log_frequency', 5,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 30,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Eval Flags
tf.app.flags.DEFINE_string('eval_dir', './eval_logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 300,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")  

def train(vgg=None):
  """ Trains the network. 

  Prepares the network [vgg] or if [vgg] is None, builds a vgg and loads it
  with its training data. Calculates its loss with default values and calls the
  train function on it in a MonitoredTrainingSession, making checkpoints in
  [FLAGS.train_dir], and logging every [FLAGS.log_frequency] steps until 
  [FLAGS.max_steps] steps elapse. 
  Requires: 
  - [vgg]: has a ['train'] dataset in its datadict. 
  """
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/train.npz")
    if vgg is None:
      vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
      vgg.split({'train':1, 'test':1})
    data, labels = vgg.dic_to_inputs(vgg.datadict['train'])

    logits = vgg.build(data)
    loss = vgg.loss(logits, labels)

    train_op = vgg.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                tf.train.NanTensorHook(loss),
                _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
    # with tf.Session() as sess:
    #   sess.run(train_op)

def eval_step(saver, summary_writer, top_k_op, summary_op):
  """ Evaluates the vgg once. 

  Evaluates the calculated labels to their true labels and calculates the 
  predictive precision of the model. Saves the results and prints the precision.
  Uses the checkpoint found in [FLAGS.checkpoint_dir]

  Requires:
  - [saver]: A saver
  - [summary_writer]: A summary writer to log the eval data. 
  - [top_k_op]: predictions
  - [summary_op]: The summary operator corresponding to the summary_writer
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
    coord = tf.train.Coordinator()
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
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

    except Exception as e: #pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def evaluate(vgg=None):
  """Eval CIFAR-10 for a number of steps.
  
  Similar to train(), evaluate() builds the [vgg], and if None builds its own.
  Gathers data from 'test' dataset of the vgg's datadict and builds a graph and
  calculates predictions from the trained set. Runs eval_step() on the model 
  until manual interruption. 
  Requires: 
  - [vgg]: contains a ['test'] dataset  
  """
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz")
    if vgg is None:
      vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
      vgg.split({'train':1, 'test':1})
    data, labels = vgg.dic_to_inputs(vgg.datadict['test'])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = vgg.build(data)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        VGG.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_step(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
  print " Initializing the Network... "
  vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_merged.npz")
  vgg.split({'train':6, 'test':4})

  print " Training... "
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train(vgg)

  print " Eval..."
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(vgg)

if __name__ == '__main__':
  tf.app.run()