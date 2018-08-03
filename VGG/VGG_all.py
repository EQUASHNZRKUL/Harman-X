import VGG
import tensorflow as tf
import time
from datetime import datetime

# Basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('eval_dir', './eval_logs',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 300,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")  

def train():
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/train.npz")
    # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
    # vgg.split({'train':1, 'test':4})
    data, labels = vgg.dic_to_inputs(vgg.datadict['train'])

    logits = vgg.build(data)
    loss = vgg.loss(logits, labels)

    train_op = vgg.train(loss, global_step)

    # print ("---data: ")
    # print (data)
    # print ("---logits: ")
    # print (logits)
    # print ("---loss: ")
    # print (loss)

    # print ("---train_op: ")
    # print(train_op)

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

    # print (logits)
    # print("train section. ")
    # print loss
    # print tf.train.SessionRunArgs(loss)
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

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz")
    # vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
    # vgg.split({'train':1, 'test':1})
    data, labels = vgg.dic_to_inputs(vgg.datadict['test'])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = vgg.build(data)

    # print "data: "
    # print data.shape
    # print "logits: "
    # print logits.shape
    # print "labels: "
    # print labels.shape

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
      sleep(FLAGS.eval_interval_secs)

def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()