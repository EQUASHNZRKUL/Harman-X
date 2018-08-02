import VGG
# import VGG_train
import tensorflow as tf
import time
from datetime import datetime

# GLOBALS:

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")

with tf.Graph().as_default():

  print("2. vgg_d: (w/ data)")
  print("3. instantiate: ")
  vgg_d = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/train.npz")
  # vgg_d.graph
  print(tf.Graph())

  print("4. splitting data into train and test")
  # vgg_d.split({'train':6, 'test':4})

  print("5. translating datadict train into tensors")
  data, labels = vgg_d.dic_to_inputs(vgg_d.datadict['train'])

  print(data.graph) 
  print(labels.graph)
  print(tf.get_default_graph)

  print("6. building network and obtaining logits")
  logits = vgg_d.build(data)

  print("7. getting loss operation")
  loss = vgg_d.loss(logits, labels)

  print(loss.graph)

  print("8. getting training op")
  global_step = tf.train.get_or_create_global_step()
  train_op = vgg_d.train(loss, global_step)

  print(train_op.graph)
  print(global_step)

  # LoggerHook:
  class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime"""
    def begin(self):
      self._global_step_tensor = global_step
      self._step = -1
      self._start_time = time.time()
    
    def before_run(self, run_context):
      self._step += 1
      return tf.train.SessionRunArgs(loss)

    def after_run(self, run_context, run_values):
      if self._step % FLAGS.log_frequency == 0:
        curr_time = time.time()
        duration = curr_time - self._start_time
        self._start_time = curr_time

        loss_value = run_values.results
        examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
        sec_per_batch = float(duration / FLAGS.log_frequency)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
        print((format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch)))

  print("train section. ")
  with tf.train.MonitoredTrainingSession(
    checkpoint_dir=FLAGS.train_dir,
    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
      tf.train.NanTensorHook(loss), 
      _LoggerHook()]) as mon_sess:

    print("TensorBoard section. ")
    writer = tf.summary.FileWriter('./summary')
    writer.add_graph(train_op.graph)
    writer.close

    print("training... ")
    print(mon_sess.run(global_step))
    while not mon_sess.should_stop():
      print(mon_sess.run(global_step))
      mon_sess.run(train_op)