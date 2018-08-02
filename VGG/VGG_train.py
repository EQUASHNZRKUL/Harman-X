import VGG
import tensorflow as tf
import time

# Basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")

vgg_train = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/train.npz")

with tf.Graph().as_default():
  global_step = tf.train.get_or_create_global_step()

  vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData.npz")
  vgg.split({'train':6, 'test':4})
  data, labels = vgg.dic_to_inputs(vgg.datadict['train'])

  logits = vgg.build(data)
  loss = vgg.loss(logits, labels)
  train_op = vgg.train(loss, global_step)
  print(vgg.batch_size)

  class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime"""
    def begin(self):
      self._step = -1
      self._start_time = time.time()
    
    def before_run(self, run_context):
      self._step += 1
      return tf.train.SessionRunArgs(loss) # Asks for loss value. 

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

  with tf.Session() as sess:
    print("TensorBoard section. ")
    writer = tf.summary.FileWriter('./summary')
    writer.add_graph(train_op.graph)
    writer.close

  print("train section. ")
  with tf.train.MonitoredTrainingSession(
    checkpoint_dir=FLAGS.train_dir,
    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
      tf.train.NanTensorHook(loss), 
      _LoggerHook()], 
      save_checkpoint_secs=120) as mon_sess:

    print("training...")
    while not mon_sess.should_stop():
      print(mon_sess.run(global_step))
      mon_sess.run(train_op)

