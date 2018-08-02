import VGG
import tensorflow as tf
import time
from datetime import datetime

# Basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")

with tf.Graph().as_default():
  global_step = tf.train.get_or_create_global_step()

  vgg = VGG.VGG("../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/train.npz")
  vgg.split({'train':6, 'test':4})
  data, labels = vgg.dic_to_inputs(vgg.datadict['train'])

  logits = vgg.build(data)
  loss = vgg.loss(logits, labels)
  print "train op:"
  train_op = vgg.train(loss, global_step)
  print(train_op)

  with tf.Session() as sess:
    print("TensorBoard section. ")
    writer = tf.summary.FileWriter('./summary')
    writer.add_graph(train_op.graph)
    writer.close

  print("train section. ")
  print loss
  print tf.train.SessionRunArgs(loss)
  # with tf.train.MonitoredTrainingSession(
  #   checkpoint_dir=FLAGS.train_dir,
  #   hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), 
  #   _LoggerHook()], save_checkpoint_secs=120) as mon_sess:
  #   print("training...")
  #   print (mon_sess.should_stop())
  #   while not mon_sess.should_stop():
  #     print (mon_sess.should_stop())
  #     mon_sess.run(train_op)