import VGG
import tensorflow as tf

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
  print vgg.batch_size

with tf.train.MonitoredTrainingSession(
  checkpoint_dir=FLAGS.train_dir,

) as sess:
  print(sess.run(tf.report_uninitialized_variables()))

  # writer = tf.summary.FileWriter('./summary')
  # writer.add_graph(train_op.graph)
  # writer.close

