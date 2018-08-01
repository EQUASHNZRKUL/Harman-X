import VGG
# import VGG_train
import tensorflow as tf
import time
from datetime import datetime

# GLOBALS:
LOG_FREQ = 10

with tf.Graph().as_default():
  # print "1. vgg_p: (w/ Placeholder)"
  # vgg_p = VGG.VGG("./MFCCData_folder/MFCCData.npz")
  # vgg_p.build(tf.placeholder(tf.float32, shape=(995, 1586, 14, 1)))

  # print vgg_p.datadict.keys()
  # print vgg_p.datadict['MFCCData'].keys()
  # print vgg_p.mapping

  print "2. vgg_d: (w/ data)"
  # Data (need to check if this ever worked) (think it did, not entirely sure)
  print "3. instantiate: "
  vgg_d = VGG.VGG("./MFCCData_folder/MFCCData.npz")

  # print vgg_d.datadict.keys()
  # print vgg_d.datadict['MFCCData'].keys()
  # print vgg_d.mapping

  print "4. splitting data into train and test"
  vgg_d.split({'train':6, 'test':4})

  # print vgg_d.datadict.keys()
  # print vgg_d.datadict['train'].keys()
  # print vgg_d.mapping

  print "5. translating datadict train into tensors"
  data, labels = vgg_d.dic_to_inputs(vgg_d.datadict['train'])

  print data.graph 
  print labels.graph
  print tf.get_default_graph

  # print data
  # print labels
  # print data.shape()

  print "6. building network and obtaining logits"
  logits = vgg_d.build(data)

  print logits
  print logits.graph

  print "7. getting loss operation"
  loss = vgg_d.loss(logits, labels)

  print loss.graph

  print "8. getting training op"
  global_step = tf.train.get_or_create_global_step()
  train_op = vgg_d.train(loss, global_step)

  print train_op.graph

  # # LoggerHook:
  # class _LoggerHook(tf.train.SessionRunHook):
  #   """Logs loss and runtime"""
  #   def begin(self):
  #     self._step = -1
  #     self._start_time = time.time()
    
  #   def before_run(self, run_context):
  #     self._step += 1
  #     return tf.train.SessionRunArgs(loss)

  #   def after_run(self, run_context, run_values):
  #     if self._step % LOG_FREQ == 0:
  #       curr_time = time.time()
  #       duration = curr_time - self._start_time
  #       self._start_time = curr_time

  #       loss_value = run_values.results
  #       examples_per_sec = LOG_FREQ * BATCH_SIZE / duration
  #       sec_per_batch = float(duration / LOG_FREQ)

  #       format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
  #                       'sec/batch)')
  #       print (format_str % (datetime.now(), self._step, loss_value,
  #                              examples_per_sec, sec_per_batch))

print "TensorBoard section. "
with tf.Session() as sess:
  writer = tf.summary.FileWriter('./summary')
  writer.add_graph(train_op.graph)
  writer.close