import VGG
# import VGG_train
import tensorflow as tf

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

print "TensorBoard section. "
with tf.Session() as sess:
  writer = tf.summary.FileWriter('./summary')
  writer.add_graph(train_op.graph)
  writer.close