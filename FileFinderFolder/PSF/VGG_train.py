import VGG
import tensorflow as tf

vgg = VGG.VGG("./MFCCData_folder/MFCCData.npz")
vgg.split({'train':6, 'test':4})
# vgg_m = VGG.VGG("./MFCCData_folder/MFCCData_merged.npz")
vgg_train = VGG.VGG("./MFCCData_folder/MFCCData_split/train.npz")

def train():
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # TODO: Get data, labels
    data, labels = vgg_train.dic_to_inputs(vgg_train.datadict)

    # TODO: Pass data into [build]
    # TODO: Calculate loss
    # TODO: Execute train_step with loss and global_step and build new graph

    logits = vgg_train.build(data)

    loss = vgg_train.loss(logits, labels)

    train_op = vgg_train.train(loss, global_step)
    
    # vgg_train.build(tf.placeholder(tf.float32, shape=(14, 1568, 1, 1)))

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./summary')
  writer.add_graph(tf.get_default_graph())
  writer.close

