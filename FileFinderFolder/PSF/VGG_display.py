import VGG
import tensorflow as tf

vgg = VGG.VGG("./MFCCData.npz")
vgg_m = VGG.VGG("./MFCCData_merged.npz")
vgg.build(tf.placeholder(tf.float32, shape=(14, 1568, 1, 1)))

# TensorBoard graph:
with tf.Session() as sess:
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    writer.close
