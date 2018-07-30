import VGG
import tensorflow as tf

vgg = VGG.VGG("./MFCCData_Folder/MFCCData.npz")
# vgg_m = VGG.VGG("./MFCCData_merged.npz")
vgg.build(tf.placeholder(tf.float32, shape=(14, 1568, 1, 1)))

# TensorBoard graph:
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./summary')
    writer.add_graph(tf.get_default_graph())
    writer.close
