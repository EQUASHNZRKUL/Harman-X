import VGG
import tensorflow as tf

vgg = VGG.VGG("./MFCCData")
vgg.build(tf.placeholder(tf.float32, shape=(14, 1568, 3, 3)))
with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
    writer.close
