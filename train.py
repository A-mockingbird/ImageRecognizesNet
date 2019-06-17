import vgg16
import tensorflow as tf
with tf.Session() as sess:
  vgg = vgg16.vgg16(sess, 9)
  re = vgg._build_network(True)