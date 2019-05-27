import tensorflow as tf
class vgg16(Object):
    def init(self):
        self.sess = sess
        self.conv_size = 3
        self.pool_size = 2
        self.input = tf.placeholder({})
    def build_network(self, scope='vgg16'):
        with tf.name_scope(scope):
            conv1 = self.conv_op()
           
    def conv_op(self, conv_input, channel, depth, num):
        kernel = tf.get_variable("conv_kernel"+num, [self.conv_size, self.conv_size, channel, depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("conv_bias"+num, [depth], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d()