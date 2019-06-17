import tensorflow as tf
class vgg16(object):
    def __init__(self, sess, num_classes):
        self.sess = sess
        self.conv_size = 3
        self.pool_size = 2
        self.input = tf.placeholder(tf.float32, [1, None, None, 3])
        self.label = tf.placeholder(tf.float32, [1, num_classes])
        self.num_classes = num_classes
        self.results = {}

    def _build_network(self, is_training):
        conv = self._build_head_net('vgg16')
        prediction_classification = self._build_tail_net(conv, scope="vgg16_classification")
        if is_training:
            loss = self.crossentropy_loss_compute(prediction_classification, self.label)
            self.results['loss'] = loss
        self.results['predictons'] = prediction_classification
        return prediction_classification

    def _build_head_net(self, scope='vgg16_conv'):
        with tf.name_scope(scope):
            conv1 = self.conv_op(self.input, 3, 64, 'conv_1')
            conv2 = self.conv_op(conv1, 64, 64, "conv_2")
            pool1 = self.pool_op(conv2)
            conv3 = self.conv_op(pool1, 64, 128, 'conv_3')
            conv4 = self.conv_op(conv3, 128, 128, 'conv_4')
            pool2 = self.pool_op(conv4)
            conv5 = self.conv_op(pool2, 128, 256, 'conv_5')
            conv6 = self.conv_op(conv5, 256, 256, 'conv_6')
            conv7 = self.conv_op(conv6, 256, 256, 'conv_7')
            pool3 = self.pool_op(conv7)
            conv8 = self.conv_op(pool3, 256, 512, 'conv_8')
            conv9 = self.conv_op(conv8, 512, 512, 'conv_9')
            conv10 = self.conv_op(conv9, 512, 512, 'conv_10')
            pool4 = self.pool_op(conv10)
            conv11 = self.conv_op(pool4, 512, 512, 'conv_11')
            conv12 = self.conv_op(conv11, 512, 512, 'conv_12')
            conv13 = self.conv_op(conv12, 512, 512, 'conv_13')
        return conv13

    def _build_tail_net(self, input_conv, scope):
        with tf.name_scope(scope):
            pool5 = self.pool_op(input_conv)
            fc_in = tf.contrib.layers.flatten(pool5)
            fc1 = self.fc_op_head(fc_in, 4096, "fc_1")
            fc2 = self.fc_op(fc1, 4096, "fc_2")
            fc3 = self.sigmoid_fc_op(fc2, self.num_classes, "fc_3")
            prediction_classfication = tf.nn.softmax(fc3)
        return prediction_classfication

    def conv_op(self, conv_input, channel, depth, scope):
        kernel = tf.get_variable("kernel"+scope, [self.conv_size, self.conv_size, channel, depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("conv_bias"+scope, [depth], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv_input, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv_output = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return conv_output

    def pool_op(self, pool_input):
        pool = tf.nn.max_pool(pool_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool
    
    def fc_op(self, input, nodes, scope):
        input_shape = input.get_shape()
        #print(input_shape)
        weights = tf.get_variable("weights"+scope, [input_shape[1], nodes], 
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        #weights = tf.get_variable("weights"+scope, [32, nodes], 
        #                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias"+scope, [nodes], initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input, weights), bias)
        relu = tf.nn.relu(fc)
        return relu      

    def fc_op_head(self, input, nodes, scope):
        input_shape = tf.shape(input)
        #print(input_shape)
        #weights = tf.get_variable("weights"+scope, [input_shape[1], nodes], 
        #                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        weights = tf.get_variable("weights"+scope, [32, nodes], 
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias"+scope, [nodes], initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input, weights), bias)
        relu = tf.nn.relu(fc)
        return relu   

    def sigmoid_fc_op(self, input, nodes, scope):
        input_shape = input.get_shape()
        weights = tf.get_variable("weights"+scope, [input_shape[1], nodes], 
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias"+scope, [nodes], initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input, weights), bias)
        sigmoid = tf.nn.sigmoid(fc)
        return sigmoid
    
    def crossentropy_loss_compute(self, prediction_classification, label):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_classification, labels=tf.argmax(label, 1)))
        return loss
    
    def train_vgg16(self, globle_step, learning_rate, learning_rate_decay, decay_step, batch_size):
        self._build_network(is_training=True)
        steps = tf.Variable(0, trainable=False)
        train_learning_rate = tf.train.exponential_decay(learning_rate, steps, decay_step, learning_rate_decay)
        train_step  = tf.train.GradientDescentOptimizer(train_learning_rate).minimize(loss, global_step=steps)
        


#    def regonization(self)

        

