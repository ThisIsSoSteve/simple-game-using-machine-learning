import tensorflow as tf
from lazy_property import lazy_property

class Model:

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        feature_size = 10
        label_size = 4
        layer_1_size = 2

        with tf.variable_scope('layer_1') as scope:
            layer_1_weights = tf.Variable(tf.random_uniform(shape=[feature_size, layer_1_size], minval=0.001, maxval=0.01), name="weights")
            layer_1_biases = tf.Variable(tf.constant(0.01, shape = [layer_1_size]), name = "biases")
            layer_1 = tf.nn.sigmoid(tf.matmul(self.feature, layer_1_weights) + layer_1_biases)

        with tf.variable_scope('layer_output') as scope:
            output_weights = tf.Variable(tf.random_uniform(shape = [layer_1_size, label_size], minval = 0.001, maxval = 0.01), name = "weights")
            output_biases = tf.Variable(tf.constant(0.0, shape = [label_size]), name = "biases")
            output = tf.matmul(layer_1, output_weights) + output_biases

        return output

    @lazy_property
    def optimize(self):
        learning_rate = 0.1
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.label))
        #return tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost), cost
        return tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost), cost

    @lazy_property
    def error(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
