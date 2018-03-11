import tensorflow as tf
from lazy_property import lazy_property

'''
using structure from https://danijar.com/structuring-your-tensorflow-models/
'''
class Model:

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        feature_size = 7
        label_size = 4
        layer_1_size = 30

        with tf.variable_scope('layer_1') as scope:
            layer_1_weights = tf.Variable(tf.random_uniform(shape=[feature_size, layer_1_size], minval = 0.0001, maxval = 0.001), name="weights")
            #layer_1_weights = tf.Variable(tf.constant(0.0, shape=[feature_size, layer_1_size]), name="weights")
            layer_1_biases = tf.Variable(tf.constant(0.0, shape = [layer_1_size]), name = "biases")
            layer_1 = tf.nn.relu(tf.matmul(self.feature, layer_1_weights) + layer_1_biases)

        with tf.variable_scope('layer_output') as scope:
            output_weights = tf.Variable(tf.random_uniform(shape = [layer_1_size, label_size], minval = 0.0001, maxval = 0.001), name = "weights")
            output_biases = tf.Variable(tf.constant(0.0, shape = [label_size]), name = "biases")
            output = tf.matmul(layer_1, output_weights) + output_biases

        return output

    @lazy_property
    def optimize(self):
        learning_rate = 0.01
        squared_error = tf.square(self.prediction - self.label)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        cost = tf.reduce_mean(sum_squared_error)
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label))
        #return tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost), cost
        return tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost), cost

    @lazy_property
    def error(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
