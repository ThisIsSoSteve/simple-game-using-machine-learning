#Model.py
import tensorflow as tf

# with tf.name_scope('input'):
#     X = tf.placeholder(tf.float32, [None, 10])
#     Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

def simple_model(x):

    with tf.variable_scope('layer_1') as scope:
        layer1_weights = tf.Variable(tf.random_uniform(shape = [10, 2], minval = 0.001, maxval = 0.01), name = "weights")
        layer1_biases = tf.Variable(tf.constant(0.01, shape = [2]), name = "biases")
        layer1 = tf.nn.sigmoid(tf.matmul(x, layer1_weights) + layer1_biases)

    with tf.variable_scope('layer_output') as scope:
        output_weights = tf.Variable(tf.random_uniform(shape = [2, 4], minval = 0.001, maxval = 0.01), name = "weights")
        output_biases = tf.Variable(tf.constant(0.0, shape = [4]), name = "biases")
        output = tf.matmul(layer1, output_weights) + output_biases

    return output