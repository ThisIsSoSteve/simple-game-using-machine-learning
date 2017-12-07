#Use.py

import tensorflow as tf
from AI import model
import numpy as np

def use_simple_model(restore_checkpoint_path, data):

     tf.reset_default_graph()
     with tf.name_scope("predict"):
        X = tf.placeholder(tf.float32, [None, 10])
        Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

        prediction = model.simple_model(X)

        
        if restore_checkpoint_path != '':
            saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore_checkpoint_path != '':
                saver.restore(sess, restore_checkpoint_path)

            predicted_actions = sess.run(prediction, feed_dict={X: data})[0]

            predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))

            print(predicted_actions)

            #mean_vector = np.mean(np.abs(predicted_actions))

            #threshold = (1-mean_vector)*np.random.random() + mean_vector

            #max_value = np.max(predicted_actions)


            #print('thershold: {}'.format(threshold))
            #if max_value > threshold:
            #    print('argmax')
            #    return np.argmax(predicted_actions)
            #else:
            #    print('random')
            #    return np.random.randint(4)

            return np.argmax(predicted_actions)
