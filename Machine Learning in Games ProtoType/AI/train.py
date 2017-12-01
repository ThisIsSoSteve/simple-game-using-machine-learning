#Train.py
import tensorflow as tf
import numpy as np
from AI import model

def train_simple_model(training_data_X, training_data_Y, restore_checkpoint_path):

    number_of_epochs = 50
    online_training = True
    checkpoint_file_path =  'E:/Machine Learning in Games/Checkpoints/turn_based_ai.ckpt'
    learning_rate = 0.01 #default = 0.001
    current_accuracy = 0.0

    tf.reset_default_graph()
    with tf.name_scope("train"):

        X = tf.placeholder(tf.float32, [None, 10])
        Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

        prediction = model.simple_model(X)

        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Y))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

       

        saver = tf.train.Saver()#(max_to_keep=10)
        #Start Training
        with tf.Session() as sess:
        
            sess.run(tf.global_variables_initializer())

            if restore_checkpoint_path != '':
                saver.restore(sess, restore_checkpoint_path)

            for step in range(number_of_epochs):

                if online_training:
                    #print(np.size(training_data_X, 0))
                    for i in range(np.size(training_data_X, 0)):
                        _, loss = sess.run([optimizer, cost], feed_dict = { X: np.reshape(training_data_X[i], (-1, 10)), Y: np.reshape(training_data_Y[i],(-1, 4))})
                else:
                    _, loss = sess.run([optimizer, cost], feed_dict = { X: training_data_X, Y: training_data_Y })

                #Consider changing so we are not using the training data
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                current_accuracy = accuracy.eval(feed_dict={ X: training_data_X, Y: training_data_Y})

                print('Epoch {} - Loss {} - Accuracy {}'.format(step, loss, current_accuracy))
                if current_accuracy == 1.0:
                    break
        
            print('Saving...')
            #saver.save(sess, '{}/turn_based_ai-{}.ckpt'.format(checkpoint_file_path, current_accuracy))
            saver.save(sess, checkpoint_file_path)

    print('Completed')
    return checkpoint_file_path, current_accuracy