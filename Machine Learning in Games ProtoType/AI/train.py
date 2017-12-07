#Train.py
import tensorflow as tf
import numpy as np
from AI import model



def train_simple_model(training_data_X, training_data_Y, restore_checkpoint_path, starting_step):

    number_of_epochs = 50
    online_training = True
    checkpoint_file_path =  'E:/Machine Learning in Games/Checkpoints/turn_based_ai.ckpt'
    learning_rate = 0.01 #default = 0.001
    current_accuracy = 0.0
    global_step = starting_step

    tf.reset_default_graph()
    #with tf.name_scope("train"):

    X = tf.placeholder(tf.float32, [None, 10])
    Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

    prediction = model.simple_model(X)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('layer_1', reuse=True):
        with tf.name_scope('weights'):
            weights = tf.get_variable('fully_connected/weights')#, [10,6]
        with tf.name_scope('biases'):
            biases = tf.get_variable('fully_connected/biases')

        #with tf.name_scope('activations'):
        #    activations = tf.get_variable('activations', [6])

    # create a summary for our cost and accuracy
    tf.summary.scalar("Cross_entropy", cost)
    tf.summary.scalar("Accuracy", accuracy)
    tf.summary.histogram("layer_1_weights", weights)
    tf.summary.histogram("layer_1_biases", biases)
    #tf.summary.histogram("layer_1_activations", activations)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()#(max_to_keep=10)
    #Start Training
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        if restore_checkpoint_path != '':
            saver.restore(sess, restore_checkpoint_path)

        #create log writer object
        writer = tf.summary.FileWriter('E:/Logs/{}'.format(starting_step), graph=tf.get_default_graph())

        for step in range(number_of_epochs):

            if online_training:
                #print(np.size(training_data_X, 0))
                for i in range(np.size(training_data_X, 0)):
                    _, loss = sess.run([optimizer, cost], feed_dict = { X: np.reshape(training_data_X[i], (-1, 10)), Y: np.reshape(training_data_Y[i],(-1, 4))})
                    #final_step += 1
                    # write log
                    #writer.add_summary(summary, final_step)
            else:
                _, loss = sess.run([optimizer, cost], feed_dict = { X: training_data_X, Y: training_data_Y })

            #current_accuracy = accuracy.eval(feed_dict={ X: training_data_X, Y: training_data_Y})
            summary, current_accuracy, loss = sess.run([merged, accuracy, cost], feed_dict={ X: training_data_X, Y: training_data_Y})
            global_step += 1
            writer.add_summary(summary, global_step)

            #predict/layer_1/fully_connected/Sigmoid
            #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #    print(i.name)   # i.name if you want just a name
            #    print(i)

            #vars = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1'))  
            #print(vars)

            print('Epoch {} - Loss {} - Accuracy {}'.format(global_step, loss, current_accuracy))
            if current_accuracy == 1.0:
                break
        
        print('Saving...')
        #saver.save(sess, '{}/turn_based_ai-{}.ckpt'.format(checkpoint_file_path, current_accuracy))
        saver.save(sess, checkpoint_file_path)

    print('Completed')
    return checkpoint_file_path, global_step


#Notes
#weights=sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'layer_1/fully_connected/weights'))
#print(weight)
#bias=sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'layer_1/fully_connected/bias'))
#print(bias)

#Get all scope variables
#for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1'):
#                print(i.name)   # i.name if you want just a name
#                print(i)

#ValueError: initial_value must have a shape specified: Tensor("predict/layer_1/fully_connected/Sigmoid:0", shape=(?, 6), dtype=float32)