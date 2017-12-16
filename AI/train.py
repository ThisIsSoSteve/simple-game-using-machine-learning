#Train.py
import tensorflow as tf
import numpy as np
from AI import model
from plot import Plot
import matplotlib.pyplot as plt

class Train:
    def __init__(self, max_epochs, learning_rate):
        self.number_of_epochs = max_epochs
        self.learning_rate = learning_rate
        self.checkpoint_file_path =  'E:/Machine Learning in Games/Checkpoints/turn_based_ai.ckpt'
        self.global_step = 0
        #Setup Data Plots
        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')

    def train_simple_model(self, training_data_X, training_data_Y, restore):

        online_training = True
        current_accuracy = 0.0

        tf.reset_default_graph()
        #with tf.name_scope("train"):

        X = tf.placeholder(tf.float32, [None, 10])
        Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

        prediction = model.simple_model(X)

        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        with tf.name_scope('cross_entropy'):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Y))

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        #tf.summary.scalar("Cross_entropy", cost)
        #tf.summary.scalar("Accuracy", accuracy)


        #merged = tf.summary.merge_all()

        saver = tf.train.Saver()#(max_to_keep=10)
        #Start Training
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint_file_path)

            #create log writer object
            #writer = tf.summary.FileWriter('E:/Logs/{}'.format(starting_step), graph=tf.get_default_graph())

            for _ in range(self.number_of_epochs):

                if online_training:
                    #print(np.size(training_data_X, 0))
                    for i in range(np.size(training_data_X, 0)):
                        _, loss = sess.run([optimizer, cost], feed_dict = { X: np.reshape(training_data_X[i], (-1, 10)), Y: np.reshape(training_data_Y[i],(-1, 4))})
                else:
                    _, loss = sess.run([optimizer, cost], feed_dict = { X: training_data_X, Y: training_data_Y })

                #current_accuracy = accuracy.eval(feed_dict={ X: training_data_X, Y: training_data_Y})
                current_accuracy, loss = sess.run([accuracy, cost], feed_dict={ X: training_data_X, Y: training_data_Y})
                self.global_step += 1

                self.cost_plot.data.append(loss)
                self.accuracy_plot.data.append(current_accuracy)

                # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                #     print(i.name)   # i.name if you want just a name
                #     print(i)

               

                print('Epoch {} - Loss {} - Accuracy {}'.format(self.global_step, loss, current_accuracy))
                if current_accuracy == 1.0:
                    break
            
            print('Saving...')
            #saver.save(sess, '{}/turn_based_ai-{}.ckpt'.format(checkpoint_file_path, current_accuracy))
            saver.save(sess, self.checkpoint_file_path)

            weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
            #print(weights)
            
            plt.close('all')
            plt.figure()
            plt.imshow(weights, cmap='Greys_r', interpolation='none')
            plt.xlabel('Nodes')
            plt.ylabel('Inputs')
            plt.show()
            plt.close()

        #self.cost_plot.show_sub_plot(self.accuracy_plot)
        self.cost_plot.save_sub_plot(self.accuracy_plot,
         "E:/Charts/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))

       
        
        print('Completed')
        return self.checkpoint_file_path, self.global_step


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