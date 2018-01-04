'''
Main.py Starting File
'''
import os
import numpy as np
import tensorflow as tf
from model import Model
from plot import Plot
from game import Game
#import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.feature_length = 6
        self.label_length = 4

        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')
        self.checkpoint = 'data/Checkpoints/turn_based_ai.ckpt'
        
        self.X = tf.placeholder(tf.float32, [None, self.feature_length])
        self.Y = tf.placeholder(tf.float32, [None, self.label_length])
        self.model = Model(self.X, self.Y)
        self.global_step = 0

        self.training_data_x = np.empty((0, self.feature_length))
        self.training_data_y = np.empty((0, self.label_length))

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def add_training_data(self, features, labels):
        self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
        self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)
        # print('Features')
        # print(features)
        # print('Labels')
        # print(labels)

    def get_data_for_prediction(self, user, opponent):
        data = np.array([1, 0.4, 1, 1, 0.4, 1])#Default starting data (not great)

        if user != None:
            data = np.array([user.attack / user.max_attack,
                            user.defence / user.max_defence,
                            user.health / user.max_health,
                            opponent.attack / opponent.max_attack,
                            opponent.defence / opponent.max_defence,
                            opponent.health / opponent.max_health
                            ])
        return np.reshape(data, (-1, self.feature_length))

    def start(self, restore):
        train = True
        players_turn = True
        player_goes_first = True
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint)

            while(train):
                game = Game(player_goes_first, self.feature_length, self.label_length)
                if player_goes_first:
                    players_turn = True
                else:
                    players_turn = False

                player_goes_first = not player_goes_first
                game_over = False
                user = None
                opponent = None

                while(not game_over):
                    predicted_action = 0

                    if players_turn is False:
                        #Predict opponent's action
                        data = self.get_data_for_prediction(user, opponent)
                        #print('opponents\'s view: {}'.format(data))
                        predicted_actions = sess.run(self.model.prediction, { self.X: data })[0]
                        predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))
                        predicted_action = np.argmax(predicted_actions) + 1

                    #Play Game
                    game_over, players_turn, user, opponent, training_data = game.run(predicted_action)

                    if game_over and training_data == None:
                        train = False
                    elif game_over:
                        #record winning data
                        self.add_training_data(training_data.feature, training_data.label)

                #Train
                if train:
                    for _ in range(50):
                        for i in range(np.size(self.training_data_x, 0)):
                            _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_x[i], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[i],(-1, 4))})
                        #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        self.global_step += 1

                        current_accuracy = sess.run(self.model.error, { self.X: self.training_data_x, self.Y: self.training_data_y })
                        self.cost_plot.data.append(loss)
                        self.accuracy_plot.data.append(current_accuracy)

                    print('Saving...')
                    saver.save(sess, self.checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {}'.format(self.global_step, loss, current_accuracy))

                    #weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
                    
                    #Move out into class
                    # plt.close('all')
                    # plt.figure()
                    # plt.imshow(weights, cmap='Greys_r', interpolation='none')
                    # plt.xlabel('Nodes')
                    # plt.ylabel('Inputs')
                    # plt.show()
                    # plt.close()

                    self.cost_plot.save_sub_plot(self.accuracy_plot,
                    "data/Charts/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main().start(False)
