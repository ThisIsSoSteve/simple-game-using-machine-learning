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
        self.checkpoint = 'data/checkpoints/turn_based_ai.ckpt'
        
        self.X = tf.placeholder(tf.float32, [None, self.feature_length])
        self.Y = tf.placeholder(tf.float32, [None, self.label_length])
        self.model = Model(self.X, self.Y)
        self.global_step = 0

        self.training_data_x = np.empty((0, self.feature_length))
        self.training_data_y = np.empty((0, self.label_length))

        self.test_training_data_x = np.empty((0, self.feature_length))
        self.test_training_data_y = np.empty((0, self.label_length))

        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # def add_training_data(self, features, labels):
    #     self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
    #     self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)
    def add_training_data(self, features, labels, add_to_test_data):
        self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
        self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)

        if add_to_test_data:
            self.test_training_data_x = np.concatenate((self.test_training_data_x, features), axis=0)
            self.test_training_data_y = np.concatenate((self.test_training_data_y, labels), axis=0)



    def get_data_for_prediction(self, user, opponent):
        #data = np.array([1, 0.4, 1, 1, 0.4, 1])#Default starting data (not great)
        #if user != None:
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
        #players_turn = True
        player_goes_first = True
        saver = tf.train.Saver()

        with tf.Session() as sess:

            if restore:
                saver.restore(sess, self.checkpoint)
            else:
                sess.run(tf.global_variables_initializer())

            while train:
                game = Game(player_goes_first, self.feature_length, self.label_length)
                # if player_goes_first:
                #     players_turn = True
                # else:
                #     players_turn = False

                player_goes_first = not player_goes_first
                # game_over = False
                # user = None
                # opponent = None

                while not game.game_over:
                    predicted_action = 0

                    if game.players_turn is False:
                        #Predict opponent's action
                        data = self.get_data_for_prediction(game.user, game.opponent)
                        #print('opponents\'s view: {}'.format(data))
                        predicted_actions = sess.run(self.model.prediction, { self.X: data })[0]
                        #predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))
                        predicted_action = np.argmax(predicted_actions) + 1

                    #Play Game
                    did_player_win = game.run(predicted_action)
                    #game_over, players_turn, user, opponent, training_data = game.run(predicted_action)

                    # if game.game_over and training_data == None:
                    #     train = False
                    # elif game_over:
                    #     #record winning data
                    #     self.add_training_data(training_data.feature, training_data.label)
                    if game.game_over and did_player_win == None:
                        train = False
                    elif game.game_over:
                        #record winning data
                        if did_player_win:
                            self.add_training_data(game.player_training_data.feature, game.player_training_data.label, False)
                        else:
                            self.add_training_data(game.opponent_training_data.feature, game.opponent_training_data.label, False)


                #Train
                if train:
                    for _ in range(50):
                        
                        training_data_size = np.size(self.training_data_x, 0)
                        random_range = np.arange(training_data_size)
                        np.random.shuffle(random_range)

                        for i in range(training_data_size):
                            random_index = random_range[i]
                            _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_x[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[random_index],(-1, 4))})
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
                    "data/charts/{} and {}.png".format(self.cost_plot.y_label, self.accuracy_plot.y_label))
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main().start(False)
