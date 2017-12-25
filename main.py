'''
Main.py Starting File
'''
import os
import shutil
import numpy as np
import tensorflow as tf
from player import Player
from actions import Actions
from data import Data
from model import Model
from plot import Plot
import matplotlib.pyplot as plt
 
class Main:
    def __init__(self):
        self.play_game = True
        self.player_turn = True
        self.player_goes_first = True
        self.number_of_turns = 0

        self.feature_length = 10
        self.label_length = 4

        self.user = Player('user', self.label_length)
        self.opponent = Player('opponent', self.label_length)
        self.game_actions = Actions()
        #training data
        self.player_training_data = Data(self.feature_length, self.label_length)
        self.opponent_training_data = Data(self.feature_length, self.label_length)

        #checkpoint to use
        self.checkpoint = False

        self.starting_stats = np.array([0, 0, 0, 0,
                                           self.user.attack / self.user.max_attack,
                                           self.user.defence / self.user.max_defence,
                                           self.user.health / self.user.max_health,
                                           self.opponent.attack / self.opponent.max_attack,
                                           self.opponent.defence / self.opponent.max_defence,
                                           self.opponent.health / self.opponent.max_health])

        self.step = 0

        self.training_data_x = np.empty((0, self.feature_length))
        self.training_data_y = np.empty((0, self.label_length))

        self.logs_directory = "C:/python/Logs"
        shutil.rmtree(self.logs_directory)
        os.makedirs(self.logs_directory)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        #self.trainer = train.Train(50, 0.01)

    def set_defaults(self):
        self.player_turn = self.player_goes_first

        self.number_of_turns = 0

        self.user = Player('user', self.label_length)
        self.opponent = Player('opponent', self.label_length)

        #reset training data
        self.player_training_data = Data(self.feature_length, self.label_length)
        self.opponent_training_data = Data(self.feature_length, self.label_length)

        os.system('cls')

    #ToDo: Move to common
    def intTryParse(self, value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    def run_game(self, opponents_action):        
        if self.player_turn:
            self.user.print_health()
            self.game_actions.display_player_actions(self.user)
            print('5. Exit')

            user_input = input('Action (1-5)')
            players_action, is_valid = self.intTryParse(user_input)
            os.system('cls')

            if is_valid and players_action > 0 and players_action <= 5:
                if players_action == 5:
                    self.play_game = False
                else:
                    self.player_training_data.record(players_action, self.user, self.opponent, True)
                    self.game_actions.perfrom(self.user, self.opponent, players_action)
                    self.game_actions.display_player_chosen_action(self.user, players_action)
                    
                self.player_turn = False

            else:
                print('Please enter a valid option from 1-5')
        else: #AI's turn
            #print('opponent\'s choice number: {}'.format(opponents_action))

            self.opponent_training_data.record(opponents_action, self.user, self.opponent, False)

            #print('')
            self.opponent.print_health()
            self.game_actions.display_ai_chosen_action(self.opponent, opponents_action)
            self.game_actions.perfrom(self.opponent, self.user, opponents_action)

            self.player_turn = True

        #if player_turn: #if player chooses an invalid action don't +1 to number_of_turns
        self.number_of_turns += 1

        if self.user.alive is False or self.opponent.alive is False:
            os.system('cls')
            self.add_training_data(self.user.alive)

            if self.user.alive is False:
                print('You lost')
            else:
                print('You Won')

            self.player_goes_first = not self.player_goes_first
            input('Press any key to continue..')
            self.set_defaults()
            return True
        
        return False

    def add_training_data(self, did_player_win):
        if did_player_win:
            self.training_data_x = np.concatenate((self.training_data_x, self.player_training_data.feature), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.player_training_data.label), axis=0)
        else:
            self.training_data_x = np.concatenate((self.training_data_x, self.opponent_training_data.feature), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.opponent_training_data.label), axis=0)   

    def get_data_for_prediction(self):
        data = self.starting_stats
        if len(self.player_training_data.feature) != 0:#if not self.player_training_data.data
            data = np.array([self.user.last_action[0, 0],
                                self.user.last_action[0, 1],
                                self.user.last_action[0, 2],
                                self.user.last_action[0, 3],
                                self.user.attack / self.user.max_attack,
                                self.user.defence / self.user.max_defence,
                                self.user.health / self.user.max_health,
                                self.opponent.attack / self.opponent.max_attack,
                                self.opponent.defence / self.opponent.max_defence,
                                self.opponent.health / self.opponent.max_health
                ])

        return np.reshape(data, (-1, 10))

    def start(self, restore):
        global_step = 0
        cost_plot = Plot([], 'Step', 'Cost')
        accuracy_plot = Plot([], 'Step', 'Accuracy')
        checkpoint = 'C:/python/Checkpoints/turn_based_ai.ckpt'
        #Todo: Move to init
        X = tf.placeholder(tf.float32, [None, 10])
        Y = tf.placeholder(tf.float32, [None, 4])
        model = Model(X, Y)
        saver = tf.train.Saver()
        global_step = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, checkpoint)

            while(self.play_game):
                predicted_action = np.zeros((1, 4))
                #Use 
                if self.player_turn == False:
                    data = self.get_data_for_prediction()
                    #print('opponents\'s view: {}'.format(data))
                    predicted_actions = sess.run(model.prediction, { X: data })[0]
                    predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))
                    predicted_action = np.argmax(predicted_actions) + 1

                #Play Game
                train_network = self.run_game(predicted_action)
                #Train
                if train_network:
                    for _ in range(50):
                        for i in range(np.size(self.training_data_x, 0)):
                            _, loss = sess.run(model.optimize, { X: np.reshape(self.training_data_x[i], (-1, 10)), Y: np.reshape(self.training_data_y[i],(-1, 4))})
                        #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        global_step += 1

                        current_accuracy = sess.run(model.error, { X: self.training_data_x, Y: self.training_data_y })
                        cost_plot.data.append(loss)
                        accuracy_plot.data.append(current_accuracy)

                        #if current_accuracy == 1.0:    
                            #break

                    print('Saving...')
                    saver.save(sess, checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {}'.format(global_step, loss, current_accuracy))

                    weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
                    
                    #Move out into class
                    # plt.close('all')
                    # plt.figure()
                    # plt.imshow(weights, cmap='Greys_r', interpolation='none')
                    # plt.xlabel('Nodes')
                    # plt.ylabel('Inputs')
                    # plt.show()
                    # plt.close()

                    cost_plot.save_sub_plot(accuracy_plot,
                    "C:/python/Charts/{} and {}.png".format(cost_plot.y_label, accuracy_plot.y_label))
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main().start(False)

'''
Training cycle
User's go:
    get opponent's previous move
    get current state of opponent's and user's stats
    get user's choice
    store above data
Opponents's go:
    get user's previous move
    get current state of user's and opponent's stats
    get opponent's choice
    store above data

Train neural networks
    get moves from the winner

'''