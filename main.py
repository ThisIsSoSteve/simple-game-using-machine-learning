'''
Main.py Starting File
'''
import os
import numpy as np
import tensorflow as tf
from model import Model
from plot import Plot
from game import Game

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import matplotlib.pyplot as plt

#todo separate ai_vs_ai and user_vs_ai
class Main:
    def __init__(self):
        self.feature_length = 6
        self.label_length = 4

        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')
        self.checkpoint = 'data/Checkpoints/turn_based_ai.ckpt'

        self.X = tf.placeholder(tf.float32, [None, self.feature_length])
        self.Y = tf.placeholder(tf.float32, [None, self.label_length])
        self.global_step = 0

        self.training_data_x = np.empty((0, self.feature_length))
        self.training_data_y = np.empty((0, self.label_length))

        #user_vs_ai_only
        self.test_training_data_x = np.empty((0, self.feature_length))
        self.test_training_data_y = np.empty((0, self.label_length))

        #ai_vs_ai_only
        self.negative_training_data_x = np.empty((0, self.feature_length))
        self.negative_training_data_y = np.empty((0, self.label_length))

        self.ai_1_test_training_data_x = np.empty((0, self.feature_length))
        self.ai_1_test_training_data_y = np.empty((0, self.label_length))

        self.ai_2_test_training_data_x = np.empty((0, self.feature_length))
        self.ai_2_test_training_data_y = np.empty((0, self.label_length))

    #user_vs_ai_only
    def add_training_data(self, features, labels, add_to_test_data):
        self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
        self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)

        if add_to_test_data:
            self.test_training_data_x = np.concatenate((self.test_training_data_x, features), axis=0)
            self.test_training_data_y = np.concatenate((self.test_training_data_y, labels), axis=0)

    #ai_vs_ai_only
    def add_ai_training_data(self, features, labels, add_to_test_data, did_player_1_win, add_negative_data):
        if add_negative_data:
            self.negative_training_data_x = np.concatenate((self.negative_training_data_x, features), axis=0)
            self.negative_training_data_y = np.concatenate((self.negative_training_data_y, labels), axis=0)
        else:
            self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)

        if add_to_test_data:
            if did_player_1_win:
                self.ai_1_test_training_data_x = np.concatenate((self.ai_1_test_training_data_x, features), axis=0)
                self.ai_1_test_training_data_y = np.concatenate((self.ai_1_test_training_data_y, labels), axis=0)
            else:
                self.ai_2_test_training_data_x = np.concatenate((self.ai_2_test_training_data_x, features), axis=0)
                self.ai_2_test_training_data_y = np.concatenate((self.ai_2_test_training_data_y, labels), axis=0)

    def get_data_for_prediction(self, user, opponent):
        data = np.array([user.attack / user.max_attack,
                        user.defence / user.max_defence,
                        user.health / user.max_health,
                        opponent.attack / opponent.max_attack,
                        opponent.defence / opponent.max_defence,
                        opponent.health / opponent.max_health
                        ])
        return np.reshape(data, (-1, self.feature_length))

    def start_user_vs_ai(self, restore):
        model = Model('ai', self.X, self.Y)
        train = True
        players_turn = True
        player_goes_first = True
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint)

            while train:
                game = Game(player_goes_first, self.feature_length, self.label_length)

                player_goes_first = not player_goes_first

                while not game.game_over:
                    predicted_action = 0

                    if game.players_turn is False:
                        #Predict opponent's action
                        data = self.get_data_for_prediction(game.user, game.opponent)
                        #print('opponents\'s view: {}'.format(data))
                        predicted_actions = sess.run(model.prediction, { self.X: data })[0]
                        #predicted_actions = sess.run(tf.nn.sigmoid(predicted_actions))
                        predicted_action = np.argmax(predicted_actions) + 1

                    #Play Game
                    did_player_win = game.run(predicted_action)

                    if game.game_over and did_player_win == None:
                        train = False
                    elif game.game_over:
                        #record winning data
                        if did_player_win:
                            self.add_training_data(game.player_training_data.feature, game.player_training_data.label, False)
                        else:
                            self.add_training_data(game.opponent_training_data.feature, game.opponent_training_data.label, False)

                #Train
                if train:# ToDo: put back to train this is to test  )
                    for _ in range(50):
                        
                        training_data_size = np.size(self.training_data_x, 0)
                        random_range = np.arange(training_data_size)
                        np.random.shuffle(random_range)

                        for i in range(training_data_size):
                            random_index = random_range[i]
                            _, loss = sess.run(model.optimize, { self.X: np.reshape(self.training_data_x[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[random_index],(-1, 4))})
                        #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        self.global_step += 1

                        current_accuracy = sess.run(model.error, { self.X: self.training_data_x, self.Y: self.training_data_y })
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def start_ai_vs_ai(self, restore, number_of_games):
        ai_1_model = Model('ai_1', self.X, self.Y)
        ai_2_model = Model('ai_2', self.X, self.Y)

        ai_1_wins = 0
        ai_2_wins = 0

        ai_1_cost_plot = Plot([], 'Step', 'ai_1_Cost')
        ai_1_accuracy_plot = Plot([], 'Step', 'ai_1_Accuracy')

        ai_2_cost_plot = Plot([], 'Step', 'ai_2_Cost')
        ai_2_accuracy_plot = Plot([], 'Step', 'ai_2_Accuracy')

        train = False
        number_of_games_played = 0
        max_turns = 40
        current_turns = 0
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if restore:
                saver.restore(sess, self.checkpoint)

            while number_of_games_played < number_of_games:
                game = Game(True, self.feature_length, self.label_length)
                current_turns = 0
                #game_over = False
                # user = None
                # opponent = None
                train = False

                while not game.game_over:
                    predicted_action = 0
                    #Predict action
                    if game.players_turn is False:
                        data = self.get_data_for_prediction(game.user, game.opponent)
                        predicted_actions = sess.run(ai_2_model.prediction, { self.X: data })[0]
                    else:
                        data = self.get_data_for_prediction(game.opponent, game.user)
                        predicted_actions = sess.run(ai_1_model.prediction, { self.X: data })[0]
                        
                    #print('features view: {}'.format(data))
                    
                    predicted_actions = self.sigmoid(predicted_actions)

                    #get prodability (much faster than tf.nn.softmax)
                    max_prediction = np.sum(predicted_actions)
                    probability_predicted = predicted_actions / max_prediction
                    probabilities = probability_predicted
                    choices = np.arange(1, self.label_length + 1)

                    choice = np.random.choice(choices, p=probabilities)   
                    #Some reason tf.nn.softmax progressively slows the program to a crawl in about 20 games.
                    #not sure why seems to only happen in windows
                    #probabilities = sess.run(tf.nn.softmax(predicted_actions[0]))                                    

                    #Play Game
                    did_player_1_win = game.run_ai_game(choice)

                    if game.game_over:
                        #record winning data
                        if did_player_1_win:
                            self.add_ai_training_data(game.player_training_data.feature, game.player_training_data.label, True, True, False)
                            ai_1_wins += 1
                            self.add_ai_training_data(game.opponent_training_data.feature, 1 - game.opponent_training_data.label, False, True, True)
                        else:
                            self.add_ai_training_data(game.opponent_training_data.feature, game.opponent_training_data.label, True, False, False)
                            ai_2_wins += 1
                            self.add_ai_training_data(game.player_training_data.feature, 1 - game.player_training_data.label, False, False, True)
                            #self.add_training_data(game.player_training_data.feature, 1 - game.player_training_data.label, False)

                        number_of_games_played += 1
                        train = True

                    current_turns += 1
                    if current_turns >= max_turns:
                        game.game_over = True
                        train = True
                        print("hit max turns")
                        

                    
                #Train
                if train and np.size(self.training_data_x, 0) > 0:
                    for _ in range(1):
                        
                        training_data_size = np.size(self.training_data_x, 0)
                        # random_range = np.arange(training_data_size)
                        # np.random.shuffle(random_range)
                        ai_1_loss = 0
                        ai_2_loss = 0

                        #Train 'positive' data for loser
                        for i in range(training_data_size):
                            #random_index = random_range[i]
                            if did_player_1_win:
                                _, ai_1_loss = sess.run(ai_1_model.optimize, { self.X: np.reshape(self.training_data_x[i], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[i],(-1, 4))})
                            else:
                                _, ai_2_loss = sess.run(ai_2_model.optimize, { self.X: np.reshape(self.training_data_x[i], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_y[i],(-1, 4))})
                        
                        training_data_size = np.size(self.negative_training_data_x, 0)
                        #Train 'negative' data for loser
                        for i in range(training_data_size):

                            if did_player_1_win is False:
                                _, ai_1_loss = sess.run(ai_1_model.optimize, { self.X: np.reshape(self.negative_training_data_x[i], (-1, self.feature_length)), self.Y: np.reshape(self.negative_training_data_y[i],(-1, 4))})
                            else:
                                _, ai_2_loss = sess.run(ai_2_model.optimize, { self.X: np.reshape(self.negative_training_data_x[i], (-1, self.feature_length)), self.Y: np.reshape(self.negative_training_data_y[i],(-1, 4))})

                        self.global_step += 1

                    
                    ai_1_current_accuracy = sess.run(ai_1_model.error, { self.X: self.ai_1_test_training_data_x, self.Y: self.ai_1_test_training_data_y })
                    ai_1_cost_plot.data.append(ai_1_loss)
                    ai_1_accuracy_plot.data.append(ai_1_current_accuracy)
                    ai_1_cost_plot.save_sub_plot(ai_1_accuracy_plot,
                        "data/Charts/ai_1 {} and {}.png".format(ai_1_cost_plot.y_label, ai_1_accuracy_plot.y_label))
                
                    ai_2_current_accuracy = sess.run(ai_2_model.error, { self.X: self.ai_2_test_training_data_x, self.Y: self.ai_2_test_training_data_y })
                    ai_2_cost_plot.data.append(ai_2_loss)
                    ai_2_accuracy_plot.data.append(ai_2_current_accuracy)
                    ai_2_cost_plot.save_sub_plot(ai_2_accuracy_plot,
                        "data/Charts/ai_2 {} and {}.png".format(ai_2_cost_plot.y_label, ai_2_accuracy_plot.y_label))
                    
                    #clear training data
                    self.training_data_x = np.empty((0, self.feature_length))
                    self.training_data_y = np.empty((0, self.label_length))

                    self.negative_training_data_x = np.empty((0, self.feature_length))
                    self.negative_ttraining_data_y = np.empty((0, self.label_length))
                    #self.test_training_data_x = np.empty((0, self.feature_length))
                    #self.test_training_data_y = np.empty((0, self.label_length))


                    #user_input = input('paused')
                    #print('Saving...')
                    #saver.save(sess, self.checkpoint)

                    print('Epoch {}'.format(self.global_step))

                    print('AI_1: Loss {} - Accuracy: {} - Wins {}'
                    .format(ai_1_loss, ai_1_current_accuracy, ai_1_wins))

                    print('AI_2: Loss {} - Accuracy: {} - Wins {}'
                    .format(ai_2_loss, ai_2_current_accuracy, ai_2_wins))

                    #weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_1/weights:0'))[0]
                    
                    #Move out into class
                    # plt.close('all')
                    # plt.figure()
                    # plt.imshow(weights, cmap='Greys_r', interpolation='none')
                    # plt.xlabel('Nodes')
                    # plt.ylabel('Inputs')
                    # plt.show()
                    # plt.close()

                    # user_input = input('Continue (y/n)')
                    # if user_input == 'n':
                    #     number_of_games_played = number_of_games
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

#Main().start_user_vs_ai(False)
Main().start_ai_vs_ai(False, 1000)
