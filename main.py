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

        #replay memory
        self.training_data_states = np.empty((0, self.feature_length))#X or Features
        self.training_data_q_values = np.empty((0, self.label_length))#Y or label
        self.training_data_actions = np.empty((0, self.label_length))#the actions that are taken for a state
        self.training_data_reward = np.empty((0, 1))#rewards vector

        self.reward_discount_factor = 0.80

        self.agent_1_wins = 0
        self.agent_2_wins = 0

        self.test_training_data_x = np.empty((0, self.feature_length))
        self.test_training_data_y = np.empty((0, self.label_length))

        self.strategies = ''
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # def add_training_data(self, features, labels):
    #     self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
    #     self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)
    def add_training_data(self, states, q_values, actions, rewards, add_to_test_data):
        self.training_data_states = np.concatenate((self.training_data_states, states), axis=0)#features
        self.training_data_q_values = np.concatenate((self.training_data_q_values, q_values), axis=0)#labels
        self.training_data_actions = np.concatenate((self.training_data_actions, actions), axis=0)
        self.training_data_reward = np.concatenate((self.training_data_reward, rewards), axis=0)

        if add_to_test_data:
            self.test_training_data_x = np.concatenate((self.test_training_data_x, states), axis=0)
            self.test_training_data_y = np.concatenate((self.test_training_data_y, q_values), axis=0)



    def get_state_for_prediction(self, user, opponent, is_player_1):
        if is_player_1:
            new_state = np.array([opponent.attack / opponent.max_attack,
                        opponent.defence / opponent.max_defence,
                        opponent.health / opponent.max_health,
                        user.attack / user.max_attack,
                        user.defence / user.max_defence,
                        user.health / user.max_health])
        else:
            new_state = np.array([user.attack / user.max_attack,
                        user.defence / user.max_defence,
                        user.health / user.max_health,
                        opponent.attack / opponent.max_attack,
                        opponent.defence / opponent.max_defence,
                        opponent.health / opponent.max_health])

        return np.reshape(new_state, (-1, self.feature_length))

    def update_q_values(self, q_values, actions, rewards):
        range_length = np.size(q_values, 0)

        for i in range(range_length-1, 0, -1):
            action_index = np.argmax(actions[i])
            reward = rewards[i]

            if reward == 1:
                q_values[i][action_index] = reward
            else:
                q_values[i][action_index] = reward + self.reward_discount_factor * np.max(q_values[i + 1])
            #print(q_values[i])
        return q_values

    def get_epsilon_greedy(self, iteration):
        #ToDo:Refactor 
        start_value = 1.0
        end_value = 0.1
        max_intergrations = 300#5e6
        _coefficient = (end_value - start_value) / max_intergrations

        if iteration < max_intergrations:
            value = iteration * _coefficient + start_value
        else:
            value = end_value

        return value

    #move to stats class
    def log_winning_actions(self, actions):
        size = np.size(actions, 0)
        strategy = ""
        for i in range(size):
            strategy = strategy + str(np.argmax(actions[i]))

        self.strategies = self.strategies + strategy + '\n'


    def start_testing(self):
        exit_game = False
        players_turn_first = True
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.checkpoint)
            #sess.run(tf.global_variables_initializer())

            while not exit_game:
                game = Game(players_turn_first, self.feature_length, self.label_length, self.label_length, 1)
                players_turn_first = not players_turn_first
                game.agent_1.print_game_text = True
                game.agent_2.print_game_text = True

                while not game.game_over:
                    
                    action = 4
                    if game.players_turn == False:
                        #Get current state of the game
                        state = self.get_state_for_prediction(game.agent_1, game.agent_2, False)
                        #Predict q values
                        q_value = sess.run(self.model.prediction, { self.X: state })[0]
                        print(q_value)
                        action = np.argmax(q_value) + 1

                    #Play Game
                    did_player_win = game.run(action)

                    if game.game_over and did_player_win == None:
                        exit_game = True

    def start_training(self, restore, max_iterations):
        play = True
        train = False
        saver = tf.train.Saver()
        
        max_number_of_turns = 10
        number_of_games = 0
        players_turn_first = True
        switch = False

        with tf.Session() as sess:

            if restore:
                saver.restore(sess, self.checkpoint)
            else:
                sess.run(tf.global_variables_initializer())

            while play:
                #New game
                game = Game(players_turn_first, self.feature_length, self.label_length, self.label_length, 1)
                
                number_of_turns = 0
                train = False
                while not game.game_over:
                    
                    #Get current state of the game
                    state = self.get_state_for_prediction(game.agent_1, game.agent_2, game.players_turn)
                    #Predict q values
                    q_value = sess.run(self.model.prediction, { self.X: state })[0]
                    
                    choose_random = True
                    if switch == False:
                        if game.players_turn:
                            epsilon_greedy = self.get_epsilon_greedy(self.global_step % 10)
                        else:
                            choose_random = False

                    if switch:
                        if game.players_turn == False:
                            epsilon_greedy = self.get_epsilon_greedy(self.global_step % 10)
                        else:
                            choose_random = False

                    #decide to use random action
                    if choose_random and np.random.random() < epsilon_greedy:#self.get_epsilon_greedy(self.global_step % 100):
                        #print("random action")
                        action = np.random.randint(low=1, high=self.label_length+1)
                    else:
                        #print("action")
                        action = np.argmax(q_value) + 1

                    #Play Game
                    did_player_win = game.run_training(state, q_value, action)

                    if game.game_over and did_player_win == None:
                        play = False
                    elif game.game_over:
                        #record winning data

                        if self.agent_1_wins % 10 == 0 and self.agent_2_wins < self.agent_1_wins:
                            switch = True
                        else:
                            switch = False

                        if did_player_win and switch == False:
                            #print('Switch {}'.format(switch))
                            train = True
                            number_of_games += 1
                            #print('player 1 wins')
                            self.agent_1_wins += 1
                            self.log_winning_actions(game.player_1_training_data.actions)
                            game.player_1_training_data.q_values = self.update_q_values(game.player_1_training_data.q_values, game.player_1_training_data.actions, game.player_1_training_data.rewards)
                            self.add_training_data(game.player_1_training_data.states, game.player_1_training_data.q_values, game.player_1_training_data.actions, game.player_1_training_data.rewards, True)
                        
                        if did_player_win == False and switch:
                            #print('Switch {}'.format(switch))
                            train = True
                            number_of_games += 1
                            #print('player 2 wins')
                            self.agent_2_wins += 1
                            self.log_winning_actions(game.player_2_training_data.actions)
                            game.player_2_training_data.q_values = self.update_q_values(game.player_2_training_data.q_values, game.player_2_training_data.actions, game.player_2_training_data.rewards)
                            self.add_training_data(game.player_2_training_data.states, game.player_2_training_data.q_values, game.player_2_training_data.actions, game.player_2_training_data.rewards, True)

                    if number_of_turns > max_number_of_turns:
                        break
                    number_of_turns += 1
                    
                if number_of_games == max_iterations:
                    #print(self.training_data_q_values)
                    break
                #Train
                if train:
                    
                    #NOTE change to train once ai has won 10 times
                    for _ in range(1):
                        
                        training_data_size = np.size(self.training_data_reward, 0)
                        random_range = np.arange(training_data_size)
                        np.random.shuffle(random_range)

                        for i in range(training_data_size):
                            random_index = random_range[i]
                            _, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_states[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_q_values[random_index],(-1, 4))})
                        #_, loss = sess.run(model.optimize, { X: self.training_data_x, Y: self.training_data_y })
                        self.global_step += 1

                        current_accuracy = sess.run(self.model.error, { self.X: self.test_training_data_x, self.Y: self.test_training_data_y })
                        self.cost_plot.data.append(loss)
                        self.accuracy_plot.data.append(current_accuracy)

                    #print('Saving...')
                    saver.save(sess, self.checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {} - A1W={} A2W={}'.format(self.global_step, loss, current_accuracy, self.agent_1_wins, self.agent_2_wins))

                    self.training_data_states = np.empty((0, self.feature_length))#X or Features
                    self.training_data_q_values = np.empty((0, self.label_length))#Y or label
                    self.training_data_actions = np.empty((0, self.label_length))#the actions that are taken for a state
                    self.training_data_reward = np.empty((0, 1))#rewards vector
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

            with open('data/charts/strategies.txt', 'w') as file:
                file.write(self.strategies)
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

#Main().start_training(True, 1000)
Main().start_testing()