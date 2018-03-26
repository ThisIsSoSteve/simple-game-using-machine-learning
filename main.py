'''
Main.py Starting File
'''
import os
import numpy as np
import tensorflow as tf
from model import Model
from plot import Plot
from game import Game
from training_data import Training_data


#import matplotlib.pyplot as plt

class Main:
    def __init__(self, train):
        self.feature_length = 6
        self.label_length = 4

        self.cost_plot = Plot([], 'Step', 'Cost')
        self.accuracy_plot = Plot([], 'Step', 'Accuracy')
        self.checkpoint = 'data/checkpoints/turn_based_ai.ckpt'
        
        self.X = tf.placeholder(tf.float32, [None, self.feature_length])
        self.Y = tf.placeholder(tf.float32, [None, self.label_length])

        self.sequence_length = tf.placeholder(tf.float32)
        self.keep_probability = tf.placeholder(tf.float32)

        self.model = Model(self.X, self.Y, self.sequence_length, self.keep_probability)
        self.global_step = 0

        #replay memory
        #self.training_data_states = np.empty((0, self.feature_length))#X or Features
        #self.training_data_q_values = np.empty((0, self.label_length))#Y or label
        #self.training_data_actions = np.empty((0, self.label_length))#the actions that are taken for a state
        #self.training_data_reward = np.empty((0, 1))#rewards vector

        self.max_turns = 13

        self.training_data = Training_data(self.max_turns)
        self.test_data = Training_data(self.max_turns)

        self.reward_discount_factor = 0.60

        self.agent_1_wins = 0
        self.agent_2_wins = 0

        #self.test_training_data_x = np.empty((0, self.feature_length))
        #self.test_training_data_y = np.empty((0, self.label_length))

        self.strategies = ''

        fname = 'data/charts/record_everything.txt'

        if(train and os.path.isfile(fname)):
            os.remove(fname)
            with open(fname,'a') as file:
                file.write('')

        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # def add_training_data(self, features, labels):
    #     self.training_data_x = np.concatenate((self.training_data_x, features), axis=0)
    #     self.training_data_y = np.concatenate((self.training_data_y, labels), axis=0)
    # def add_training_data(self, states, q_values, actions, rewards, add_to_test_data):
    #     self.training_data_states = np.concatenate((self.training_data_states, states), axis=0)#features
    #     self.training_data_q_values = np.concatenate((self.training_data_q_values, q_values), axis=0)#labels
    #     self.training_data_actions = np.concatenate((self.training_data_actions, actions), axis=0)
    #     self.training_data_reward = np.concatenate((self.training_data_reward, rewards), axis=0)

    #     if add_to_test_data:
    #         self.test_training_data_x = np.concatenate((self.test_training_data_x, states), axis=0)
    #         self.test_training_data_y = np.concatenate((self.test_training_data_y, q_values), axis=0)

    def get_state_for_prediction(self, user, opponent, is_player_1, current_turn):
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
        range_length = np.size(q_values, 0) - 1

        for i in range(range_length, 0, -1):
            action_index = np.argmax(actions[i])
            reward = rewards[i]

            if i == range_length:
                q_values[i][action_index] = reward
            else:
                q_values[i][action_index] = reward + self.reward_discount_factor * np.max(q_values[i + 1])
            #print(q_values[i])
        return q_values

    def get_epsilon_greedy(self, iteration):
        #ToDo:Refactor 
        start_value = 1.0
        end_value = 0.1
        max_intergrations = 100#5e6
        _coefficient = (end_value - start_value) / max_intergrations

        if iteration < max_intergrations:
            value = iteration * _coefficient + start_value
        else:
            value = end_value

        return value

    #move to stats class
    def get_actions(self, actions):
        size = np.size(actions, 0)
        strategy = ""
        for i in range(size):
            strategy = strategy + str(np.argmax(actions[i]))

        #self.strategies = self.strategies + strategy + '\n'
        return strategy

    def log_everything(self, agent1_data, agent2_data, game_number, did_agent_1_won):
        agent1_data_size = np.size(agent1_data.states, 0)
        agent2_data_size = np.size(agent2_data.states, 0)
        turn_number = 0
        record = ''

        game_number_string = str(game_number)

        record = '\nGame #{}'.format(game_number_string)

        for i in range(agent1_data_size):

            if i <= agent1_data_size - 1:
                state = str(agent1_data.states[i]).replace('\n','')
                qvalues = agent1_data.q_values[i]
                action = agent1_data.actions[i]

                record += '\n\t\tAgent 1\'s Turn #{}\n\t\t\tState {}\n\t\t\tQvalues {}\n\t\t\tAction {}'.format(str(turn_number), state, qvalues, str(action))
                turn_number +=1

            if i <= agent2_data_size - 1:
                state = str(agent2_data.states[i]).replace('\n','')
                qvalues = agent2_data.q_values[i]
                action = agent2_data.actions[i]

                record += '\n\t\tAgent 2\'s Turn #{}\n\t\t\tState {}\n\t\t\tQvalues {}\n\t\t\tAction {}'.format(str(turn_number), state, qvalues, str(action))
                turn_number +=1

        if did_agent_1_won:
            record += '\n\t Agent 1 Won'
        else:
            record += '\n\t Agent 2 Won'

        with open('data/charts/record_everything.txt','a') as file:
            file.write(record)

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
                number_of_turns = 0

                while not game.game_over:
                    
                    current_state = np.zeros((0,self.feature_length))

                    action = 4
                    if game.players_turn == False:
                        #Get current state of the game
                        current_state = self.get_state_for_prediction(game.agent_1, game.agent_2, players_turn_first, number_of_turns)

                        states = game.player_2_training_data.states

                        states = np.append(states, current_state, axis=0)

                        sequence_length = len(states)

                        #add padding
                        new_records_size = self.max_turns - sequence_length

                        padding_array = np.zeros((new_records_size, self.feature_length))

                        states = np.append(states, padding_array,0)

                        print(states)

                        #Predict q values
                        q_value = sess.run(self.model.prediction, { self.X: states, self.sequence_length: sequence_length, self.keep_probability: 1.0})[0]
                        print(q_value)
                        action = np.argmax(q_value) + 1

                    #Play Game
                    did_player_win = game.run(current_state, action)

                    number_of_turns += 1

                    if game.game_over and did_player_win == None:
                        exit_game = True

    def start_training(self, restore, max_iterations):
        play = True
        train = False
        saver = tf.train.Saver()
        
        max_number_of_turns = self.max_turns - 1#12
        number_of_games = 0
        players_turn_first = True
        force_agent_1_to_lose = False
        number_of_forced_wins = 10
        #self.global_step = 400
        failed_games = 0
        #prev_training_number = 0
        #training_data_sets_size = np.empty((0), dtype=int)

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
                    current_state = self.get_state_for_prediction(game.agent_1, game.agent_2, game.players_turn, number_of_turns)
                    if game.players_turn: #if agents ones turn
                        states = game.player_1_training_data.states
                    else:
                        states = game.player_2_training_data.states

                    states = np.append(states, current_state, axis=0)

                    sequence_length = len(states)
                    
                    #add padding
                    new_records_size = self.max_turns - sequence_length

                    padding_array = np.zeros((new_records_size, self.feature_length))

                    states = np.append(states, padding_array,0)

                    #print(states)

                    #Predict q values
                    q_value = sess.run(self.model.prediction, { self.X: states, self.sequence_length: sequence_length, self.keep_probability: 1.0 })[0]
                    
                    choose_random = True
                    if force_agent_1_to_lose == False:
                        if game.players_turn:
                            epsilon_greedy = self.get_epsilon_greedy(self.global_step % 10)
                        else:
                            choose_random = False

                    if force_agent_1_to_lose:
                        if game.players_turn == False:
                            epsilon_greedy = self.get_epsilon_greedy(self.global_step % 10)
                        else:
                            choose_random = False

                    #decide to use random action
                    if choose_random: #and np.random.random() < epsilon_greedy:#self.get_epsilon_greedy(self.global_step % 100):
                        #print("random action")
                        action = np.random.randint(low=1, high=self.label_length+1)
                    else:
                        #print("action")
                        action = np.argmax(q_value) + 1

                    #Play Game
                    did_player_win = game.run_training(current_state, q_value, action)

                    if did_player_win == None and number_of_turns >= max_number_of_turns:
                        # if self.global_step > 100:
                        #     print('failed game')
                        failed_games += 1
                        
                        if failed_games > 1000 and self.training_data.number_of_examples() == 0:
                            print('Failed Games')
                            print('force agent 1 to lose: {}'.format(str(force_agent_1_to_lose)))
                            print('agent_1: {}'.format(self.get_actions(game.player_1_training_data.actions)))
                            print('agent_2: {}'.format(self.get_actions(game.player_2_training_data.actions)))
                            #play = False
                            #break
                            failed_games = 0
                            #max_number_of_turns = np.random.randint(low=8, high=20)
                            game.game_over = True
                        # else:
                        #     if np.size(self.training_data_reward, 0) > prev_training_number:
                        #         prev_training_number = np.size(self.training_data_reward, 0)
                                
                                # print('force agent 1 to lose: {}'.format(str(force_agent_1_to_lose)))
                                # print('agent_1: {}'.format(self.get_actions(game.player_1_training_data.actions)))
                                # print('agent_2: {}'.format(self.get_actions(game.player_2_training_data.actions)))
                                
                    elif game.game_over:
                        
                        
                        #record winning data
                        #print('found game')
                        #decide who we want to win
                        if self.agent_1_wins % number_of_forced_wins == 0 and self.agent_2_wins < self.agent_1_wins:
                            force_agent_1_to_lose = True
                        else:
                            force_agent_1_to_lose = False

                        if did_player_win and force_agent_1_to_lose == False:
                            failed_games = 0
                            print('New training data')
                            print('force agent 1 to lose: {}'.format(str(force_agent_1_to_lose)))
                            print('agent_1: {}'.format(self.get_actions(game.player_1_training_data.actions)))
                            print('agent_2: {}'.format(self.get_actions(game.player_2_training_data.actions)))
                            #train = True
                            number_of_games += 1
                            #print('player 1 wins')
                            self.agent_1_wins += 1
                            #self.log_everything(game.player_1_training_data, game.player_2_training_data, number_of_games, True)
                            #self.log_winning_actions(game.player_1_training_data.actions)

                            game.player_1_training_data.q_values = self.update_q_values(game.player_1_training_data.q_values, game.player_1_training_data.actions, game.player_1_training_data.rewards)
                            self.training_data.add_batch(game.player_1_training_data.states, game.player_1_training_data.q_values)
                            self.test_data.add_batch(game.player_1_training_data.states, game.player_1_training_data.q_values)
                            #self.add_training_data(game.player_1_training_data.states, game.player_1_training_data.q_values, game.player_1_training_data.actions, game.player_1_training_data.rewards, True)
                            #training_data_sets_size = np.append(training_data_sets_size, np.size(self.training_data_reward, 0))
                            #print(str(np.size(self.training_data_reward, 0)))
                        if did_player_win == False and force_agent_1_to_lose:
                            failed_games = 0
                            print('New training data')
                            print('force agent 1 to lose: {}'.format(str(force_agent_1_to_lose)))
                            print('agent_1: {}'.format(self.get_actions(game.player_1_training_data.actions)))
                            print('agent_2: {}'.format(self.get_actions(game.player_2_training_data.actions)))
                            
                            #train = True
                            number_of_games += 1
                            #print('player 2 wins')
                            self.agent_2_wins += 1
                            #self.log_everything(game.player_1_training_data, game.player_2_training_data, number_of_games, False)
                            #self.log_winning_actions(game.player_2_training_data.actions)
                            game.player_2_training_data.q_values = self.update_q_values(game.player_2_training_data.q_values, game.player_2_training_data.actions, game.player_2_training_data.rewards)
                            self.training_data.add_batch(game.player_2_training_data.states, game.player_2_training_data.q_values)
                            self.test_data.add_batch(game.player_2_training_data.states, game.player_2_training_data.q_values)
                            #self.add_training_data(game.player_2_training_data.states, game.player_2_training_data.q_values, game.player_2_training_data.actions, game.player_2_training_data.rewards, True)
                            #training_data_sets_size = np.append(training_data_sets_size, np.size(self.training_data_reward, 0))
                            #print(str(np.size(self.training_data_reward, 0)))
                        
                    if number_of_turns >= max_number_of_turns:
                        break
                    number_of_turns += 1
                
                #we only train if we have correct number of wins in the training data
                if(self.training_data.number_of_examples() > 1 and number_of_games % number_of_forced_wins == 0 ):
                    train = True

                

                #Train
                if train:
                    #prev_training_number = 0
                    for _ in range(10):
                        avarage_loss = 0
                        #training_data_size = np.size(self.training_data_reward, 0)
                        #random_range = np.arange(training_data_size)
                        #np.random.shuffle(random_range)

                        #for i in range(training_data_size):
                            #random_index = random_range[i]
                            #_, loss = sess.run(self.model.optimize, { self.X: np.reshape(self.training_data_states[random_index], (-1, self.feature_length)), self.Y: np.reshape(self.training_data_q_values[random_index],(-1, 4))})
                            #avarage_loss += loss

                        # training_data_size = np.size(self.training_data_states, 0)
                        
                        # print(self.training_data_states.shape)

                        # x = self.training_data_states.reshape(1,training_data_size,7)
                        
                        # print(x.shape)
                        # print(x)


                        # print(training_data_sets_size)
                        # print(training_data_sets_size.shape)
                        # print(training_data_sets_size[0])
                        training_data_length = self.training_data.number_of_examples()

                        random_range = np.arange(training_data_length)
                        np.random.shuffle(random_range)

                        # print(self.training_data.batched_features)
                        # print(self.training_data.batched_features[0])
                        # print(self.training_data.batched_labels[0])

                        for i in range(training_data_length):
                            
                            random_index = random_range[i]

                            state, label, sequence_length = self.training_data.get_batch_by_index(i)
                            
                            
                            #transform list to numpy array
                            state = np.asarray(state)
                            label = np.asarray([label])

                            # state = np.reshape(state, (-1, self.feature_length))
                            # label = np.reshape(label, (-1, self.label_length))
                            # #print(np.reshape(label.shape, -1))


                            #print('about to train')
                            # print(state)
                            # print(label)

                            _, loss = sess.run(self.model.optimize, { self.X: state, self.Y: label, self.sequence_length: sequence_length, self.keep_probability: 1.0 })

                            # batch_index_start = 0
                            # batch_index_end = training_data_sets_size[i]

                            # if i > 0:
                            #     batch_index_start = training_data_sets_size[i-1]
                            #     batch_index_end = training_data_sets_size[i]
                            
                            # if i == training_data_size - 1:
                            #     batch_index_end = np.size(self.training_data_reward, 0)
                                
                            

                            #print("batch_start: {} batch_end: {}".format(str(batch_index_start), str(batch_index_end)))
                            #print(self.training_data_states[batch_index_start:batch_index_end])
                                # print(self.training_data_states[batch_index_start:batch_index_end].shape)

                            #_, loss = sess.run(self.model.optimize, { self.X: self.training_data_states[batch_index_start:batch_index_end], self.Y: self.training_data_q_values[batch_index_start:batch_index_end], self.keep_probability: 1.0 })
                            avarage_loss += loss

                        self.global_step += 1

                        final_loss = avarage_loss / training_data_length
                        self.cost_plot.data.append(final_loss)#save avarage loss
                        
                        avarage_accuracy = 0

                        test_data_length = self.test_data.number_of_examples()

                        for i in range(test_data_length):

                            state, label, sequence_length = self.test_data.get_batch_by_index(i)
                            
                            #print('about to find accuracy')
                            #transform list to numpy array
                            state = np.asarray(state)
                            label = np.asarray([label])

                            accuracy = sess.run(self.model.error, { self.X: state, self.Y: label, self.sequence_length: sequence_length, self.keep_probability: 1.0 })
                            avarage_accuracy += accuracy

                        final_avarage = avarage_accuracy / test_data_length
                        self.accuracy_plot.data.append(final_avarage)

                    #print('Saving...')
                    saver.save(sess, self.checkpoint)

                    print('Epoch {} - Loss {} - Accuracy {} - A1W={} A2W={}'.format(self.global_step, final_loss, final_avarage, self.agent_1_wins, self.agent_2_wins))

                    #clear training data
                    self.training_data = Training_data(self.max_turns)

                    # self.training_data_states = np.empty((0, self.feature_length))#X or Features
                    # self.training_data_q_values = np.empty((0, self.label_length))#Y or label
                    # self.training_data_actions = np.empty((0, self.label_length))#the actions that are taken for a state
                    # self.training_data_reward = np.empty((0, 1))#rewards vector

                    #print(training_data_sets_size)
                    #training_data_sets_size = np.empty((0),dtype=int)

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

                if number_of_games == max_iterations:
                    break
            # with open('data/charts/strategies.txt', 'w') as file:
            #     file.write(self.strategies)
#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main(True).start_training(False, 200)
#Main(False).start_testing()

# training_data_states = np.random.rand(10, 7)
# print(training_data_states.shape)

# print('sep')
# print(training_data_states[0:2])
# print(training_data_states[0:2].shape)