'''
Main.py Starting File
'''
import os
import shutil
import numpy as np
from AI import train
from AI import use
from player import Player
from actions import Actions
from data import Data

class Main:
    def __init__(self):
        self.play_game = True
        self.player_turn = True
        self.player_goes_first = True
        self.number_of_turns = 0

        self.user = Player('user')
        self.opponent = Player('opponent')
        self.game_actions = Actions()
        #training data
        self.player_training_data = Data(10)
        self.opponent_training_data = Data(10)

        #checkpoint to use
        self.checkpoint = ''

        self.training_data_x = np.empty((0, 10))
        self.training_data_y = np.empty((0, 4))

        self.starting_stats = []

        self.step = 0

        self.logs_directory = "E:/Logs"
        shutil.rmtree(self.logs_directory)
        os.makedirs(self.logs_directory)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def set_defaults(self):

        self.player_turn = self.player_goes_first

        self.number_of_turns = 0

        self.user = Player('user')
        self.opponent = Player('opponent')

        #reset training data
        self.player_training_data = Data(10)
        self.opponent_training_data = Data(10)

        os.system('cls')

    #ToDo: Move to common
    def intTryParse(self, value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    def train_ai(self, did_player_win, players_turn_first):
        if did_player_win:
            if players_turn_first:
                self.opponent_training_data.data = np.insert(self.opponent_training_data.data, 0, self.starting_stats, axis=0)
                
            self.training_data_x = np.concatenate((self.training_data_x, self.opponent_training_data.data), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.player_training_data.data[:, :4]), axis=0)
        else:
            if players_turn_first is False:
                self.player_training_data.data = np.insert(self.player_training_data.data, 0, self.starting_stats, axis=0)

            self.training_data_x = np.concatenate((self.training_data_x, self.player_training_data.data), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.opponent_training_data.data[:, :4]), axis=0)

        self.checkpoint, self.step = train.train_simple_model(self.training_data_x, self.training_data_y, self.checkpoint, self.step)

    def run_game(self):
        while(self.play_game):
            if self.number_of_turns == 0:
                #ToDo Move this out
                self.starting_stats = np.array([0, 0, 0, 0,
                                           self.user.attack / self.user.max_attack,
                                           self.user.defence / self.user.max_defence,
                                           self.user.health / self.user.max_health,
                                           self.opponent.attack / self.opponent.max_attack,
                                           self.opponent.defence / self.opponent.max_defence,
                                           self.opponent.health / self.opponent.max_health])

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
                        self.game_actions.perfrom(self.user, self.opponent, players_action)
                        self.player_training_data.record(players_action, self.opponent, self.user) #Stores as AI would see ((ai, user) NOT (user, ai))
                        self.game_actions.display_player_chosen_action(self.user, players_action)
                        
                    self.player_turn = False

                else:
                    print('Please enter a valid option from 1-5')
            else: #AI's turn

                data = self.starting_stats
                if len(self.player_training_data.data) != 0:#if not self.player_training_data.data
                    data = self.player_training_data.data[-1] 

                data = np.reshape(data, (-1, 10))
                
                ais_action = use.use_simple_model(self.checkpoint, data) + 1
                print('opponent\'s choice number: {}'.format(ais_action))

                self.opponent_training_data.record(ais_action, self.user, self.opponent)

                print('')
                self.opponent.print_health()
                self.game_actions.display_ai_chosen_action(self.opponent, ais_action)
                self.game_actions.perfrom(self.opponent, self.user, ais_action)

                self.player_turn = True

            #if player_turn: #if player chooses an invalid action don't +1 to number_of_turns
            self.number_of_turns += 1

            if self.user.alive is False or self.opponent.alive is False:

                if self.user.alive is False:
                    os.system('cls')
                    self.train_ai(False, self.player_goes_first)
                    print('You lost')
                else:
                    os.system('cls')
                    self.train_ai(True, self.player_goes_first)
                    print('You Won')

                self.player_goes_first = not self.player_goes_first
                input('Any key to continue..')
                self.set_defaults()

#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main().run_game()
