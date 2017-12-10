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

        self.feature_length = 10
        self.label_length = 4

        self.user = Player('user', self.label_length)
        self.opponent = Player('opponent', self.label_length)
        self.game_actions = Actions()
        #training data
        self.player_training_data = Data(self.feature_length, self.label_length)
        self.opponent_training_data = Data(self.feature_length, self.label_length)

        #checkpoint to use
        self.checkpoint = ''

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

        self.logs_directory = "E:/Logs"
        shutil.rmtree(self.logs_directory)
        os.makedirs(self.logs_directory)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    def train_ai(self, did_player_win):
        if did_player_win:
            self.training_data_x = np.concatenate((self.training_data_x, self.player_training_data.feature), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.player_training_data.label), axis=0)
        else:
            self.training_data_x = np.concatenate((self.training_data_x, self.opponent_training_data.feature), axis=0)
            self.training_data_y = np.concatenate((self.training_data_y, self.opponent_training_data.label), axis=0)

        print(self.training_data_x)
        print(self.training_data_y)

        self.checkpoint, self.step = train.train_simple_model(self.training_data_x, self.training_data_y, self.checkpoint, self.step)

    def run_game(self):
        while(self.play_game):
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

                data = self.starting_stats
                if len(self.player_training_data.feature) != 0:#if not self.player_training_data.data
                    #ToDo: find better way to do this
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

                data = np.reshape(data, (-1, 10))
                print('opponents\'s view: {}'.format(data))

                opponents_action = use.use_simple_model(self.checkpoint, data) + 1
                print('opponent\'s choice number: {}'.format(opponents_action))

                self.opponent_training_data.record(opponents_action, self.user, self.opponent, False)

                print('')
                self.opponent.print_health()
                self.game_actions.display_ai_chosen_action(self.opponent, opponents_action)
                self.game_actions.perfrom(self.opponent, self.user, opponents_action)

                self.player_turn = True

            #if player_turn: #if player chooses an invalid action don't +1 to number_of_turns
            self.number_of_turns += 1

            if self.user.alive is False or self.opponent.alive is False:

                if self.user.alive is False:
                    os.system('cls')
                    self.train_ai(False)
                    print('You lost')
                else:
                    os.system('cls')
                    self.train_ai(True)
                    print('You Won')

                self.player_goes_first = not self.player_goes_first
                input('Any key to continue..')
                self.set_defaults()

#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/

Main().run_game()

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