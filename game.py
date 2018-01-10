import os
from player import Player
from actions import Actions
from data import Data


class Game:
    def __init__(self, players_turn, feature_length, label_length):
        self.players_turn = players_turn
        self.game_over = False
        self.user = Player('user') #ToDo: Rename variable to Player_1
        self.opponent = Player('opponent') #ToDo: Rename variable to Player_2

        self.game_actions = Actions()

        self.player_training_data = Data(feature_length, label_length)
        self.opponent_training_data = Data(feature_length, label_length)

    def int_try_parse(self, value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    def run(self, opponents_action):
        if self.players_turn:
            self.user.print_health()
            self.game_actions.display_player_actions(self.user)
            print('5. Exit')

            user_input = input('Action (1-5)')
            players_action, is_valid = self.int_try_parse(user_input)
            os.system('cls')

            if is_valid and players_action > 0 and players_action <= 5:
                if players_action == 5:
                    self.game_over = True
                else:
                    self.player_training_data.record(players_action, self.user, self.opponent, True)
                    self.game_actions.perfrom(self.user, self.opponent, players_action)
                    self.game_actions.display_player_chosen_action(self.user, players_action)
                    
                self.players_turn = False

            else:
                print('Please enter a valid option from 1-5')
        else: #AI's turn
            #print('opponent\'s choice number: {}'.format(opponents_action))

            self.opponent_training_data.record(opponents_action, self.user, self.opponent, False)

            self.opponent.print_health()
            self.game_actions.display_ai_chosen_action(self.opponent, opponents_action)
            self.game_actions.perfrom(self.opponent, self.user, opponents_action)

            self.players_turn = True

        if self.user.alive is False or self.opponent.alive is False:
            os.system('cls')
            self.game_over = True

            if self.user.alive is False:
                print('You lost')
                return False #self.opponent_training_data
            else:
                print('You Won')
                return True #self.player_training_data
        
        return None

    def run_ai_game(self, action):

        if self.players_turn: #AI_1
            self.player_training_data.record(action, self.user, self.opponent, True)

            self.user.print_health()
            #self.game_actions.display_player_chosen_action(self.user, action)
            self.game_actions.perfrom(self.user, self.opponent, action)

            self.players_turn = False
           
        else: #AI_2
            self.opponent_training_data.record(action, self.user, self.opponent, False)

            self.opponent.print_health()
            #self.game_actions.display_ai_chosen_action(self.opponent, action)
            self.game_actions.perfrom(self.opponent, self.user, action)

            self.players_turn = True

        if self.user.alive is False or self.opponent.alive is False:
            #os.system('cls')
            self.game_over = True
            if self.user.alive is False:
                print('AI_2 Won')
                return False #self.opponent_training_data
            else:
                print('AI_1 Won')
                return True #self.player_training_data
        
        return None