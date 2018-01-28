import os
from player import Player
from actions import Actions
from data import Data


class Game:
    def __init__(self, players_turn, feature_length, label_length):
        self.players_turn = players_turn
        self.game_over = False
        self.user = Player('user')
        self.opponent = Player('opponent')

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
                return False
                #return True, self.players_turn, self.user, self.opponent, self.opponent_training_data
            else:
                print('You Won')
                return True
                #return True, self.players_turn, self.user, self.opponent, self.player_training_data
        return None
        #return self.game_over, self.players_turn, self.user, self.opponent, None