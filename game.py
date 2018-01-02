import os
from player import Player
from actions import Actions
from data import Data


class Game:
    def __init__(self, players_turn, feature_length, label_length):
        self.players_turn = players_turn
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
        #self.number_of_turns += 1

        if self.user.alive is False or self.opponent.alive is False:
            os.system('cls')
            #self.add_training_data(self.user.alive)

            if self.user.alive is False:
                print('You lost')
                return True, False, self.opponent_training_data.feature, self.opponent_training_data.label
            else:
                print('You Won')
                return True, False, self.player_training_data.feature, self.player_training_data.label

            #self.player_goes_first = not self.player_goes_first
            #input('Press any key to continue..')
            #self.set_defaults()
            #return True
        
        return False, self.player_turn, None, None