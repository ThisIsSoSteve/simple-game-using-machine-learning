import os
from player import Player
from actions import Actions
from data import Data


class Game:
    def __init__(self, players_turn, state_length, q_value_length, action_length, reward_length):
        self.players_turn = players_turn
        self.game_over = False
        self.agent_1 = Player('Agent 1')
        self.agent_2 = Player('Agent 2')

        self.game_actions = Actions()

        self.player_1_training_data = Data(state_length, q_value_length, action_length, reward_length, True)
        self.player_2_training_data = Data(state_length, q_value_length, action_length, reward_length, False)

    def int_try_parse(self, value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    def check_for_reward(self):
        if self.agent_1.alive is False or self.agent_2.alive is False:
            if self.agent_1.alive is False:
                return 1
            else:
                return 1
        return 0

    def run_training(self, state, q_value, action):

        if self.players_turn: #Agents_1 turn
            #self.agent_1.print_health()

            self.game_actions.perfrom(self.agent_1, self.agent_2, action)

            self.player_1_training_data.record(state, q_value, action, self.check_for_reward())
                    
            self.players_turn = False

        else: #Agents_2 turn
            #self.agent_2.print_health()
            self.game_actions.perfrom(self.agent_2, self.agent_1, action)
            self.player_2_training_data.record(state, q_value, action, self.check_for_reward())

            self.players_turn = True

        if self.agent_1.alive is False or self.agent_2.alive is False:
            #os.system('cls')
            self.game_over = True

            if self.agent_1.alive is False:
                #print('You lost')
                return False
            else:
                #print('You Won')
                return True
        return None


    #Justs plays the game no data recording
    def run(self, opponents_action):
            if self.players_turn:
                self.agent_1.print_health()
                self.game_actions.display_player_actions(self.agent_1)
                print('5. Exit')

                user_input = input('Action (1-5)')
                players_action, is_valid = self.int_try_parse(user_input)
                os.system('cls')

                if is_valid and players_action > 0 and players_action <= 5:
                    if players_action == 5:
                        self.game_over = True
                    else:
                        self.game_actions.perfrom(self.agent_1, self.agent_2, players_action)
                        self.game_actions.display_player_chosen_action(self.agent_1, players_action)
                        
                    self.players_turn = False

                else:
                    print('Please enter a valid option from 1-5')
            else: #AI's turn

                self.agent_2.print_health()
                self.game_actions.display_ai_chosen_action(self.agent_2, opponents_action)
                self.game_actions.perfrom(self.agent_2, self.agent_1, opponents_action)

                self.players_turn = True

            if self.agent_1.alive is False or self.agent_2.alive is False:
                os.system('cls')
                self.game_over = True

                if self.agent_1.alive is False:
                    print('You lost')
                    return False
                else:
                    print('You Won')
                    return True
            return None
            