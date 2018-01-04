'''
Used to display and perfrom actions for the user and their opponent
'''

class Actions:

    def __init__(self):
        #ToDo: pass in user and ai
        #ToDo: create action type enum
        self.action_type_attack = 1
        self.action_type_affect = 2

        #actions lists
        self.player_actions = (('Attack', self.action_type_attack),
                               ('Increase my defence', self.action_type_affect),
                               ('Decrease their attack', self.action_type_affect),
                               ('Decrease their defence', self.action_type_affect))

        self.ai_actions = (('Attack', self.action_type_attack),
                           ('Increase their defence', self.action_type_affect),
                           ('Decrease your attack', self.action_type_affect),
                           ('Decrease your defence', self.action_type_affect))

    def get_value_for_action(self, player, action_type):
        value = 0
        if action_type == 1:
            value = player.attack
        elif action_type == 2:
            value = player.affect
        return value

    def display_player_actions(self, player):
        for index, action in enumerate(self.player_actions, start=1):
            value = self.get_value_for_action(player, action[1])

            print('{}. {} ({})'.format(index, action[0], value))

    def display_ai_actions(self, player):
        for index, action in enumerate(self.ai_actions, start=1):
            value = self.get_value_for_action(player, action[1])

            print('{}. {} ({})'.format(index, action[0], value))

    def display_player_chosen_action(self, player, choice):
        value = self.get_value_for_action(player, self.player_actions[choice - 1][1])
        print('You chose to {} ({})\n'.format(self.player_actions[choice - 1][0], value))

    def display_ai_chosen_action(self, player, choice):
        value = self.get_value_for_action(player, self.ai_actions[choice - 1][1])
        print('AI chose to {} ({})\n'.format(self.ai_actions[choice - 1][0], value))


    def perfrom(self, player_chooser, player_receiver, choice):
        if choice == 4:#decrease ai defence
            player_receiver.decrease_defence(player_chooser.affect)
        elif choice == 3:#decrease ai attack
            player_receiver.decrease_attack(player_chooser.affect)
        elif choice == 2:#increase my defence
            player_chooser.increase_defence(player_chooser.affect)
        elif choice == 1: #damage ai
            player_receiver.decrease_health(player_chooser.attack)