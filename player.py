import numpy as np

class Player:

    max_health = 10
    max_attack = 5
    max_defence = 5
    min_limit = 1

    def __init__(self, name, print_game_text = True): #label_length
        #display name
        self.name = name
        self.alive = True
        self.print_game_text = print_game_text
        #stats
        self.attack = 5
        self.defence = 2
        self.health = Player.max_health
        self.affect = 1
        #self.last_action = np.zeros((1, label_length))

    def print_health(self):
        print('{}\'s Health: {}/{}'.format(self.name, self.health, Player.max_health))

    def increase_health(self, amount):
        if self.health < Player.max_health:
            self.health += amount
            if self.print_game_text:
                print('{}\'s health increased by {}'.format(self.name, amount))
        else:
            if self.print_game_text:
                print('{}\'s health stayed the same'.format(self.name))

    def decrease_health(self, amount):
        amount = amount - self.defence
        if amount <= 0:
            amount = 1

        self.health -= amount
        if self.print_game_text:
            print('{}\'s health decreased by {}'.format(self.name, amount))
        if self.health <= 0:
            self.alive = False
            if self.print_game_text:
                print('{} has died!'.format(self.name))

    def increase_defence(self, amount):
        if self.defence < Player.max_defence:
            self.defence += amount
            if self.print_game_text:
                print('{}\'s defence increased by {}'.format(self.name, amount))
        else:
            if self.print_game_text:
                print('{}\'s defence stayed the same'.format(self.name))

    def decrease_defence(self, amount):
        if self.defence > Player.min_limit:
            self.defence -= amount
            if self.print_game_text:
                print('{}\'s defence decreased by {}'.format(self.name, amount))
        else:
            if self.print_game_text:
                print('{}\'s defence stayed the same'.format(self.name))

    def increase_attack(self, amount):
        if self.attack < Player.max_attack:
            self.attack += amount
            if self.print_game_text:
                print('{}\'s attack increased by {}'.format(self.name, amount))
        else:
            if self.print_game_text:
                print('{}\'s attack stayed the same'.format(self.name))

    def decrease_attack(self, amount):
        if self.attack > Player.min_limit:
            self.attack -= amount
            if self.print_game_text:
                print('{}\'s attack decreased by {}'.format(self.name, amount))
        else:
            if self.print_game_text:
                print('{}\'s attack stayed the same'.format(self.name))