#Main.py Starting File
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil

from random import randint
import numpy as np
from AI import train
from AI import use
from player import Player
from actions import Actions
from data import Data

#Initialize game variables
play_game = True
player_turn = True
player_goes_first = True
number_of_turns = 0

user = Player('user')
ai = Player('opponent')
game_actions = Actions()

#training data 
player_training_data = Data(10)
ai_training_data = Data(10)

#checkpoint to use
checkpoint = ''

training_data_x = np.empty((0,10))
training_data_y = np.empty((0,4))

starting_stats = []

step = 0

logs_directory = "E:/Logs"
shutil.rmtree(logs_directory)
os.makedirs(logs_directory)

def set_defaults():

    global player_goes_first

    global player_turn
    player_turn = player_goes_first

    global number_of_turns
    number_of_turns = 0

    global user
    user = Player('user')

    global ai
    ai = Player('opponent')

    #reset training data
    global player_training_data

    global ai_training_data

    player_training_data = Data(10)
    ai_training_data = Data(10)

    os.system('cls')

def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def train_ai(did_player_win, players_turn_first):

    global training_data_x
    global training_data_y

    global ai_training_data
    global player_training_data
    
    if did_player_win:
        if players_turn_first:
            ai_training_data.data = np.insert(ai_training_data.data, 0, starting_stats, axis=0)
           
        training_data_x = np.concatenate((training_data_x, ai_training_data.data), axis=0)
        training_data_y = np.concatenate((training_data_y, player_training_data.data[:, :4]), axis=0)

    else:

        if players_turn_first == False:
            player_training_data.data = np.insert(player_training_data.data, 0, starting_stats, axis=0)

        training_data_x = np.concatenate((training_data_x, player_training_data.data), axis=0)
        training_data_y = np.concatenate((training_data_y, ai_training_data.data[:, :4]), axis=0)
        
    #print(training_data_x)
    #print(training_data_y)

    global checkpoint
    global step


    checkpoint, step = train.train_simple_model(training_data_x, training_data_y, checkpoint, step)

while(play_game):
    if number_of_turns == 0:
        starting_stats = np.array([0,0,0,0,user.attack/user.max_attack, user.defence/user.max_defence, user.health/user.max_health, ai.attack/ai.max_attack, ai.defence/ai.max_defence, ai.health/ai.max_health])

    if player_turn:
        user.print_health()
        game_actions.display_player_actions(user)
        print('5. Exit')

        user_input = input('Action (1-5)')
        players_action, is_valid = intTryParse(user_input)
        os.system('cls')

        if is_valid and players_action > 0 and players_action <= 5:

            player_training_data.record(players_action, ai, user) #Stores as AI would see ((ai, user) NOT (user, ai))
            game_actions.display_player_chosen_action(user, players_action)

            if players_action == 5:
                play_game = False
            else:
                game_actions.perfrom(user, ai, players_action)

            player_turn = False

        else:
            print('Please enter a valid option from 1-5')
    else: #AI's turn

        data = starting_stats
        if len(player_training_data.data) != 0:
            data = player_training_data.data[-1] 

        data = np.reshape(data, (-1, 10))
        
        ais_action = use.use_simple_model(checkpoint, data) + 1
        print('ai choice number: {}'.format(ais_action))

        ai_training_data.record(ais_action, user, ai)

        print('')
        ai.print_health()
        game_actions.display_ai_chosen_action(ai, ais_action)
        game_actions.perfrom(ai, user, ais_action)

        player_turn = True

    #if player_turn: #if player chooses an invalid action don't +1 to number_of_turns
    number_of_turns += 1

    if user.alive == False or ai.alive == False:

        if user.alive == False:
            os.system('cls')
            train_ai(False, player_goes_first)
            print('You lost')
        else:
            os.system('cls')
            train_ai(True, player_goes_first)
            print('You Won')

        player_goes_first = not player_goes_first
        play_again = input('Any key to continue..')
        set_defaults()

#using tensorboard
#E:
#tensorboard --logdir=Logs

#http://localhost:6006/