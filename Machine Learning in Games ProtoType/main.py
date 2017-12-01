#Main.py Starting File
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from random import randint
import numpy as np
from AI import train
from AI import use
from player import Player
from actions import Actions

#Initialize game variables
play_game = True
player_turn = True
player_goes_first = True
number_of_turns = 0

user = Player('user')
ai = Player('opponent')
game_actions = Actions()

#training data 
player_training_data = np.empty((0,10))#dtype=np.int16 
ai_training_data = np.empty((0,10))

#checkpoint to use
checkpoint = ''

training_data_x = np.empty((0,10))
training_data_y = np.empty((0,4))

starting_stats = []

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

    #global starting_stats
    #starting_stats = np.array([0,0,0,0,user.attack, user.defence, user.health, ai.attack, ai.defence, ai.health])

    #reset training data
    global player_training_data
    player_training_data = np.empty((0,10))

    global ai_training_data
    ai_training_data = np.empty((0,10))

    os.system('cls')

def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def record_player_action_data(action_performed, current_player, current_ai):
    #print('player {}'.format(action_performed))
    label = np.zeros((1,10))
    label[0, action_performed-1] = 1 #one hot label
    #label[0-3] = one hot action performed

    label[0,4] = current_player.attack / current_player.max_attack
    label[0,5] = current_player.defence / current_player.max_defence
    label[0,6] = current_player.health / current_player.max_health

    label[0,7] = current_ai.attack / current_ai.max_attack
    label[0,8] = current_ai.defence / current_ai.max_defence
    label[0,9] = current_ai.health / current_ai.max_health

    #player_training_data.append(label)
    global player_training_data
    player_training_data = np.append(player_training_data, label, axis=0)
    #print(player_training_data)

def record_ai_action_data(action_performed, current_player, current_ai):
   # print('ai {}'.format(action_performed))
    label = np.zeros((1,10))
    label[0,action_performed-1] = 1 #one hot label

    label[0,4] = current_player.attack / current_player.max_attack
    label[0,5] = current_player.defence / current_player.max_defence
    label[0,6] = current_player.health / current_player.max_health

    label[0,7] = current_ai.attack / current_ai.max_attack
    label[0,8] = current_ai.defence / current_ai.max_defence
    label[0,9] = current_ai.health / current_ai.max_health
    
    #ai_training_data.append(label)
    global ai_training_data
    ai_training_data = np.append(ai_training_data, label, axis=0)


def train_ai(did_player_win, players_turn_first):

    global training_data_x
    global training_data_y

    global ai_training_data
    global player_training_data
    
    if did_player_win:
        if players_turn_first:
            ai_training_data = np.insert(ai_training_data, 0, starting_stats, axis=0)
           
        training_data_x = np.concatenate((training_data_x, ai_training_data), axis=0)
        training_data_y = np.concatenate((training_data_y, player_training_data[:, :4]), axis=0)

    else:

        if players_turn_first == False:
            player_training_data = np.insert(player_training_data, 0, starting_stats, axis=0)

        training_data_x = np.concatenate((training_data_x, player_training_data), axis=0)
        training_data_y = np.concatenate((training_data_y, ai_training_data[:, :4]), axis=0)
        
    #Return the unique rows of a 2D array

    #a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    #np.unique(a, axis=0)
    #array([[1, 0, 0], [2, 3, 4]])

    #print(training_data_x)
    #print(training_data_y)

    global checkpoint

    current_accuracy = 0.0

    #while current_accuracy == 0.0:
    checkpoint, current_accuracy = train.train_simple_model(training_data_x, training_data_y, checkpoint)

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
            record_player_action_data(players_action, user, ai)
            game_actions.display_player_chosen_action(user, players_action)

            if players_action == 5:
                play_game = False
            else:
                game_actions.perfrom(user, ai, players_action)

            player_turn = False

        else:
            print('Please enter a valid option from 1-5')
    else: #AI's turn

        data = starting_stats #np.array([np.zeros(4)])
        if len(player_training_data) != 0:
            data = player_training_data[-1] 

        data = np.reshape(data, (-1, 10))
        ais_action = use.use_simple_model(checkpoint, data) + 1
        print('ai choice number: {}'.format(ais_action))
        record_ai_action_data(ais_action, user, ai)
        #ais_action = randint(1, 4)
        #print('AIs raw Action = {}'.format(ais_action))
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

        #player_goes_first = not player_goes_first
        #play_again = input('Play again? (y/n)')
        #if play_again == 'y':
        #    set_defaults()
        #else:
        #    play_game = False
        #os.system('cls')
