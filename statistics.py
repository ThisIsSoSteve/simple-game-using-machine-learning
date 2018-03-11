import os
import numpy as np

file_path = 'data/charts/record_everything.txt'

with open(file_path) as f:
    content = f.readlines()

#get unique strategies
def find_all_unique_games():

    games = np.empty((0,1), dtype=str)

    player_1_game_temp = ''
    player_2_game_temp = ''

    look_for_game = True
    #look_for_action = False
    player_1 = True

    for line in content:
        if 'Game' in line:
            look_for_game = not look_for_game
            if look_for_game:
                games = np.vstack((games, np.array([player_1_game_temp + ' '])))
                games = np.vstack((games, np.array([player_2_game_temp + ' '])))
                player_1_game_temp = ''
                player_2_game_temp = ''
                player_1 = True

        if look_for_game == False:
            if 'Action' in line:

                new_line = str(np.argmax(np.array(list(remove_unwanted_text(line)))) + 1)

                if player_1:
                    player_1_game_temp += new_line
                else:
                    player_2_game_temp += new_line

                player_1 = not player_1

    unique_games = np.unique(games)
    #print(unique_games)
    np.savetxt('data/charts/unique_games.txt', unique_games, fmt='%s')#delimiter=" "

def remove_unwanted_text(string):
    return string.replace('Action ', '').replace('\t','').replace('\n','').replace('[','').replace(']','').replace('.','').replace(' ','')

#get all states that are the same but that occur for different agents

#get number of unique games


find_all_unique_games()