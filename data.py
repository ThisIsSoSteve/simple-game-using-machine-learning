import numpy as np

class Data:
    def __init__(self, array_length):
        
        self.data = np.empty((0, array_length)) #dtype=np.int16 

    def record(self, action, player_1, player_2):
        label = np.zeros((1,10))
        label[0, action-1] = 1 #one hot label
        #label[0-3] = one hot action performed

        label[0,4] = player_1.attack / player_1.max_attack
        label[0,5] = player_1.defence / player_1.max_defence
        label[0,6] = player_1.health / player_1.max_health

        label[0,7] = player_2.attack / player_2.max_attack
        label[0,8] = player_2.defence / player_2.max_defence
        label[0,9] = player_2.health / player_2.max_health

        self.data = np.append(self.data, label, axis=0)
        #print(label)
