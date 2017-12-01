import numpy as np

class Data:
    def __init__(self, array_length):
        
        self.data = np.empty((0, array_length)) #dtype=np.int16 

    def record(self, action, user, ai):
        label = np.zeros((1,10))
        label[0, action-1] = 1 #one hot label
        #label[0-3] = one hot action performed

        label[0,4] = user.attack / user.max_attack
        label[0,5] = user.defence / user.max_defence
        label[0,6] = user.health / user.max_health

        label[0,7] = ai.attack / ai.max_attack
        label[0,8] = ai.defence / ai.max_defence
        label[0,9] = ai.health / ai.max_health

        self.data = np.append(self.data, label, axis=0)
        #print(label)
