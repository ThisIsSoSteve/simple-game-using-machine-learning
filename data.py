import numpy as np

class Data:
    def __init__(self, feature_length, label_length):
        self.feature_length = feature_length
        self.label_length = label_length
        self.feature = np.empty((0, feature_length)) #dtype=np.int16
        self.label = np.empty((0, label_length))

    '''
    int action
    Player user
    Player opponent
    bool is_user: is the the user(true) or the ai(false) recording?

    The is_user is there because we need to store the data from the AI's
    perspective.
    '''
    def record(self, action, user, opponent, is_user):
        new_feature = np.zeros((1, self.feature_length))
        new_label = np.zeros((1, self.label_length))

        new_label[0, action - 1] = 1

        if is_user:
            new_feature[0, 0] = opponent.attack / opponent.max_attack
            new_feature[0, 1] = opponent.defence / opponent.max_defence
            new_feature[0, 2] = opponent.health / opponent.max_health

            new_feature[0, 3] = user.attack / user.max_attack
            new_feature[0, 4] = user.defence / user.max_defence
            new_feature[0, 5] = user.health / user.max_health
        else:
            new_feature[0, 0] = user.attack / user.max_attack
            new_feature[0, 1] = user.defence / user.max_defence
            new_feature[0, 2] = user.health / user.max_health

            new_feature[0, 3] = opponent.attack / opponent.max_attack
            new_feature[0, 4] = opponent.defence / opponent.max_defence
            new_feature[0, 5] = opponent.health / opponent.max_health

        self.feature = np.append(self.feature, new_feature, axis=0)
        self.label = np.append(self.label, new_label, axis=0)
