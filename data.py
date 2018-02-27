import numpy as np

class Data:
    def __init__(self, state_length, q_value_length, action_length, reward_length, is_player_1):
        self.state_length = state_length
        self.q_value_length = q_value_length
        self.action_length = action_length
        self.reward_length = reward_length
        self.is_player_1 = is_player_1 #bool

        self.states = np.empty((0, state_length)) #dtype=np.int16
        self.q_values = np.empty((0, q_value_length))
        self.actions = np.empty((0, action_length))
        self.rewards = np.empty((0, reward_length))
    '''
    int action
    Player user
    Player opponent
    bool is_user: is the the user(true) or the ai(false) recording?

    The is_user is there because we need to store the data from the AI's
    perspective.
    '''
    def record(self, state, q_value, action, reward):
        #new_feature = np.zeros((1, self.feature_length))
        new_action = np.zeros((1, self.action_length))
        new_action[0, action - 1] = 1

        self.states = np.append(self.states, state, axis=0)
        self.q_values = np.append(self.q_values, [q_value], axis=0)
        self.actions = np.append(self.actions, new_action, axis=0)
        self.rewards = np.append(self.rewards, np.array([[reward]]), axis=0)
