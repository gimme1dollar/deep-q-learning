import numpy as np
import random

class QLearning():
    def __init__(self, num_state=16, num_action=4,
                 exploration_rate = 0.2, learning_rate=0.95, discount_factor=0.8):
        super().__init__()
        self.num_state = num_state
        self.num_action = num_action

        self.table = np.zeros([num_state, num_action])
        self.e = exploration_rate
        self.a = learning_rate
        self.r = discount_factor

    def __call__(self, state, epoch):
        exploit = self.table[state, :]
        explore = np.random.randn(1, self.num_action) / (epoch+1)

        return np.argmax(exploit + explore)

    def update(self, prev_state, action, curr_state, reward):
        self.table[prev_state, action] \
            = (1-self.a) * self.table[prev_state, action] + \
              self.a * (reward + self.r * np.max(self.table[curr_state, :]))