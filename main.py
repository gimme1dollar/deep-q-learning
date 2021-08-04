import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model.q_learning import QLearning

env = gym.make('FrozenLake-v0')
policy = QLearning(num_state=env.observation_space.n, num_action=env.action_space.n)

# Set learning parameters
num_epoch = 200000
episode_reward, episode_length = [], []
observation = env.reset()
env.render()
for epoch in range(num_epoch):
    state = env.reset()

    episode_reward.append(0.0)
    episode_length.append(0)

    done = False
    while episode_length[-1] < 100:
        action = policy(state, epoch)

        state_, reward, done, info = env.step(action)

        episode_reward[-1] += reward
        episode_length[-1] += 1

        policy.update(state, action, state_, reward)
        state = state_
        if done:
            print(f"[{epoch:03}] episode_reward: {episode_reward[-1]}, episode_length: {episode_length[-1]}")
            break
print(f"*** mean reward: {sum(episode_reward) / len(episode_reward)}, mean length: {sum(episode_length) / len(episode_reward)}")
print ("Final Q-Table Values")
print (np.round(policy.table,3))
env.close()