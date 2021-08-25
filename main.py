import numpy as np
import random
import math
import gym.spaces
from itertools import count
from collections import namedtuple
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.dqn import DQN, image_embedding
from model.dqn import ReplayBuffer, select_epilson_greedy_action
from util.gym import get_env
import torch
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore")

# hyperparameters
N_EPISODES = 400
N_FRAMES = 4
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
TARGET_UPDATE = 1000
GAMMA = 0.99 # Q-value
MEMORY_SIZE = 100_000

def get_epsilone(steps_done, EPS_START=1, EPS_END=1000000, EPS_DECAY=1000000):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY), steps_done+1

# make env
env_name = "SpaceInvadersNoFrameskip-v0"
env = gym.make(env_name)
env = get_env(env, N_FRAMES)

# build model
img_h, img_w, img_c = env.observation_space.shape
in_channels = img_c
num_actions = env.action_space.n

policy = DQN(in_channels=in_channels, num_actions=num_actions).to(device)
target = DQN(in_channels=in_channels, num_actions=num_actions).to(device)
target.load_state_dict(policy.state_dict())

criterion = F.smooth_l1_loss
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# replay buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))
memory = ReplayBuffer(MEMORY_SIZE)

# training
episode_counts, episode_reward = [], []
episode_count, steps_done = 0, 0
last_obs = env.reset()
for episode in tqdm(range(N_EPISODES)):
    obs = env.reset()
    state = image_embedding(obs)
    total_reward = 0.0
    for t in count():
        EPSILON, steps_done = get_epsilone(steps_done)
        action = select_epilson_greedy_action(policy, state, EPSILON)
        obs, reward, done, info = env.step(action)

        if done:
            next_state = None
        else:
            next_state = image_embedding(obs)
        total_reward += reward

        reward = torch.tensor([reward], device=device)
        memory.push(state, action.cpu(), next_state, reward.cpu())
        state = next_state

        # no warm start
        if len(memory) < BATCH_SIZE:
            continue
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states   = torch.cat(batch.state).to(device)
        actions  = torch.cat(tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))))
        rewards  = torch.cat(tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))))
        dones    = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
        n_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        # compute Q-values
        cur_Q = policy(states).gather(1, actions)
        nxt_Q = torch.zeros(BATCH_SIZE, device=device)
        nxt_Q[dones] = target(n_states).max(1)[0].detach()
        exp_Q = (nxt_Q * GAMMA) + rewards
        loss = F.smooth_l1_loss(cur_Q, nxt_Q.unsqueeze(1))
        
        # step optimizer
        optimizer.zero_grad()
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if steps_done % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        if done:
            break
    episode_count += 1
    episode_counts.append(episode_count)
    episode_reward.append(total_reward)
env.close()

# save figure
torch.save(policy.state_dict(), f'./result/policy.pt')
plt.figure(figsize=(15, 15))
plt.title('reward')
plt.plot(episode_counts, episode_reward, 'r')
plt.savefig('./result/metrics.png')

