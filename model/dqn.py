import random
import numpy as np
from collections import namedtuple


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=4):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super().__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.num_actions)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, 0, 0.1)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return x

def image_embedding(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def select_epilson_greedy_action(model, obs, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        obs = torch.tensor(obs, dtype=torch.float, device=device)[None, ...] / 255.0
        with torch.no_grad():
            prediction = model(obs)
            action = prediction.data.max(1)[1].view(1,1)
            return action
    else:
        return torch.tensor([[random.randrange(model.num_actions)]], device=device, dtype=torch.long)
