import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, num_state=16, num_action=4, hidden_dim=32):
        super().__init__()
        self.num_state = num_state
        self.num_action = num_action

        self.linear1 = nn.Linear(num_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_action)

    def observation_embedding(self, x):
        y = torch.eye(self.num_state, device=device)
        return y[x]

    def forward(self, x):
        x = self.observation_embedding(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def explore(prob, action, num_action, epsilon=0.3):
  if prob < epsilon:
    return torch.randint(0, num_action, (1,))
  else:
    return action

class Trainer:
    def __init__(self, model,
                 criterion = None, optimizer = None):
        self.model = model

        self.criterion = nn.MSELoss()
        if criterion is not None: self.criterion = criterion

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        if optimizer is not None: self.optimizer = optimizer


    def train(self, pred, target, reward, y=0.99):
        self.model.train()

        value, action = torch.max(target, dim=-1)
        target[action] = reward + y * value

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()