import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, num_state=16, num_action=4):
        super().__init__()
        self.num_state = num_state
        self.num_action = num_action

        self.linear = nn.Linear(num_state, num_action)

    def observation_embedding(self, x):
        y = torch.eye(self.num_state, device=device)
        return y[x]

    def forward(self, x):
        x = self.observation_embedding(x)
        x_ = self.linear(x)

        e = torch.tensor(np.random.randn(1, self.num_action), device=device,
                               dtype=torch.long)[0]
        x = x_ + e
        _, x = torch.max(x, dim=-1)
        return x, x_

class Trainer:
    def __init__(self, model,
                 criterion = None, optimizer = None):
        self.criterion = nn.MSELoss()
        if criterion is not None: self.criterion = criterion

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        if optimizer is not None: self.optimizer = optimizer


    def train(self, prediction, action, reward):
        target = [p + reward if i == action.item() else p for (i, p) in enumerate(prediction)]
        target = torch.stack(target)

        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()