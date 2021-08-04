import torch
import torch.nn as nn
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
        x = self.linear(x)

        return x