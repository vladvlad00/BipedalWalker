import torch
import torch.nn as nn
import torch.nn.functional as func


# https://spinningup.openai.com/en/latest/algorithms/td3.html
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.layers = [
            self.l1,
            self.l2,
            self.l3
        ]

        self.activations = [
            func.relu,
            func.relu,
            torch.tanh
        ]

        self.max_action = max_action

    def forward(self, state):
        for i in range(len(self.layers)):
            state = self.activations[i](self.layers[i](state))
        return state * self.max_action
