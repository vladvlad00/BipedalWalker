import torch
import torch.nn as nn
import torch.nn.functional as func

# https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
# https://spinningup.openai.com/en/latest/algorithms/td3.html
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.layers = [
            self.l1,
            self.l2,
            self.l3
        ]

        self.activations = [
            func.relu,
            func.relu,
            lambda x: x
        ]

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)

        return x
