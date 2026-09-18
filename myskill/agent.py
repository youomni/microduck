import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=2, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, obs_dim=2, action_dim=4, lr=1e-3):
        self.model = PolicyNet(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.model(obs)
        probs = torch.softmax(logits, dim=-1)

        action = torch.multinomial(probs, 1).item()
        return action, probs.detach().numpy()

    def learn(self, log_probs, rewards):
        """
        Very simple REINFORCE-style update
        """
        loss = 0
        G = 0

        for log_p, r in zip(reversed(log_probs), reversed(rewards)):
            G = r + 0.99 * G
            loss -= log_p * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
