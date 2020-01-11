import torch
from copy import deepcopy as dc


class Environment:
    def __init__(self, size, rewards):
        self.size = size
        self.rewards = rewards

    def reset(self):
        state = torch.randint(low=0, high=2, size=(self.size,), dtype=torch.float)
        target = torch.randint(low=0, high=2, size=(self.size,), dtype=torch.float)
        while (state - target).abs().sum().numpy() == 0:
            target = torch.randint(low=0, high=2, size=(self.size,), dtype=torch.float)
        done = False
        return torch.cat([state, target], dim=-1), done

    def step(self, s, action):
        s_ = dc(s)
        s_[action] = 1 - s_[action]
        dist = (s_[:self.size] - s_[self.size:]).abs().sum().numpy()
        if dist == 0:
            r = self.rewards[0]
            done = True
        else:
            r = self.rewards[1]
            done = False
        return s_, r, done
