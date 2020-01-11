import random
from copy import deepcopy as dc
from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from model import DQN


class Agent:
    def __init__(self, size, gamma, n_step, tau, memory_size, epsilon_high, epsilon_low, epsilon_decay, lr, batch_size):
        self.size = size
        self.gamma = gamma
        self.tau = tau
        self.memory_size = memory_size
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = self.epsilon_high
        self.memory = []
        self.steps = 0
        self.n_step = n_step
        self.eval_model = DQN(self.size)
        self.target_model = dc(self.eval_model)
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=self.lr)

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def update_target_network_params(self):
        eval_params = self.eval_model.parameters()
        target_params = self.target_model.parameters()
        tau = self.tau
        for eval_param, target_param in zip(eval_params, target_params):
            target_param.data.copy_(tau * eval_param.data + (1 - tau) * target_param.data)

    def choose_action(self, s):
        self.steps += 1
        self.epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * exp(-self.epsilon_decay * self.steps)
        if torch.rand(1)[0].item() < self.epsilon:
            action = torch.randint(low=0, high=self.size, size=(1,))[0]
        else:
            action_val = self.eval_model(torch.unsqueeze(s, dim=0))[0].detach()
            # detach is not needed because does not flow through arg-max
            action = torch.argmax(action_val)
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.stack(s, dim=0)
        a = torch.stack(a, dim=0).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float)
        s_ = torch.stack(s_, dim=0)
        d = torch.tensor(d, dtype=torch.float)

        q_next = self.gamma ** self.n_step * self.target_model(s_).detach().max(dim=1)[0] * (1 - d) + r
        q_eval = self.eval_model(s).gather(dim=1, index=a).view(-1)
        loss = F.smooth_l1_loss(q_eval, q_next)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network_params()
        return loss.detach().item()
