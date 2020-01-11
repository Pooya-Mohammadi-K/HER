from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, size):
        super(DQN, self).__init__()
        self.size = size

        self.fc_1 = nn.Linear(self.size * 2, 128)
        self.output = nn.Linear(128, self.size)

    def forward(self, inputs):
        x = F.relu(self.fc_1(inputs))
        x = self.output(x)
        return x
