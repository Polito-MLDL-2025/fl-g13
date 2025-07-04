from typing import Type
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim, device, cuda, nn, ones_like
from fl_g13.modeling.load import load_or_create, get_model
from fl_g13.editing import SparseSGDM
from fl_g13.architectures import BaseDino

class TinyCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [B, 16, 32, 32]
        x = F.max_pool2d(x, 2)  # -> [B, 16, 16, 16]
        x = F.relu(self.conv2(x))  # -> [B, 32, 16, 16]
        x = F.max_pool2d(x, 2)  # -> [B, 32, 8, 8]
        x = x.view(x.size(0), -1)  # -> [B, 32*8*8]
        x = self.fc1(x)  # -> [B, 100]
        return x