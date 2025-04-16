#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR
from torchvision import datasets, transforms

from fl_g13.modeling import train, train_one_epoch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ### Load data

transform = transforms.Compose([
    transforms.ToTensor()
])
cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)


# ### Train and save model

class TinyCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> [B, 16, 32, 32]
        x = F.max_pool2d(x, 2)        # -> [B, 16, 16, 16]
        x = F.relu(self.conv2(x))     # -> [B, 32, 16, 16]
        x = F.max_pool2d(x, 2)        # -> [B, 32, 8, 8]
        x = x.view(x.size(0), -1)     # -> [B, 32*8*8]
        x = self.fc1(x)               # -> [B, 100]
        return x


checkpoint_dir = "/home/massimiliano/Projects/fl-g13/checkpoints"

# Parameters
batch_size  = 32
start_epoch = 1
num_epochs  = 2
save_every  = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True)

model = TinyCNN(100)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)

loss_fn = torch.nn.CrossEntropyLoss()


train(checkpoint_dir, train_dataloader, loss_fn, start_epoch, num_epochs, save_every, model, optimizer, scheduler=None, prefix=None, verbose=False)


train(checkpoint_dir, train_dataloader, loss_fn, start_epoch, num_epochs, save_every, model, optimizer, scheduler=None, prefix="TinyCNN", verbose=False)


# **Resume training**

from fl_g13.modeling import load

# Load the model from the latest checkpoint
model2 = TinyCNN(num_classes=100)
optimizer2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
loss_fn2 = torch.nn.CrossEntropyLoss()

start_epoch = load(checkpoint_dir, model=model2, optimizer=optimizer2, filename="TinyCNN_epoch_3.pth")


num_epochs = 4
save_every = 2

train(checkpoint_dir, train_dataloader, loss_fn2, start_epoch, num_epochs, save_every, model2, optimizer2, scheduler=None, prefix="TinyCNN", verbose=False)




