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
criterion = torch.nn.CrossEntropyLoss()


train(
    checkpoint_dir=checkpoint_dir,
    prefix="", # Will automatically generate a name for the model
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    criterion=criterion,
    start_epoch=start_epoch,
    num_epochs=num_epochs,
    save_every=save_every,
    model=model,
    optimizer=optimizer,
    scheduler=None,
    verbose=False,
)


train(
    checkpoint_dir=checkpoint_dir,
    prefix="TinyCNN", # Setting a name for the model
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    criterion=criterion,
    start_epoch=start_epoch,
    num_epochs=num_epochs,
    save_every=save_every,
    model=model, # Use the same model as before (partially pre-trained)
    optimizer=optimizer,
    scheduler=None,
    verbose=False,
)


# **Resume training**

from fl_g13.modeling import load

# Generate untrained objects
model2 = TinyCNN(num_classes=100)
optimizer2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
criterion2 = torch.nn.CrossEntropyLoss()

# Load the model from the latest checkpoint
path = checkpoint_dir + "/TinyCNN_epoch_2.pth"
start_epoch = load(path=path, model=model2, optimizer=optimizer2, scheduler=None)


num_epochs = 4
save_every = 2

train(
    checkpoint_dir=checkpoint_dir,
    prefix="TinyCNN", # Use the same name as before to continue training!
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    criterion=criterion2,
    start_epoch=start_epoch, # Now start epoch is not 1 (will resume from where it was left)
    num_epochs=num_epochs, # This is not the number of epochs to reach, but how many to do starting from now!
    save_every=save_every,
    model=model2,
    optimizer=optimizer2,
    scheduler=None,
    verbose=False,
)




