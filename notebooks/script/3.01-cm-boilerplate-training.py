#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR
from torchvision import datasets, transforms

from fl_g13.modeling import train
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


# This will train the model, using an automatically generated name
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


# This will train again the same model, but with a custom name
# The epochs will still start from 1
train(
    checkpoint_dir=checkpoint_dir,
    prefix="TinyCNN", # Setting a name for the model
    start_epoch=start_epoch,
    num_epochs=num_epochs,
    save_every=save_every,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    model=model, # Use the same model as before (partially pre-trained)
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    verbose=False,
)


# ### Resume training

from fl_g13.modeling import load

# Generate untrained model and optimizer
model2 = TinyCNN(num_classes=100)
optimizer2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
criterion2 = torch.nn.CrossEntropyLoss()

# Load the model from the latest checkpoint (a specific file!)
path = checkpoint_dir + "/TinyCNN_epoch_2.pth"
start_epoch = load(path=path, model=model2, optimizer=optimizer2, scheduler=None)


num_epochs = 4
save_every = 2

# Now we can continue training the model (model2 now!) from the last checkpoint (which has been loaded)
# Note again: loading is done by the user in the cell above! It is not done automatically!
# The start_epoch is now 2 (value returned by the load function)
train(
    checkpoint_dir=checkpoint_dir,
    prefix="TinyCNN", # Use the same name as before to continue training!
    start_epoch=start_epoch, # Now start epoch is not 1 (will resume from where it was left)
    num_epochs=num_epochs, # This is not the number of epochs to reach, but how many to do starting from now!
    save_every=save_every,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    model=model2,
    criterion=criterion2,
    optimizer=optimizer2,
    scheduler=None,
    verbose=False,
)




