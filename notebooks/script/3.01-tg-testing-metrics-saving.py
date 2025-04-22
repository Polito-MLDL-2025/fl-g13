#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR, PROJ_ROOT
from torchvision import datasets, transforms

from fl_g13.modeling import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# # Test NN

transform = transforms.Compose([
    transforms.ToTensor()
])
cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)


class TinyCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

        # Store configuration for later loading
        # This is a bit of a hack, but we need to store the number of classes
        self._config = {
            "num_classes": num_classes,
            # In fact, the followings could be avoided as from_config loads only the num_classes
            "conv1_out_channels": 16,
            "conv2_out_channels": 32,
            "fc1_in_features": 32 * 8 * 8,
        }

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> [B, 16, 32, 32]
        x = F.max_pool2d(x, 2)        # -> [B, 16, 16, 16]
        x = F.relu(self.conv2(x))     # -> [B, 32, 16, 16]
        x = F.max_pool2d(x, 2)        # -> [B, 32, 8, 8]
        x = x.view(x.size(0), -1)     # -> [B, 32*8*8]
        x = self.fc1(x)               # -> [B, 100]
        return x

    # Now we need to be careful to define how to load from config
    @classmethod
    def from_config(cls, config):
        return cls(num_classes=config["num_classes"])


CHECKPOINT_DIR = str(PROJ_ROOT / "checkpoints")

# Parameters
batch_size  = 32
start_epoch = 1
num_epochs  = 6
save_every  = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True)

model = TinyCNN(100)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
criterion = torch.nn.CrossEntropyLoss()


_, _, _, _ = train(
    checkpoint_dir = CHECKPOINT_DIR,
    name = "first_train",
    train_dataloader = train_dataloader,
    val_dataloader = test_dataloader,
    criterion = criterion,
    start_epoch = start_epoch,
    num_epochs = num_epochs,
    save_every = save_every,
    backup_every = None,
    model = model,
    optimizer = optimizer,
    scheduler = None,
    verbose = False,
)


from fl_g13.modeling.load import load_loss_and_accuracies, plot_metrics

plot_metrics(path = CHECKPOINT_DIR + '\\first_train_TinyCNN_epoch_6.loss_acc.json')


metrics = load_loss_and_accuracies(path = CHECKPOINT_DIR + '\\first_train_TinyCNN_epoch_6.loss_acc.json', verbose=True)


from fl_g13.modeling.load import load

loaded_model, resume_epoch = load(
    path = CHECKPOINT_DIR + '\\first_train_TinyCNN_epoch_6.json',
    model_class = TinyCNN,
    device=device,
    optimizer=optimizer,
    verbose=True
)


# Resume training

_, _, _, _ = train(
    checkpoint_dir = CHECKPOINT_DIR,
    name = "second_train",
    train_dataloader = train_dataloader,
    val_dataloader = test_dataloader,
    criterion = criterion,
    start_epoch = resume_epoch,
    num_epochs = 6,
    save_every = 2,
    backup_every = None,
    model = loaded_model,
    optimizer = optimizer,
    scheduler = None,
    verbose = False,
)


plot_metrics(path = CHECKPOINT_DIR + '\\second_train_TinyCNN_epoch_12.loss_acc.json')


new_metrics = load_loss_and_accuracies(path = CHECKPOINT_DIR + '\\second_train_TinyCNN_epoch_12.loss_acc.json')


full_metric = dict()

for k in metrics.keys():
    full_metric[k] = metrics[k] + new_metrics[k]


import matplotlib.pyplot as plt

train_epochs = full_metric['train_epochs']
val_epochs = full_metric.get('val_epochs', [])

# Crea i subplot affiancati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot Loss ---
ax1.plot(train_epochs, full_metric['train_loss'], label='Train Loss', color='tab:blue')
if full_metric['val_loss']:
    ax1.plot(val_epochs, full_metric['val_loss'], label='Val Loss', color='tab:orange')
ax1.set_title('Loss over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# --- Plot Accuracy ---
ax2.plot(train_epochs, full_metric['train_acc'], label='Train Accuracy', color='tab:green')
if full_metric['val_acc']:
    ax2.plot(val_epochs, full_metric['val_acc'], label='Val Accuracy', color='tab:red')
ax2.set_title('Accuracy over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

