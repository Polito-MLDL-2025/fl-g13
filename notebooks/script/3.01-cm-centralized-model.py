#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR
from fl_g13.modeling import train, eval, save, load

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader


# ## Boilerplate usecase of Dino

from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor

import requests
from IPython.display import display
from PIL import Image

# Load the pretrained DINO ViT-S/16 model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Download a sample image from ImageNet
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
response = requests.get(url, stream=True)
img = Image.open(response.raw).convert("RGB")
display(img.resize((128, 128)))

imagenet_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_tensor = imagenet_transform(img).unsqueeze(0)  # Add batch dimension

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Load ImageNet class labels
imagenet_classes_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_classes = requests.get(imagenet_classes_url).json()

# Get the predicted label string
predicted_label = output.argmax(dim=1).item()
predicted_label_str = imagenet_classes[predicted_label]
print(f"Predicted label: {predicted_label_str} ({predicted_label})")


# ## Load data

from torchvision import models
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor

# Define preprocessing pipeline
train_transform = Compose([
    Resize(256), # CIFRA100 is originally 32x32
    RandomCrop(224), # But Dino works on 224x224
    RandomHorizontalFlip(),
    #RandomVerticalFlip(), # Dino was not pretrained with Vertical flip, lets avoid
    ToTensor(),
    Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

eval_transform = Compose([
    Resize(256), # CIFRA100 is originally 32x32
    CenterCrop(224), # But Dino works on 224x224
    ToTensor(),
    Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=train_transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=eval_transform)


train_dataloader = DataLoader(cifar100_train)
test_dataloader = DataLoader(cifar100_test)


# Uncomment to extract mean and var
# WARNING: DO NOT RUN IF YOU APPLY OTHER TRANSOFRMATIONS THAN ToTensor()

# # Stack all images into a single tensor
# all_images = torch.cat([cifar100_train[i][0].unsqueeze(0) for i in range(len(cifar100_train))], dim=0)

# print(all_images.shape)

# # Calculate mean and std for each channel (RGB)
# mean = all_images.mean(dim=(0, 2, 3))  # Mean across batch, height, and width
# std = all_images.std(dim=(0, 2, 3))    # Std across batch, height, and width

# print(f"Mean: {mean}")
# print(f"Std: {std}")


# ## Train Model

from timm.models.layers import DropPath

# Load model from torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {device}")

dropout_rate = 0.1      # Dropout rate for MLP and attention layers
drop_path_rate = 0.1    # DropPath rate for stochastic depth

# Eventually TODO for avoiding overfitting: provide dropout, drop_path within the ViT, and dropout in the head
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model.head = nn.Sequential(
    nn.Linear(384, 1024),
    nn.ReLU(),
    nn.Dropout(p=dropout_rate),  # Add dropout
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=dropout_rate),  # Add dropout
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=dropout_rate),  # Add dropout
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=dropout_rate),  # Add dropout
    nn.Linear(1024, 100),
)
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

model.head.apply(initialize_weights)

# Add dropout to attention layers
for block in model.blocks:
    block.attn.attn_drop = nn.Dropout(p=dropout_rate)  # Dropout in attention
    block.attn.proj_drop = nn.Dropout(p=dropout_rate)  # Dropout in projection
    block.mlp.drop = nn.Dropout(p=dropout_rate)  # Dropout in MLP

# Add DropPath to transformer blocks
for i, block in enumerate(model.blocks):
    drop_prob = drop_path_rate * (i / len(model.blocks))  # Linearly scale drop rate
    block.drop_path = DropPath(drop_prob)  # Replace Identity with DropPath

# Freeze whole model
for param in model.parameters():
    param.requires_grad = False

# Allow to access some of the blocks in the backbone
for param in model.blocks[-3:].parameters():
    param.requires_grad = True

# Also allow the LayerNorm
for param in model.norm.parameters():
    param.requires_grad = True

# And obviously the head (MLP)
for param in model.head.parameters():
    param.requires_grad = True

model.to(device)


CHECKPOINT_DIR = "/home/massimiliano/Projects/fl-g13/checkpoints"

# Parameters
batch_size = 128
start_epoch = 1
num_epochs = 5
save_every = 1

# Optimizer and loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = torch.nn.CrossEntropyLoss()

all_training_losses=[]       # Pre-allocated list for training losses
all_validation_losses=[]     # Pre-allocated list for validation losses
all_training_accuracies=[]   # Pre-allocated list for training accuracies
all_validation_accuracies=[] # Pre-allocated list for validation accuracies

# Train the model and save periodically
train(
    checkpoint_dir=CHECKPOINT_DIR,
    prefix="", # Automatically find a name for the model
    start_epoch=start_epoch,
    num_epochs=num_epochs,
    save_every=save_every,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    model=model, # Use the same model as before (partially pre-trained)
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=False,
    all_training_losses=all_training_losses,  # Pre-allocated list for training losses
    all_validation_losses=all_validation_losses,  # Pre-allocated list for validation losses
    all_training_accuracies=all_training_accuracies,  # Pre-allocated list for training accuracies
    all_validation_accuracies=all_validation_accuracies,  # Pre-allocated list for validation accuracies
)


import matplotlib.pyplot as plt

# Plot losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(all_training_losses, label='Training Loss')
plt.plot(all_validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(all_training_accuracies, label='Training Accuracy')
plt.plot(all_validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

plt.tight_layout()
plt.show()




