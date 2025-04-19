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


train_dataloader = DataLoader(cifar100_train) # Inline training (batch_size=1)
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

from fl_g13.architectures import BaseDino

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BaseDino()
model.to(device)


model.get_config()


CHECKPOINT_DIR = "/home/massimiliano/Projects/fl-g13/checkpoints"
LR = 0.001

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=LR)
# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) # Numb of epochs non relevant
# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Preallocated lists: if the training interrupts, it will still save their values
all_training_losses=[]       # Pre-allocated list for training losses
all_validation_losses=[]     # Pre-allocated list for validation losses
all_training_accuracies=[]   # Pre-allocated list for training accuracies
all_validation_accuracies=[] # Pre-allocated list for validation accuracies


# Train the model and save periodically
# NOTE: If a checkpoint with the exact same model name, 
# model class and epoch number exists, it will be overwritten!!!
_, _, _, _ = train(
    checkpoint_dir=CHECKPOINT_DIR,
    name="arceus", # If empty, will automatically find a name for the model
    start_epoch=1, # Try one epoch
    num_epochs=1, # Try one epoch
    save_every=1, # Try one epoch
    backup_every=None,
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


# Load the model from a checkpoint
model_class = BaseDino
# NOTE: If you do not specify a file name, it will automatically find the latest checkpoint
checkpoint_path = f"{CHECKPOINT_DIR}/{model_class.__name__}/arceus_{model_class.__name__}_epoch_1.pth"

# Pass device also here to directly load the state dict (not the model itself!) on the device
model, start_epoch = load(path=checkpoint_path, model_class=model_class, verbose=True)
model.to(device)  # Move model to the device, for real this time!


# Resume training
_, _, _, _ = train(
    checkpoint_dir=CHECKPOINT_DIR,
    name="arceus", # Use the same name, or just a different one if you are afraid of overwriting!
    start_epoch=start_epoch, # Resume from the correct epoch
    num_epochs=3, 
    save_every=1,
    backup_every=None,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    model=model, # Use the same model as before (partially pre-trained)
    criterion=criterion,
    optimizer=optimizer,  # I could also change the optimizer if I wanted to!
    scheduler=scheduler,
    verbose=False,
    all_training_losses=all_training_losses,  # Will not overwrite the original, but just append
    all_validation_losses=all_validation_losses,
    all_training_accuracies=all_training_accuracies,
    all_validation_accuracies=all_validation_accuracies,
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




