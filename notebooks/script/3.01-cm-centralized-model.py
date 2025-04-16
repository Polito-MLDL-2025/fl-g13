#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fl_g13.config import RAW_DATA_DIR
from fl_g13.modeling import train, test, save, load

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ## Load data

transform = transforms.Compose([
    transforms.ToTensor(),
])

cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)


train_dataloader = DataLoader(cifar100_train)
test_dataloader = DataLoader(cifar100_test)


# Stack all images into a single tensor
all_images = torch.cat([cifar100_train[i][0].unsqueeze(0) for i in range(len(cifar100_train))], dim=0)

print(all_images.shape)

# Calculate mean and std for each channel (RGB)
mean = all_images.mean(dim=(0, 2, 3))  # Mean across batch, height, and width
std = all_images.std(dim=(0, 2, 3))    # Std across batch, height, and width

print(f"Mean: {mean}")
print(f"Std: {std}")


# ## Train Model

from torchvision import models
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize
from PIL import Image
import requests
import torch

from IPython.display import display

# Load the pretrained DINO ViT-S/16 model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define preprocessing pipeline
transform = Compose([
    Resize((256, 256)),
    RandomCrop((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

# Download a sample image from ImageNet
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
response = requests.get(url, stream=True)
img = Image.open(response.raw).convert("RGB")
display(img.resize((128, 128)))

# Preprocess the image
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Move to device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted label
predicted_label = output.argmax(dim=1).item()
print("Predicted label:", predicted_label)

# Get the predicted label
# Load ImageNet class labels
imagenet_classes_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_classes = requests.get(imagenet_classes_url).json()

# Get the predicted label string
predicted_label = output.argmax(dim=1).item()
predicted_label_str = imagenet_classes[predicted_label]
print("Predicted label:", predicted_label_str)


# Load model from torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model.head = nn.Sequential(
    Linear(384, 1024),
    nn.ReLU(),
    Linear(1024, 1024),
    nn.ReLU(),
    Linear(1024, 1024),
    nn.ReLU(),
    Linear(1024, 1024),
    nn.ReLU(),
    Linear(1024, 100),
)
def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

model.head.apply(initialize_weights)

# Freeze whole model
for param in model.parameters():
    param.requires_grad = False

# Allow to access some of the blocks
for param in model.blocks[-4:].parameters():
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
num_epochs = 50
save_every = 5

# Optimizer and loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = torch.nn.CrossEntropyLoss()

# Train
train(
    checkpoint_dir=CHECKPOINT_DIR,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    criterion=criterion,
    start_epoch=start_epoch,
    num_epochs=num_epochs,
    save_every=save_every,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    prefix=None,
    verbose=False,
)




