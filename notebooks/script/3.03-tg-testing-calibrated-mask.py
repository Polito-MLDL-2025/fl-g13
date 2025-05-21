#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np

import json

from fl_g13.config import RAW_DATA_DIR, PROJ_ROOT

from fl_g13.modeling import train, load, plot_metrics, get_preprocessing_pipeline

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import per_class_accuracy
from fl_g13.editing import create_mask, mask_dict_to_list

train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints'
model_name = 'arcanine'
model_checkpoint_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.pth'
model_metrics_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.loss_acc.json'

# Hyper-parameters
# model
head_layers=3
head_hidden_size=512
dropout_rate=0.0
unfreeze_blocks=1

# Dataloaders
BATCH_SIZE = 128

# SparseSGDM optimizer
LR = 1e-3
momentum = .9
weight_decay = 1e-5

# scheduler
T_max = 8
eta_min = 1e-5

# Empty model
# Will be replaced with the already trained model from the checkpoint
model = BaseDino(
    head_layers=head_layers,
    head_hidden_size=head_hidden_size,
    dropout_rate=dropout_rate,
    unfreeze_blocks=unfreeze_blocks
)
model.to(device)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Create a dummy mask for SparseSGDM
mask = [torch.ones_like(p, device = p.device) for p in model.parameters()] # Must be done AFTER the model is moved to the device
# Optimizer, scheduler, and loss function
optimizer = SparseSGDM(
    model.parameters(),
    mask = mask,
    lr = LR,
    momentum = momentum,
    weight_decay = weight_decay
)
scheduler = CosineAnnealingLR(
    optimizer = optimizer,
    T_max = T_max,
    eta_min = eta_min
)
criterion = CrossEntropyLoss()

# Load the model
model, _ = load(
    path = model_checkpoint_path,
    model_class = BaseDino,
    optimizer = optimizer,
    scheduler = scheduler,
    device = device
)
model.to(device) # manually move the model to the device

print(f'\nModel {model_name} loaded from checkpoint.')

# Compute test accuracy
# test_loss, test_accuracy, _ = eval(test_dataloader, model, criterion)
class_acc = per_class_accuracy(test_dataloader, model)
test_accuracy = np.mean(class_acc)

# print(f'Test loss: {test_loss:.3f}')
print(f'\nTest accuracy: {100*test_accuracy:.2f}%')

fisher_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=True)

density = 0.2

global_calibr_mask = create_mask(fisher_dataloader, model, density = density, mask_type = 'global', rounds = 5)
local_calibr_mask = create_mask(fisher_dataloader, model, density = density, mask_type = 'local', rounds = 5)

def fine_tune(name, train_dataloader, mask, optimizer, scheduler, criterion, epochs = 10, verbose = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    new_model, _ = load(
        path = model_checkpoint_path,
        model_class = BaseDino,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device
    )
    new_model.to(device) # manually move the model to the device

    # Create a new SparseSGDM optimizer
    new_optimizer = SparseSGDM(
        new_model.parameters(),
        mask = mask,
        lr = LR,
        momentum = momentum,
        weight_decay = weight_decay
    )

    _, _, _, _ = train(
        checkpoint_dir = CHECKPOINT_DIR,
        name = name,
        start_epoch = 1,
        num_epochs = epochs,
        save_every = epochs,
        backup_every = None,
        train_dataloader = train_dataloader,
        val_dataloader = None,
        model = new_model,
        criterion = criterion,
        optimizer = new_optimizer,
        scheduler = scheduler,
        verbose = verbose
    )

    # Compute per-class accuracy
    class_acc = per_class_accuracy(test_dataloader, new_model)

    return class_acc

# GLOBAL MASK
global_acc = fine_tune(
    name = f'{model_name}_ft_global_calibr',
    mask = mask_dict_to_list(model, global_calibr_mask),
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    train_dataloader = train_dataloader
)

new_test_accuracy = np.mean(global_acc)
print(f'\nTest accuracy: {100*new_test_accuracy:.2f}% (original: {100*test_accuracy:.2f}%)')

count = sum([1 for i in range(len(global_acc)) if global_acc[i] < class_acc[i]])
print(f'Fine-tuned model is worse in {count} classes, wrt the original model')
# Save to file the per-class accuracy difference
# Create a dictionary with new_class_acc, class_acc, and class_idx
accuracy_data = {
    "class_idx": list(range(100)),
    "new_class_acc": list(global_acc),
    "class_acc": list(class_acc)
}
output_file = f"{CHECKPOINT_DIR}/Editing/{model_name}/accuracy_comparison_global_calibr.json"

# Save the dictionary to a JSON file
with open(output_file, "w") as json_file:
    json.dump(accuracy_data, json_file, indent=4)
print(f"Accuracy data saved to {output_file}")

# LOCAL MASK
local_acc = fine_tune(
    name = f'{model_name}_ft_local_calibr',
    mask = mask_dict_to_list(model, local_calibr_mask),
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    train_dataloader = train_dataloader
)

new_test_accuracy = np.mean(local_acc)
print(f'\nTest accuracy: {100*new_test_accuracy:.2f}% (original: {100*test_accuracy:.2f}%)')

count = sum([1 for i in range(len(local_acc)) if local_acc[i] < class_acc[i]])
print(f'Fine-tuned model is worse in {count} classes, wrt the original model')
# Save to file the per-class accuracy difference
# Create a dictionary with new_class_acc, class_acc, and class_idx
accuracy_data = {
    "class_idx": list(range(100)),
    "new_class_acc": list(local_acc),
    "class_acc": list(class_acc)
}
output_file = f"{CHECKPOINT_DIR}/Editing/{model_name}/accuracy_comparison_local_calibr.json"

# Save the dictionary to a JSON file
with open(output_file, "w") as json_file:
    json.dump(accuracy_data, json_file, indent=4)
print(f"Accuracy data saved to {output_file}")
