#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np

from copy import deepcopy

import json

from fl_g13.config import RAW_DATA_DIR, PROJ_ROOT

from fl_g13.modeling import train, load, eval, plot_metrics, get_preprocessing_pipeline

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import per_class_accuracy
from fl_g13.editing import fisher_scores
from fl_g13.editing import create_gradiend_mask, mask_dict_to_list


train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# # Define model to edit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = str(PROJ_ROOT / 'checkpoints')
model_name = 'archeops'
model_checkpoint_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.pth'
model_metrics_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.loss_acc.json'

# Hyper-parameters
# model
head_layers=3
head_hidden_size=512
dropout_rate=0.0
unfreeze_blocks=12

# Dataloaders
BATCH_SIZE = 64

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
print(f'Test accuracy: {100*test_accuracy:.2f}%')


# Plot training results
plot_metrics(path = model_metrics_path)


# # Define model editing

# ## Compute fisher score
# Build a new dataloader with batch size 1 to get more accurate gradient.

fisher_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=True)

# Unfreeze of blocks when computing the fisher score
for param in model.backbone.blocks[-unfreeze_blocks:].parameters():
    param.requires_grad = True
scores = fisher_scores(fisher_dataloader, model)


# ## Create mask

global_mask = create_gradiend_mask(scores, mask_type = 'global')
local_mask = create_gradiend_mask(scores, mask_type = 'local')

global_mask_list = mask_dict_to_list(model, global_mask)
local_mask_list = mask_dict_to_list(model, local_mask)


# # Fine-tune the model

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
        scheduler = None, # No scheduler needed, too few epochs
        verbose = verbose
    )

    # Compute per-class accuracy
    class_acc = per_class_accuracy(test_dataloader, new_model)

    return class_acc


# GLOBAL MASK
global_acc = fine_tune(
    name = f'{model_name}_ft_global',
    mask = global_mask_list,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    train_dataloader = train_dataloader,
    epochs = 3,
    save_every = 3
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
output_file = f"{CHECKPOINT_DIR}/Editing/{model_name}/accuracy_comparison_global.json"

# Save the dictionary to a JSON file
with open(output_file, "w") as json_file:
    json.dump(accuracy_data, json_file, indent=4)
print(f"Accuracy data saved to {output_file}")


# LOCAL MASK
local_acc = fine_tune(
    name = f'{model_name}_ft_local',
    mask = local_mask_list,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    train_dataloader = train_dataloader,
    epochs = 3,
    save_every = 3
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
output_file = f"{CHECKPOINT_DIR}/Editing/{model_name}/accuracy_comparison_local.json"

# Save the dictionary to a JSON file
with open(output_file, "w") as json_file:
    json.dump(accuracy_data, json_file, indent=4)
print(f"Accuracy data saved to {output_file}")

