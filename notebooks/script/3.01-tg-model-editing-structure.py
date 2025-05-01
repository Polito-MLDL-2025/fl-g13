#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np

from fl_g13.config import RAW_DATA_DIR, PROJ_ROOT

from fl_g13.modeling import train, load, eval, plot_metrics, get_preprocessing_pipeline

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import per_class_accuracy, get_worst_classes, build_per_class_dataloaders
from fl_g13.editing import fisher_scores
from fl_g13.editing import create_gradiend_mask, mask_dict_to_list


train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# # Define the model to edit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = str(PROJ_ROOT / 'checkpoints')
model_name = 'yamask'
model_checkpoint_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.pth'
model_metrics_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.loss_acc.json'

# Empty model
# Will be replaced with the already trained model from the checkpoint
model = BaseDino(head_layers=5, head_hidden_size=512, dropout_rate=0.0, unfreeze_blocks=1)
model.to(device)

# Hyper-parameters
BATCH_SIZE = 128
LR = 1e-3

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Create a dummy mask for SparseSGDM
mask = [torch.ones_like(p, device = p.device) for p in model.parameters()] # Must be done AFTER the model is moved to the device
# Optimizer, scheduler, and loss function
optimizer = SparseSGDM(model.parameters(), mask = mask, lr = LR)
scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = 20, eta_min = 1e-5)
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
test_loss, test_accuracy, _ = eval(test_dataloader, model, criterion)

print(f'Test loss: {test_loss:.3f}')
print(f'Test accuracy: {100*test_accuracy:.2f}%')

# Plot training results
plot_metrics(path = model_metrics_path)


# # Model editing

# ## Compute per-class accuracy
# Find the class in which the model is underperforming

class_acc = per_class_accuracy(test_dataloader, model)
print(f'\nClass accuracy (first 10 classes): {class_acc[:10]}') # Output preview


N_worst = 3 # How many classes to fine-tune
worst_classes = get_worst_classes(class_acc, N_worst)
print(f"Worst classes: {worst_classes}")


# Note that the batch size in this case is 32 by default
# Since the dataloaders are specific to the classes, a smaller batch size is better
classes_dataloaders = build_per_class_dataloaders(train_dataset, worst_classes)


# ## Compute Fisher Sentitivity (per-class)

def compute_score_per_classes(model, classes, classes_dataloaders):
    score_per_class = {}

    for cls in classes:
        print(f"Computing scores for class {cls}")
        scores = fisher_scores(classes_dataloaders[cls], model)
        # Ensure the scores dictionary has entries for all model parameters
        for name, param in model.named_parameters():
            if name not in scores:
                scores[name] = torch.ones_like(param)
        score_per_class[cls] = scores

    return score_per_class

scores_per_class = compute_score_per_classes(model, worst_classes, classes_dataloaders)


# ## Create Gradient Masks

def compute_masks_per_classes(classes, scores_per_class):
    masks_per_class = {}

    for cls in classes:
        # print(f"Computing Mask for class {cls}")
        mask = create_gradiend_mask(scores_per_class[cls])
        masks_per_class[cls] = mask

    return masks_per_class

masks_per_class = compute_masks_per_classes(worst_classes, scores_per_class)


# Convert the masks to a list, as required by SparseSGDM
def convert_masks_to_list(classes, masks_per_class):
    masks_lists = {}

    for cls in classes:
        # print(f"Computing Mask for class {cls}")
        mask = mask_dict_to_list(masks_per_class[cls])
        masks_lists[cls] = mask

    return masks_lists

masks_list = convert_masks_to_list(worst_classes, masks_per_class)

